import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
import wandb
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

import src.new_model
from src.new_model import SpanRanker
import src.data_processing.new_train_data_prep
from src.data_processing.new_train_data_prep import SpanDataset


BAD_EXAMPLES = {302, 448, 552, 553, 954, 1044, 1046, 1458, 1702, 1703, 1760, 2094, 2232, 2289, 2511, 2518}


def prepare_datasets(documents, num_workers, tokenizer, load_dir, batch_size=2):
    """Prepare training, validation, and test datasets and dataloaders"""

    # Create or load the dataset
    full_dataset = SpanDataset(
        documents=None,
        tokenizer=tokenizer,
        max_seq_length=1024,
        num_negative=1,
        load_dir=load_dir,
    )

    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Create a generator for reproducibility
    generator = torch.Generator().manual_seed(42)

    # First split into train and temp datasets
    train_dataset, temp_dataset = random_split(
        full_dataset, 
        [train_size, val_size + test_size],
        generator=generator
    )

    # Then split the temp dataset into validation and test datasets
    val_dataset, test_dataset = random_split(
        temp_dataset,
        [val_size, test_size],
        generator=generator
    )

    print(f"Dataset split complete: {train_size} training examples, {val_size} validation examples, {test_size} test examples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_model(train_loader,
                eval_loader,
                model,
                device,
                num_epochs,
                lr,
                lr_scheduler,
                weight_decay,
                batch_size,
                output_dir='../../data/checkpoints/'
                ):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler:
        scheduler = StepLR(
            optimizer, 
            step_size=300,
            gamma=0.1,
        )

    best_eval_loss = float('inf')
    best_model_path = None
    global_step = 0 
    accumulation_steps = 4

    max_grad_norm = 1.0

    epoch_grad_norms = []
    # epoch_embedding_stats = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        # scaler = GradScaler(
        #     init_scale=2**8, 
        #     growth_factor=1.5, 
        #     backoff_factor=0.3, 
        #     growth_interval=1000,
        # )
        with tqdm(train_loader, desc=f'{epoch+1}/{num_epochs}', unit="batch") as pbar:
            for step, batch in enumerate(pbar):

                optimizer.zero_grad()
                # with autocast():
                logits = model(batch)
                labels = batch['label'].float().view(-1, 1).to(device)
                loss_fn = torch.nn.BCELoss()
                loss = loss_fn(logits, labels)
                loss = loss / accumulation_steps

                # scaler.scale(loss).backward()
                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    # scaler.unscale_(optimizer)

                    if step % 100 == 0:
                        print("\nGradient norms before clipping:")
                        grad_norms = {}
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                grad_norms[name] = grad_norm
                                print(f"{name}: {grad_norm:.4f}")
                        epoch_grad_norms.append(grad_norms)

                    # Gradient clipping
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-1.0, 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # scaler.step(optimizer)
                    optimizer.step()
                    # scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "Step Loss": f"{loss.item() * accumulation_steps:.4f}",
                        "LR": f"{current_lr:.6f}"
                    })

                step_loss = loss.item() * accumulation_steps
                epoch_loss += step_loss
                global_step += 1
                wandb.log({"train_step_loss": step_loss}, step=global_step)

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        wandb.log({"train_loss": avg_train_loss}, step=epoch)

        if eval_loader is not None:
            eval_loss, accuracy = evaluate(eval_loader, None, model, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {accuracy:.4f}")

            wandb.log({"val_loss": eval_loss, "val_accuracy": accuracy}, step=epoch)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                os.makedirs(output_dir, exist_ok=True)
                best_model_path = os.path.join(output_dir, f"best_model_epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")

    return best_model_path


def evaluate(eval_loader, checkpoint_path, model, device='cuda', similarity_threshold=0.7):
    """
    Evaluate the model on the validation/test set and compute loss and accuracy.
    Args:
        eval_loader: DataLoader for the evaluation set.
        checkpoint_path: The path that stores the saved best model, if None, use the model directly.
        model: The model to evaluate.
        device: The device to run evaluation on ('cuda' or 'cpu').
        similarity_threshold: Threshold to classify similarity scores.
    Returns:
        avg_loss: Average loss over the evaluation set.
        accuracy: Accuracy of predictions based on similarity threshold.
    """
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("model loaded from previous checkpoint")
    model.to(device)
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        # for batch in tqdm(eval_loader, desc='evaluating'):
        for batch in eval_loader:
            logits = model(batch)
            labels = batch['label'].float().view(-1, 1)

            # Compute loss and predictions
            batch_correct = 0
            batch_total = 0
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(logits, labels)
            # print('current_loss', loss.item())
            # print(sims.shape)

            total_loss += loss.item() 
            total_correct += batch_correct
            total_samples += batch_total

    avg_loss = total_loss / len(eval_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def find_best_threshold(eval_loader, checkpoint_path, model, device='cuda'):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("model loaded from previous checkpoint")
    model.to(device)
    model.eval()

    stored_similarities = []
    stored_labels = []

    with torch.no_grad():
        # for batch in tqdm(eval_loader, desc='evaluating'):
        for batch in eval_loader:

            logits = model(batch)
            labels = batch['label'].float().view(-1, 1)

            for idx, similarity in enumerate(logits[0]):
                # prob = torch.sigmoid(similarity)
                stored_similarities.append(similarity.cpu().item())
                stored_labels.append(labels[idx].cpu().item())

        # fpr, tpr, thresholds = roc_curve(stored_labels, stored_similarities)
        # optimal_idx = (tpr - fpr).argmax()
        # optimal_threshold = thresholds[optimal_idx]
        # print("optimal_threshold: ", optimal_threshold)
        auc = roc_auc_score(stored_labels, stored_similarities)
        print(f"AUROC: {auc}")
        # correlation_matrix = np.corrcoef(stored_similarities, stored_labels)

        # correlation = correlation_matrix[0, 1]

        # print(f"Pearson Correlation: {correlation}")

    return 


def parse_args():
    parser = argparse.ArgumentParser(description='Train Span Encoder Model')

    parser.add_argument(
        '--model_name', type=str, default='allenai/longformer-base-4096',
        help='Pre-trained model name or path'
    )
    parser.add_argument(
        '--project_dim', type=int, default=256,
        help='The projection head used in the model for the outputs'
    )
    parser.add_argument(
        '--dropout', type=int, default=0.2,
        help='Dropout for the projection layer.'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size', type=int, default=2,
        help='Training batch size'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=1,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate for the optimizer'
    )

    parser.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay (L2 regularization) factor'
    )
    parser.add_argument(
        '--max_seq_length', type=int, default=1024,
        help='Maximum sequence length for tokenization'
    )

    # Device configuration
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run the training on ("cuda" or "cpu")'
    )

    # Miscellaneous parameters
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of subprocesses for data loading'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--data_dir', type=str, default='/data/cjin/retrieval-augmented-event-extraction/data/dataset_v4',
        help='Directory to the dataset'
    )
    parser.add_argument(
        '--output_dir', type=str, default='/data/cjin/retrieval-augmented-event-extraction/data/checkpoints',
        help='Directory to save the trained model and checkpoints'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    wandb.init(
        project="RAEE",
        name="exp_5", 
        config={             
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
        }
    )
    model = SpanRanker(
        max_length=4096, 
        encoder_out_dim=1024 * 2,
        project_dim=args.project_dim,
        num_layers=2,
        dropout=args.dropout,
    )
    # global_attention_mask = torch.zeros_like(input_ids)
    # global_attention_mask[:, 0] = 1  # Set global attention on the [CLS] token
    # loaded_samples = torch.load('span_pair_dataset.pt')
    tokenizer = AutoTokenizer.from_pretrained('/data/cjin/stella_en_400M_v5')

    train_loader, dev_loader, test_loader = prepare_datasets(
        documents=None,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
        load_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model_path = train_model(
        train_loader=train_loader,
        eval_loader=dev_loader,
        model=model,
        device='cuda',
        num_epochs=args.num_epochs,
        lr=args.lr,
        lr_scheduler=True,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir
    )

    print(f"model trained and stored to {model_path}")

    # model_path = os.path.join(args.output_dir, 'best_model_epoch_1.pt')
    find_best_threshold(test_loader, model_path, model)


if __name__ == '__main__':
    main()
