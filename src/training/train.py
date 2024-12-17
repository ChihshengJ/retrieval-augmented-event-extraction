import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

from model import Vectorizer
from data_preprocessing.train_data_prep import SpanPairDataset, collate_fn


BAD_EXAMPLES = {302, 448, 552, 553, 954, 1044, 1046, 1458, 1702, 1703, 1760, 2094, 2232, 2289, 2511, 2518}


class SpanPairDatasetWrapper(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'doc_tokens': sample['doc_tokens'],
            'doc_attn_mask': sample['doc_attn_mask'],
            'summary_tokens': sample['summary_tokens'],
            'summary_attn_mask': sample['summary_attn_mask'],
            'human_span_masks': sample['human_span_masks'],
            'llm_span_masks': sample['llm_span_masks'],
            'labels': sample['labels']
        }


def info_nce_loss(human_span_reps, llm_span_reps, labels, temperature=0.1):
    """
    Compute InfoNCE loss for each (human_rep, llm_rep, label) tuple in the batch.

    Args:
        human_span_reps (torch.Tensor): Representations for human spans (num_pairs, embed_dim).
        llm_span_reps (torch.Tensor): Representations for LLM spans (num_pairs, embed_dim).
        labels (torch.Tensor): Binary labels for similarity (1 for positive, 0 for negative). Shape: (num_pairs,).
        temperature (float): Temperature scaling factor.

    Returns:
        torch.Tensor: Total InfoNCE loss (averaged over all pairs).
    """
    assert human_span_reps.shape == llm_span_reps.shape, "Mismatch in representation dimensions"
    # print(human_span_reps.shape, llm_span_reps.shape, labels.shape)

    N, D = human_span_reps.shape
    similarities = torch.matmul(human_span_reps, llm_span_reps.T)
    sim = similarities / temperature

    similarities_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - similarities_max.detach()

    exp_sim = torch.exp(sim)
    positive_mask = torch.eye(N, dtype=torch.float32).to('cuda') * labels.view(-1, 1)
    numerator = (exp_sim * positive_mask).sum(dim=1)
    denominator = exp_sim.sum(dim=1)
    eps = 1e-9
    loss = -torch.log(numerator / (denominator + eps) + eps)

    valid_loss = loss[labels == 1]
    # print('loss:', valid_loss)

    if valid_loss.numel() == 0:
        return torch.tensor(0.0, requires_grad=True), similarities
    else:
        return valid_loss.mean(), similarities


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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler:
        scheduler = LinearLR(
            optimizer, 
            start_factor=0.1,
            end_factor=1.0,
            total_iters=100)

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
        scaler = GradScaler(
            init_scale=2**8, 
            growth_factor=1.5, 
            backoff_factor=0.3, 
            growth_interval=1000,
        )
        with tqdm(train_loader, desc=f'{epoch+1}/{num_epochs}', unit="batch") as pbar:
            for step, batch in enumerate(pbar):

                doc_tokens = batch['doc_tokens'].to(device)
                doc_attn_mask = batch['doc_attn_mask'].to(device)
                summary_tokens = batch['summary_tokens'].to(device)
                summary_attn_mask = batch['summary_attn_mask'].to(device)
                human_span_masks = batch['human_span_masks'].to(device)
                llm_span_masks = batch['llm_span_masks'].to(device)
                labels = batch['labels'].to(device)
                sample_indices = batch['sample_indices'].to(device)

                optimizer.zero_grad()
                with autocast():
                    human_span_reps, llm_span_reps = model(
                        doc_input_ids=doc_tokens,
                        doc_attention_mask=doc_attn_mask,
                        summary_input_ids=summary_tokens,
                        summary_attention_mask=summary_attn_mask,
                        og_span_masks=human_span_masks,
                        llm_span_masks=llm_span_masks,
                        sample_indices=sample_indices
                    )
                    loss, _ = info_nce_loss(human_span_reps, llm_span_reps, labels, temperature=0.5)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)

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

                    scaler.step(optimizer)
                    scaler.update()
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
            # Move tensors to the device
            doc_tokens = batch['doc_tokens'].to(device)
            doc_attn_mask = batch['doc_attn_mask'].to(device)
            summary_tokens = batch['summary_tokens'].to(device)
            summary_attn_mask = batch['summary_attn_mask'].to(device)
            human_span_masks = batch['human_span_masks'].to(device)
            llm_span_masks = batch['llm_span_masks'].to(device)
            labels = batch['labels'].to(device)
            sample_indices = batch['sample_indices'].to(device)

            # Forward pass for human and LLM spans
            human_span_reps, llm_span_reps = model(
                doc_input_ids=doc_tokens, 
                doc_attention_mask=doc_attn_mask, 
                summary_input_ids=summary_tokens,
                summary_attention_mask=summary_attn_mask,
                og_span_masks=human_span_masks,
                llm_span_masks=llm_span_masks,
                sample_indices=sample_indices,
            )
            # print(human_span_reps)
            # print(llm_span_reps)
            # Compute loss and predictions
            batch_correct = 0
            batch_total = 0

            loss, sims = info_nce_loss(human_span_reps, llm_span_reps, labels, temperature=0.1)
            # print('current_loss', loss.item())
            # print(sims.shape)

            for idx, similarity in enumerate(sims[0]):
                print("similarity:", similarity)
                prob = torch.sigmoid(similarity)
                predicted = (prob > similarity_threshold).float() 
                # print(prob, labels[idx])
                batch_correct += (predicted == labels[idx]).sum().item()
                batch_total += 1

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
            # Move tensors to the device
            doc_tokens = batch['doc_tokens'].to(device)
            doc_attn_mask = batch['doc_attn_mask'].to(device)
            summary_tokens = batch['summary_tokens'].to(device)
            summary_attn_mask = batch['summary_attn_mask'].to(device)
            human_span_masks = batch['human_span_masks'].to(device)
            llm_span_masks = batch['llm_span_masks'].to(device)
            labels = batch['labels'].to(device)
            sample_indices = batch['sample_indices'].to(device)

            # Forward pass for human and LLM spans
            human_span_reps, llm_span_reps = model(
                doc_input_ids=doc_tokens, 
                doc_attention_mask=doc_attn_mask, 
                summary_input_ids=summary_tokens,
                summary_attention_mask=summary_attn_mask,
                og_span_masks=human_span_masks,
                llm_span_masks=llm_span_masks,
                sample_indices=sample_indices,
            )

            _, sims = info_nce_loss(human_span_reps, llm_span_reps, labels, temperature=0.1)

            for idx, similarity in enumerate(sims[0]):
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


def split_dataset(samples, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """Split samples into train, dev, and test sets."""
    filtered_samples = [sample for idx, sample in enumerate(samples) if idx not in BAD_EXAMPLES]
    total_samples = len(filtered_samples)
    train_size = int(total_samples * train_ratio)
    dev_size = int(total_samples * dev_ratio)
    test_size = total_samples - train_size - dev_size
    return random_split(filtered_samples, [train_size, dev_size, test_size])


def create_dataloaders(dataset, batch_size=1, num_workers=4):
    """Create DataLoader for a dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)


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
        '--dropout', type=int, default=0.1,
        help='Dropout for the projection layer.'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Training batch size'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=1,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
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
        '--data_dir', type=str, default='../data/dataset_v3',
        help='Directory to the dataset'
    )
    parser.add_argument(
        '--output_dir', type=str, default='../data/checkpoints',
        help='Directory to save the trained model and checkpoints'
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    wandb.init(
        project="RAEE",
        name="exp_3", 
        config={             
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
        }
    )
    model = Vectorizer(pretrained_model=args.model_name, dropout=args.dropout, project_dim=args.project_dim)
    # global_attention_mask = torch.zeros_like(input_ids)
    # global_attention_mask[:, 0] = 1  # Set global attention on the [CLS] token
    # loaded_samples = torch.load('span_pair_dataset.pt')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    loaded_dataset = SpanPairDataset(documents=[], tokenizer=tokenizer, max_seq_length=args.max_seq_length, save_dir=args.data_dir)
    samples = loaded_dataset.samples

    train_samples, dev_samples, test_samples = split_dataset(samples)
    train_dataset = SpanPairDatasetWrapper(train_samples)
    dev_dataset = SpanPairDatasetWrapper(dev_samples)
    test_dataset = SpanPairDatasetWrapper(test_samples)

    train_loader = create_dataloaders(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    dev_loader = create_dataloaders(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = create_dataloaders(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

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

    # model_path = os.path.join(args.output_dir, 'best_model_epoch_1.pt')
    find_best_threshold(test_loader, model_path, model)


if __name__ == '__main__':
    main()
