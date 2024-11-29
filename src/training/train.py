import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoModel, AutoTokenizer

from model import Vectorizer
from data_preprocessing import SpanPairDataset, collate_fn


def train_model(train_loader,
                pretrained_model='SpanBERT/spanbert-base-cased',
                project_dim=128,
                device='cuda',
                num_epochs=1,
                lr=0.005,
                lr_scheduler=None,
                batch_size=None,
                ):
    model = Vectorizer(pretrained_model=pretrained_model,
                       project_dim=project_dim)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            # Move tensors to the device
            doc_tokens = batch['doc_tokens'].to(device)
            doc_attn_mask = batch['doc_attn_mask'].to(device)
            summary_tokens = batch['summary_tokens'].to(device)
            summary_attn_mask = batch['summary_attn_mask'].to(device)
            human_span_masks = batch['human_span_masks'].to(device)
            llm_span_masks = batch['llm_span_masks'].to(device)
            labels = batch['labels'].to(device)
            doc_indices = batch['doc_indices'].to(device)
            summary_indices = batch['summary_indices'].to(device)

            # Encode documents and summaries
            doc_outputs = model.encoder(doc_tokens, attention_mask=doc_attn_mask)
            doc_hidden_states = doc_outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
            summary_outputs = model.encoder(summary_tokens, attention_mask=summary_attn_mask)
            summary_hidden_states = summary_outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

            # Gather the hidden states for the spans using indices
            human_hidden_states = doc_hidden_states[doc_indices]  # (total_spans, seq_length, hidden_size)
            llm_hidden_states = summary_hidden_states[summary_indices]  # (total_spans, seq_length, hidden_size)

            # Expand span masks
            human_span_masks_expanded = human_span_masks.unsqueeze(-1)  # (total_spans, seq_length, 1)
            llm_span_masks_expanded = llm_span_masks.unsqueeze(-1)  # (total_spans, seq_length, 1)

            # Extract span embeddings
            human_span_embeddings = human_hidden_states * human_span_masks_expanded
            llm_span_embeddings = llm_hidden_states * llm_span_masks_expanded

            # Sum over sequence length to get span representations
            human_span_sums = human_span_embeddings.sum(dim=1)  # (total_spans, hidden_size)
            llm_span_sums = llm_span_embeddings.sum(dim=1)  # (total_spans, hidden_size)

            # Compute span lengths to average the embeddings
            human_span_lengths = human_span_masks.sum(dim=1).unsqueeze(-1) + 1e-8  # Avoid division by zero
            llm_span_lengths = llm_span_masks.sum(dim=1).unsqueeze(-1) + 1e-8

            # Compute mean span representations
            human_span_reps = human_span_sums / human_span_lengths  # (total_spans, hidden_size)
            llm_span_reps = llm_span_sums / llm_span_lengths  # (total_spans, hidden_size)

            # Compute similarity and loss
            similarity = torch.cosine_similarity(human_span_reps, llm_span_reps, dim=-1)  # (total_spans,)
            loss = torch.nn.functional.mse_loss(similarity, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Span Encoder Model')

    parser.add_argument(
        '--model_name', type=str, default='/longformer-base-4096',
        help='Pre-trained model name or path'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=2e-5,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay (L2 regularization) factor'
    )
    parser.add_argument(
        '--max_seq_length', type=int, default=4096,
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
        '--output_dir', type=str, default='../data/model_output',
        help='Directory to save the trained model and checkpoints'
    )

    args = parser.parse_args()
    return args


def main():
    loaded_samples = torch.load('span_pair_dataset.pt')
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    loaded_dataset = SpanPairDataset(documents=[], tokenizer=tokenizer, max_seq_length=4096)
    loaded_dataset.samples = loaded_samples 
    args = parse_args()
    train_loader = DataLoader(loaded_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn
                              )
    train_model(train_loader=train_loader,
                pretrained_model=args.model_name,
                project_dim=args.project_dim,
                device='cuda',
                num_epochs=args.num_epochs,
                lr=args.lr,
                lr_scheduler=args.lr_scheduler,
                batch_size=args.batch_size
                )


if __name__ == '__main__':
    main()
