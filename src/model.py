import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerModel


class Vectorizer(nn.Module):
    def __init__(self, pretrained_model: str = 'allenai/longformer-base-4096', dropout: int = 0.02, project_dim: int = 256):
        super.__init__()
        self.encoder = LongformerModel.from_pretrained(pretrained_model)
        self.projection_head = nn.Sequential(
            nn.LayerNorm(self.encoder.config.hidden_size),
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.encoder.config.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.config.hidden_size, project_dim)
        )
        for m in self.projection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.005)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.span_pooling = nn.Linear(self.encoder.config.hidden_size, 1, bias=False)
        nn.init.xavier_uniform_(self.span_pooling.weight, gain=0.005)

    def forward(
        self,
        doc_input_ids,
        doc_attention_mask,
        summary_input_ids,
        summary_attention_mask,
        og_span_masks,
        llm_span_masks,
        sample_indices,
    ):
        """
        Args:
            doc_input_ids: Tensor of shape (batch_size, seq_length_doc)
            doc_attention_mask: Tensor of shape (batch_size, seq_length_doc)
            summary_input_ids: Tensor of shape (batch_size, seq_length_summary)
            summary_attention_mask: Tensor of shape (batch_size, seq_length_summary)
            og_span_masks: Tensor of shape (total_human_spans, seq_length_doc)
            llm_span_masks: Tensor of shape (total_llm_spans, seq_length_summary)
            sample_indices: Tensor of shape (total_human_spans,)
        Returns:
            human_outputs: Tensor of shape (total_human_spans, project_dim)
            llm_outputs: Tensor of shape (total_llm_spans, project_dim)
        """
        # Encode documents
        doc_outputs = self.encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
        doc_sequence_output = doc_outputs.last_hidden_state  # Shape: (batch_size, max_seq_length_doc, hidden_size)

        # Encode summaries
        summary_outputs = self.encoder(input_ids=summary_input_ids, attention_mask=summary_attention_mask)
        summary_sequence_output = summary_outputs.last_hidden_state  # Shape: (batch_size, max_seq_length_summary, hidden_size)

        # Gather the sequence outputs for each span pair
        human_sequence_output = doc_sequence_output[sample_indices]  # Shape: (total_spans, max_seq_length_doc, hidden_size)
        llm_sequence_output = summary_sequence_output[sample_indices]  # Shape: (total_spans, max_seq_length_summary, hidden_size)

        # Process human spans
        human_span_embeddings = self._process_spans(human_sequence_output, og_span_masks)

        # Process LLM spans
        llm_span_embeddings = self._process_spans(llm_sequence_output, llm_span_masks)

        return human_span_embeddings, llm_span_embeddings

    def _process_spans(self, sequence_output, span_masks):
        """
        Args:
            sequence_output: Tensor of shape (batch_size, seq_length, hidden_size)
            span_masks: Tensor of shape (total_spans, seq_length)
        Returns:
            normalized: Tensor of shape (total_spans, project_dim)
        """
        # span_lengths = span_masks.sum(dim=1).clamp(min=1e-9).unsqueeze(-1)
        # span_masks = span_masks.unsqueeze(-1)  # Shape: (total_spans, seq_length, 1)
        # span_masks = span_masks.expand(-1, -1, sequence_output.size(-1))  # Shape: (total_spans, seq_length, hidden_size)

        # masked_output = sequence_output * span_masks.float() 
        scores = self.span_pooling(sequence_output).squeeze(-1)
        scores = torch.clamp(scores, min=-100, max=100)

        # scores = scores.masked_fill(span_masks == 0, float('-inf'))
        attention_mask = (span_masks == 1).float()
        scores = scores * attention_mask + (-1e9) * (1 - attention_mask)
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        span_embedding = (sequence_output * attn_weights).sum(dim=1)
        # span_embedding = masked_output.sum(dim=1) / span_lengths  # Shape: (total_spans, hidden_size)

        # Project and normalize
        projected = self.projection_head(span_embedding)  # Shape: (total_spans, project_dim)
        normalized = F.normalize(projected, p=2, dim=1)  # Shape: (total_spans, project_dim)

        return normalized



