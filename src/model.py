import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class Vectorizer(torch.nn.Module):
    def __init__(self, pretrained_model: str = 'SpanBERT/spanbert-base-cased', project_dim: int = 128):
        super(Vectorizer, self).__init__()
        self.encoder = BertModel.from_pretrained(pretrained_model)
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.encoder.config.hidden_size, project_dim)
        )

    def forward(self, input_ids, attention_mask=None, span_masks=None):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            attention_mask: Tensor of shape (batch_size, seq_length)
            span_mask: Tensor of shape (batch_size, seq_length)
                This mask could be used to indicate which tokens belong to the span of interest.
        Returns:
            normalized: Tensor of shape (batch_size, project_dim)
                the normalized, vectorized embedding for learning
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        outputs = []
        for mask in span_masks:
            span_mask = mask.unsqueeze(-1).expand(sequence_output.size())
            masked_output = sequence_output * span_mask
            # mean pooling the span embbedings
            span_embedding = masked_output.sum(dim=1)
            span_lengths = span_mask.sum(dim=1).clamp(min=1e-9)
            span_embedding = span_embedding / span_lengths
            # projecting to vector bank's dimension
            projected = self.projection_head(span_embedding)
            normalized = F.normalize(projected, p=2, dim=1)
            outputs.append(normalized)

        return outputs


