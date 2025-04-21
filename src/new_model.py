import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel


DEFAULT_DOCUMENT_LENGTH = 4096


class SpanRanker(nn.Module):
    def __init__(
        self,
        max_length: int = DEFAULT_DOCUMENT_LENGTH,
        encoder_out_dim: int = 1024 * 2,
        project_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        ckpt: str = None,
    ):
        super().__init__()

        self.max_length = max_length

        # Encoder
        vector_dim = encoder_out_dim / 2
        model_dir = "data/cjin/stella_en_400M_v5"
        vector_linear_dir = f"2_Dense_{vector_dim}"
        self.encoder = AutoModel.from_pretrained(
            model_dir,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.encoder.requires_grad_(False)

        vector_linear = torch.nn.Linear(in_features=self.encoder.config.hidden_size, out_features=vector_dim)
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(model_dir, f"{vector_linear_dir}/pytorch_model.bin")).items()
        }
        vector_linear.load_state_dict(vector_linear_dict)

        # MLP layers
        if ckpt:
            # TODO: implement read from checkpoint
            pass
        else:
            self.mlp = nn.Sequential()
            in_features = encoder_out_dim
            for _ in range(num_layers):
                self.mlp.append(nn.Linear(in_features, project_dim))
                self.mlp.append(nn.LayerNorm(project_dim))
                self.mlp.append(nn.GELU())
                self.mlp.append(nn.Dropout(dropout))
                in_features = project_dim

            self.mlp.append(nn.Linear(project_dim, 1))
            self.mlp.append(nn.Sigmoid())

            self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch):
        """
        Forward pass for the SpanRanker model.

        Args:
            batch: Dictionary containing:
                - doc_tokens: Document token IDs [batch_size, seq_len]
                - doc_attn_mask: Document attention mask [batch_size, seq_len]
                - summary_tokens: Summary token IDs [batch_size, seq_len]
                - summary_attn_mask: Summary attention mask [batch_size, seq_len]
                - human_span_mask: Binary mask for human spans in document [batch_size, seq_len]
                - llm_span_mask: Binary mask for LLM spans in summary [batch_size, seq_len]
                - label: Binary similarity label [batch_size]

        Returns:
            Dictionary containing:
                - logits: Similarity scores [batch_size, 1]
                - loss: Binary cross-entropy loss (if labels provided)
        """

        device = next(self.parameters()).device

        # Extract inputs from batch
        doc_tokens = batch['doc_tokens'].to(device)
        doc_attn_mask = batch['doc_attn_mask'].to(device)
        summary_tokens = batch['summary_tokens'].to(device)
        summary_attn_mask = batch['summary_attn_mask'].to(device)
        candidate_span_mask = batch['candidate_span_mask'].to(device)
        llm_span_mask = batch['llm_span_mask'].to(device)

        # print("shape of token inputs: ", doc_tokens.shape)
        # print("shape of mask: ", doc_attn_mask.shape)
        # print("shape of span mask: ", human_span_mask.shape)

        # Create global attention mask for Longformer (give global attention to CLS token)
        doc_global_attention_mask = torch.zeros_like(doc_tokens)
        doc_global_attention_mask[:, 0] = 1

        summary_global_attention_mask = torch.zeros_like(summary_tokens)
        summary_global_attention_mask[:, 0] = 1

        # Encode document and extract human span representation
        with torch.no_grad():
            doc_outputs = self.encoder(
                input_ids=doc_tokens,
                attention_mask=doc_attn_mask,
                global_attention_mask=doc_global_attention_mask.to(device)
            )
            doc_embeddings = doc_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

            summary_outputs = self.encoder(
                input_ids=summary_tokens, 
                attention_mask=summary_attn_mask,
                global_attention_mask=summary_global_attention_mask.to(device)
            )
            summary_embeddings = summary_outputs.last_hidden_state

        # L2 normalization
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
        summary_embeddings = F.normalize(summary_embeddings, p=2, dim=1)

        span_mask_doc = candidate_span_mask.unsqueeze(-1) 
        masked_doc_embeddings = doc_embeddings * span_mask_doc 

        # Mean-pooling
        mask_sum_doc = span_mask_doc.sum(dim=1) + 1e-10 
        cand_span_embedding = masked_doc_embeddings.sum(dim=1) / mask_sum_doc  

        span_mask_summary = llm_span_mask.unsqueeze(-1) 
        masked_summary_embeddings = summary_embeddings * span_mask_summary 

        # Mean-pooling
        mask_sum_summary = span_mask_summary.sum(dim=1) + 1e-10  # [batch_size, 1]
        llm_span_embedding = masked_summary_embeddings.sum(dim=1) / mask_sum_summary  # [batch_size, hidden_size]

        # Combine span representations
        # Concatenation
        combined_embedding = torch.cat([cand_span_embedding, llm_span_embedding], dim=1)  # [batch_size, 2*hidden_size]

        # Element-wise difference
        # span_diff = torch.abs(human_span_embedding - llm_span_embedding)  # [batch_size, hidden_size]

        # Hadamard product (element-wise multiplication)
        # span_product = human_span_embedding * llm_span_embedding  # [batch_size, hidden_size]

        # Combined features
        # combined_embedding = torch.cat([
        #     human_span_embedding,
        #     llm_span_embedding,
        #     torch.abs(human_span_embedding - llm_span_embedding),
        #     human_span_embedding * llm_span_embedding
        # ], dim=1)  # [batch_size, 4*hidden_size]

        # Pass through MLP to get similarity score
        return self.mlp(combined_embedding)  # [batch_size, 1]





