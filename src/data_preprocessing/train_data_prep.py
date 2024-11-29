import json
import os
import sys
from tqdm import tqdm
from typing import List, Tuple
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from llms.llm_span import get_llm_prediction, parse_llm_prediction, write_llm_spans_into_docs
from llms.llm_backend import CompletionAPIFactory
from .data import Document, AnnotationSpanSet, LlmSpanSet, read_from_jsonl


api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    OPENAI_API_KEY = api_key
else:
    OPENAI_API_KEY = None
    print("Warning: OPENAI_API_KEY is not set")

backend = CompletionAPIFactory.get_api(api_name='openai', api_key=OPENAI_API_KEY)


class SpanPairDataset(Dataset):
    def __init__(self, documents: List[Document], tokenizer: AutoTokenizer, max_seq_length: int = 4096, save_dir: str = '../data/dataset'):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.save_dir = save_dir

        if self.save_dir is not None and os.path.exists(self.save_dir):
            self.samples = self.load_saved_samples(self.save_dir)

        for doc_id, doc in enumerate(tqdm(documents, desc='document processed'), start=len(self.samples)):
            # human_span_set = AnnotationSpanSet(doc, tokenizer)
            # doc_tokenized = human_span_set.ids.squeeze(0)
            # doc_attn_mask = human_span_set.attention_mask.squeeze(0)
            llm_span_set = LlmSpanSet(doc, tokenizer)
            summary_tokenized = llm_span_set.ids.squeeze(0)
            summary_attn_mask = llm_span_set.attention_mask.squeeze(0)

            # print('annotation span masks:')
            # print(human_span_set.mask_arg_map)
            # print('llm span masks:')
            # print(llm_span_set.mask_arg_map)
            human_span_masks = []
            llm_span_masks = []
            labels = []
            for argument, human_span_mask in list(human_span_set.mask_arg_map.items()):
                try:
                    correspondent_llm_mask = llm_span_set.mask_arg_map[argument]
                except KeyError:
                    print(f"LLM prediction of {doc_id} is missing {argument}.")
                    continue
                if correspondent_llm_mask.sum().item() == 0 or human_span_mask.sum().item() == 0:
                    continue
                else:
                    # print(human_span_mask)
                    # print(correspondent_llm_mask)
                    negative_human, negative_llm = self.generate_negative_pair(human_span_mask, correspondent_llm_mask)
                    human_span_masks.append(human_span_mask)
                    llm_span_masks.append(correspondent_llm_mask)
                    labels.append(torch.tensor(1, dtype=torch.float))
                    human_span_masks.append(negative_human)
                    llm_span_masks.append(negative_llm)
                    labels.append(torch.tensor(0, dtype=torch.float))

                sample = {
                    'doc_tokens': doc_tokenized,
                    'doc_attn_mask': doc_attn_mask,
                    'summary_tokens': summary_tokenized,
                    'summary_attn_mask': summary_attn_mask,
                    'human_span_masks': human_span_masks,
                    'llm_span_masks': llm_span_masks,
                    'labels': labels
                }
                save_sample = {
                    'doc_tokens': doc_tokenized.tolist(),
                    'doc_attn_mask': doc_attn_mask.tolist(),
                    'summary_tokens': summary_tokenized.tolist(),
                    'summary_attn_mask': summary_attn_mask.tolist(),
                    'human_span_masks': [mask.tolist() for mask in human_span_masks],
                    'llm_span_masks': [mask.tolist() for mask in llm_span_masks],
                    'labels': [label.item() for label in labels]  # Convert scalar tensor to Python float
                }
                if self.save_dir is not None:
                    os.makedirs(self.save_dir, exist_ok=True)
                    sample_id = len(self.samples)  # Use current number of samples as unique ID
                    save_path = os.path.join(self.save_dir, f"sample_{sample_id}.json")
                    with open(save_path, 'w') as f:
                        json.dump(save_sample, f, indent=2)

                # Append to in-memory samples
                self.samples.append(sample)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.samples[idx]

    def generate_negative_pair(self, 
                               human_mask: Tensor, 
                               llm_mask: Tensor,
                               gen_around_probability=0.5, 
                               move_probability=0.3, 
                               random_gen_probability=0.2, 
                               max_shift=4
                               ):
        """
        Modify a tensor by generating or moving `1`s randomly.

        args:
            tensor (torch.Tensor): Input 2D tensor.
            pass tensor: no modification.
            gen_around_probability (float): Probability of generating new `1`s around existing `1`s.
            move_probability (float): Probability of moving existing `1`s to nearby locations.
            random_gen_probability (float): Probability of generating completely random `1`s.
            max_shift (int): Maximum number of positions to shift or generate nearby `1`s.

        returns:
            tensor, llm_mask (no need to adjust the llm_mask itself)
        """
        modified_tensor = torch.as_tensor(human_mask).int()
        rows, cols = modified_tensor.shape

        # Iterate over the rows of the tensor
        for row_idx in range(rows):
            row = modified_tensor[row_idx]
            # Generate new 1s around existing 1s
            for col_idx, val in enumerate(row):
                if val == 1 and random.random() < gen_around_probability:
                    num_new_ones = random.randint(2, 3)  # Randomly generate 2-3 new 1s
                    for _ in range(num_new_ones):
                        shift = random.randint(-max_shift, max_shift)
                        new_idx = col_idx + shift
                        if 0 <= new_idx < cols:
                            row[new_idx] = 1

            # Move existing 1s
            for col_idx, val in enumerate(row):
                if val == 1 and random.random() < move_probability:
                    shift = random.randint(-max_shift, max_shift)
                    new_idx = col_idx + shift
                    if 0 <= new_idx < cols:
                        row[new_idx] = 1
                    row[col_idx] = 0  # Clear the original position

            # Add random 1s
            if random.random() < random_gen_probability:
                num_random_ones = random.randint(2, 5)  # Generate 2-5 random 1s
                for _ in range(num_random_ones):
                    random_idx = random.randint(0, cols - 1)
                    row[random_idx] = 1

        return (modified_tensor, llm_mask)

    def load_saved_samples(self, save_dir):
        """Load samples from JSON files in the save directory."""
        saved_samples = []
        for filename in sorted(os.listdir(save_dir)):
            if filename.endswith('.json'):
                filepath = os.path.join(save_dir, filename)
                with open(filepath, 'r') as f:
                    saved_samples.append(json.load(f))
        return saved_samples


def collate_fn(batch, tokenizer):
    # Collect the documents and summaries
    doc_tokens = [item['doc_tokens'] for item in batch]
    doc_attn_masks = [item['doc_attn_mask'] for item in batch]
    summary_tokens = [item['summary_tokens'] for item in batch]
    summary_attn_masks = [item['summary_attn_mask'] for item in batch]

    # Pad the documents and summaries
    doc_tokens_padded = pad_sequence(doc_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    doc_attn_masks_padded = pad_sequence(doc_attn_masks, batch_first=True, padding_value=0)
    summary_tokens_padded = pad_sequence(summary_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    summary_attn_masks_padded = pad_sequence(summary_attn_masks, batch_first=True, padding_value=0)

    # Handle the span masks and labels
    # Since the number of spans varies per sample, we'll flatten them across the batch
    human_span_masks_list = []
    llm_span_masks_list = []
    labels_list = []
    doc_indices = []
    summary_indices = []

    for idx, item in enumerate(batch):
        num_spans = len(item['human_span_masks'])
        human_span_masks_list.extend(item['human_span_masks'])
        llm_span_masks_list.extend(item['llm_span_masks'])
        labels_list.extend(item['labels'])
        # Keep track of which document and summary these spans belong to
        doc_indices.extend([idx] * num_spans)
        summary_indices.extend([idx] * num_spans)

    # Pad the human and LLM span masks
    max_doc_seq_length = doc_tokens_padded.size(1)
    max_summary_seq_length = summary_tokens_padded.size(1)

    # Pad human span masks
    human_span_masks_padded = [torch.nn.functional.pad(
        mask,
        (0, max_doc_seq_length - mask.size(0)),
        value=0
    ) for mask in human_span_masks_list]
    human_span_masks_tensor = torch.stack(human_span_masks_padded)

    # Pad LLM span masks
    llm_span_masks_padded = [torch.nn.functional.pad(
        mask,
        (0, max_summary_seq_length - mask.size(0)),
        value=0
    ) for mask in llm_span_masks_list]
    llm_span_masks_tensor = torch.stack(llm_span_masks_padded)

    # Stack labels
    labels_tensor = torch.stack(labels_list).float()

    # Convert indices to tensors
    doc_indices_tensor = torch.tensor(doc_indices, dtype=torch.long)
    summary_indices_tensor = torch.tensor(summary_indices, dtype=torch.long)

    return {
        'doc_tokens': doc_tokens_padded,
        'doc_attn_mask': doc_attn_masks_padded,
        'summary_tokens': summary_tokens_padded,
        'summary_attn_mask': summary_attn_masks_padded,
        'human_span_masks': human_span_masks_tensor,
        'llm_span_masks': llm_span_masks_tensor,
        'labels': labels_tensor,
        'doc_indices': doc_indices_tensor,
        'summary_indices': summary_indices_tensor,
    }


def llm_training_data_dump(dataset: List[Document], model, back_end, max_tokens):
    output_file = "../data/llm_naive_prediction_for_all_documents_v2.json"
    temp_file = "../data/llm_naive_prediction_temp.json"
    with open(temp_file, 'w') as f:
        f.write('[')
        for idx, document in enumerate(tqdm(dataset, desc='processing documents')):
            output_dict = parse_llm_prediction(get_llm_prediction(document, model, back_end))
            output_dict.update({'instance_id': document.instance_id})
            json.dump(output_dict, f, indent=4)
            if idx < len(dataset) - 1:
                f.write(",\n")
        f.write(']')

    os.replace(temp_file, output_file)
    print(f"Processed data saved to {output_file}")


def main():
    paths = ['../data/train.jsonl', '../data/dev.jsonl', '../data/test.jsonl']
    famus = []
    for path in paths:
        famus.extend(read_from_jsonl(path))
    print(len(famus))
    # llm_training_data_dump(famus, 'gpt-4o', backend, 500)
    llm_doc_path = '../data/llm_naive_prediction_for_all_documents_v2.json' 
    write_llm_spans_into_docs(famus, llm_doc_path)
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    save_dir = '../data/dataset'
    dataset = SpanPairDataset(famus, tokenizer, save_dir)


if __name__ == "__main__":
    main()
