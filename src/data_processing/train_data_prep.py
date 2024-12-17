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
    def __init__(self, documents: List[Document], tokenizer: AutoTokenizer, max_seq_length: int = 1024, num_negative: int = 3, save_dir: str = '../data/dataset_v3'):
        """
        Initiating the dataset used for training the encoder on a contrastive loss. Upon initiating, it will 
        automatically save the generated data into a directory, since the process takes a long time and you only
        need one instance of it. (approximate size would be 5GB)
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.save_dir = save_dir

        if self.save_dir is not None and os.path.exists(self.save_dir):
            self.samples = self.load_saved_samples(self.save_dir)

        for doc_id, doc in enumerate(tqdm(documents, desc='document processed'), start=len(self.samples)):
            human_span_set = AnnotationSpanSet(doc, tokenizer)
            llm_span_set = LlmSpanSet(doc, tokenizer)
            doc_tokenized = human_span_set.ids
            doc_attn_mask = human_span_set.attention_mask
            summary_tokenized = llm_span_set.llm_ids
            summary_attn_mask = llm_span_set.llm_attn_mask
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
                    human_span_masks.append(human_span_mask)
                    llm_span_masks.append(correspondent_llm_mask)
                    labels.append(torch.tensor(1, dtype=torch.float))
                    for _ in range(num_negative):
                        negative_human, negative_llm = self.generate_negative_pair(human_span_mask, correspondent_llm_mask)
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
                    'labels': [label.item() for label in labels] 
                }
                if self.save_dir is not None:
                    os.makedirs(self.save_dir, exist_ok=True)
                    sample_id = len(self.samples)
                    save_path = os.path.join(self.save_dir, f"sample_{sample_id}.json")
                    with open(save_path, 'w') as f:
                        json.dump(save_sample, f, indent=2)

                self.samples.append(sample)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.samples[idx]

    def generate_negative_pair(self, 
                               human_mask: Tensor, 
                               llm_mask: Tensor,
                               gen_around_probability=0.3, 
                               move_probability=0.2, 
                               random_gen_probability=0.5, 
                               max_shift=10
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

        for row_idx in range(rows):
            row = modified_tensor[row_idx]

            # generate some 1s around the current span
            for col_idx, val in enumerate(row):
                if val == 1 and random.random() < gen_around_probability:
                    num_new_ones = random.randint(2, 3)
                    for _ in range(num_new_ones):
                        shift = random.randint(0, max_shift)
                        new_idx = col_idx + shift
                        if 0 <= new_idx < cols:
                            row[new_idx] = 1

            # move around the span
            for col_idx, val in enumerate(row):
                if val == 1 and random.random() < move_probability:
                    shift = random.randint(-max_shift, max_shift)
                    new_idx = col_idx + shift
                    if 0 <= new_idx < cols:
                        row[new_idx] = 1
                    row[col_idx] = 0

            # randomly generate a span
            if random.random() < random_gen_probability:
                num_random_ones = random.randint(4, 8)
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
                    sample = json.load(f)
                    # Convert lists back to tensors
                    sample['doc_tokens'] = torch.tensor(sample['doc_tokens'], dtype=torch.long)
                    sample['doc_attn_mask'] = torch.tensor(sample['doc_attn_mask'], dtype=torch.long)
                    sample['summary_tokens'] = torch.tensor(sample['summary_tokens'], dtype=torch.long)
                    sample['summary_attn_mask'] = torch.tensor(sample['summary_attn_mask'], dtype=torch.long)
                    sample['human_span_masks'] = [torch.tensor(mask, dtype=torch.float) for mask in sample['human_span_masks']]
                    sample['llm_span_masks'] = [torch.tensor(mask, dtype=torch.float) for mask in sample['llm_span_masks']]
                    sample['labels'] = torch.tensor(sample['labels'], dtype=torch.float)
                    saved_samples.append(sample)
        return saved_samples


def collate_fn(batch):

    max_length = 1024

    doc_tokens_list = []
    doc_attn_masks_list = []
    summary_tokens_list = []
    summary_attn_masks_list = []

    human_span_masks_list = []
    llm_span_masks_list = []
    labels_list = []
    sample_indices = []

    for sample_idx, sample in enumerate(batch):

        doc_tokens = sample['doc_tokens'].squeeze(0)[:max_length]  # shape: (seq_length_doc,)
        doc_attn_mask = sample['doc_attn_mask'].squeeze(0)[:max_length] 
        summary_tokens = sample['summary_tokens'].squeeze(0)[:max_length] 
        summary_attn_mask = sample['summary_attn_mask'].squeeze(0)[:max_length] 

        doc_tokens_list.append(doc_tokens)
        doc_attn_masks_list.append(doc_attn_mask)
        summary_tokens_list.append(summary_tokens)
        summary_attn_masks_list.append(summary_attn_mask)

        for human_span_mask, llm_span_mask, label in zip(sample['human_span_masks'], sample['llm_span_masks'], sample['labels']):
            human_span_masks_list.append(human_span_mask.squeeze(0)[:max_length])  # shape: (seq_length_doc,)
            llm_span_masks_list.append(llm_span_mask.squeeze(0)[:max_length])      # shape: (seq_length_summary,)
            labels_list.append(label)
            sample_indices.append(sample_idx)

    doc_tokens_tensor = pad_sequence(doc_tokens_list, batch_first=True, padding_value=0)
    doc_attn_masks_tensor = pad_sequence(doc_attn_masks_list, batch_first=True, padding_value=0)

    summary_tokens_tensor = pad_sequence(summary_tokens_list, batch_first=True, padding_value=0)
    summary_attn_masks_tensor = pad_sequence(summary_attn_masks_list, batch_first=True, padding_value=0)

    human_span_masks_list = [mask for mask in human_span_masks_list]
    human_span_masks_tensor = torch.stack(human_span_masks_list)

    llm_span_masks_list = [mask for mask in llm_span_masks_list]
    llm_span_masks_tensor = torch.stack(llm_span_masks_list)

    labels_tensor = torch.tensor(labels_list, dtype=torch.float)
    sample_indices_tensor = torch.tensor(sample_indices, dtype=torch.long)

    return {
        'doc_tokens': doc_tokens_tensor,
        'doc_attn_mask': doc_attn_masks_tensor,
        'summary_tokens': summary_tokens_tensor,
        'summary_attn_mask': summary_attn_masks_tensor,
        'human_span_masks': human_span_masks_tensor,
        'llm_span_masks': llm_span_masks_tensor,
        'labels': labels_tensor,
        'sample_indices': sample_indices_tensor,
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
    save_dir = '../data/dataset_v3'
    dataset = SpanPairDataset(famus, tokenizer, save_dir)


if __name__ == "__main__":
    main()
