import json
import os
import sys
import datetime
from tqdm import tqdm
from typing import List, Tuple
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

import src.llms.llm_span
import src.llms.llm_backend
import src.data_processing.data
from src.llms.llm_span import get_llm_predictions, parse_llm_predictions, write_llm_spans_into_docs
from src.llms.llm_backend import CompletionAPIFactory
from src.data_processing.data import Document, CandSpanSet, AnnotationSpanSet, LlmSpanSet, read_from_jsonl


api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    OPENAI_API_KEY = api_key
else:
    OPENAI_API_KEY = None
    print("Warning: OPENAI_API_KEY is not set")

api_key = os.getenv("ANTHROPIC_API_KEY", "")
if api_key != "":
    CLAUDE_API_KEY = api_key
else:
    CLAUDE_API_KEY = None
    print("Warning: CLAUDE_API_KEY is not set")

# backend = CompletionAPIFactory.get_api(api_name='openai', api_key=OPENAI_API_KEY)



class SpanDataset(Dataset):
    def __init__(
        self, 
        documents: List[Document], 
        tokenizer: AutoTokenizer, 
        max_seq_length: int = 4096, 
        num_negative: int = 1, 
        save_dir: str = 'data/dataset_v5',
        load_dir: str = None
    ):
        """
        Initiating the dataset used for training the encoder on a contrastive loss. Upon initiating, it will 
        automatically save the generated data into a directory, since the process takes a long time and you only
        need one instance of it. (approximate size would be 5GB)
        """

        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.save_dir = save_dir

        if load_dir is not None and os.path.exists(load_dir):
            print(f"Loading from specified dataset: {load_dir}")
            self.examples = self.load_saved_samples(load_dir)

            if hasattr(self, 'is_complete_dataset') and self.is_complete_dataset:
                print(f"Loaded complete dataset with {len(self.examples)} examples")
                return

        if self.save_dir is not None and os.path.exists(self.save_dir):
            print(f"Checking for existing dataset in save_dir: {self.save_dir}")
            self.examples = self.load_saved_samples(self.save_dir)

            if hasattr(self, 'is_complete_dataset') and self.is_complete_dataset:
                print(f"Loaded complete dataset with {len(self.examples)} examples")
                return

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        if hasattr(self, 'processed_doc_ids'):
            example_id = self.last_example_id + 1
            print(f"Resuming from example ID: {example_id}")
        else:
            self.processed_doc_ids = set()
            example_id = 0
            print("Starting new dataset creation")

        if self.save_dir is not None:
            dataset_path = os.path.join(save_dir, 'dataset.jsonl')
            self.jsonl_file = open(dataset_path, 'a')

            meta_path = os.path.join(save_dir, 'meta_data.json')
            metadata = {
                'format': 'jsonl',
                'created_at': datetime.datetime.now().isoformat(),
                'max_seq_length': max_seq_length,
                'num_negative': num_negative,
                'is_complete': False,  # Will update to True when done
                'last_updated': datetime.datetime.now().isoformat()
            }

        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    existing_metadata = json.load(f)
                    metadata = {**existing_metadata, **metadata}
            except json.JSONDecodeError:
                pass

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        doc_map = {}
        summary_map = {}

        example_id = len(self.examples)
        for doc_id, doc in enumerate(tqdm(documents, desc='document processed'), start=0):
            human_span_set = AnnotationSpanSet(doc, tokenizer, max_seq_length)
            llm_span_set = LlmSpanSet(doc, tokenizer, max_seq_length)
            span_bank = CandSpanSet(doc, tokenizer, max_seq_length)

            doc_key = f"doc_{doc_id}"
            summary_key = f"summary_{doc_id}"

            doc_data = {
                'tokens': human_span_set.ids.tolist(),
                'attn_mask': human_span_set.attention_mask.tolist()
            }

            summary_data = {
                'tokens': llm_span_set.llm_ids.tolist(),
                'attn_mask': llm_span_set.llm_attn_mask.tolist()
            }

            doc_map[doc_key] = doc_data
            summary_map[summary_key] = summary_data

            with open(os.path.join(save_dir, 'doc_map.jsonl'), 'a') as f:
                f.write(json.dumps(doc_map) + '\n')
                f.flush()

            with open(os.path.join(save_dir, 'summary_map.jsonl'), 'a') as f:
                f.write(json.dumps(summary_map) + '\n')
                f.flush()

            # Create tensor versions for immediate use
            doc_tokenized = human_span_set.ids
            doc_attn_mask = human_span_set.attention_mask
            summary_tokenized = llm_span_set.llm_ids
            summary_attn_mask = llm_span_set.llm_attn_mask

            for role, human_span_mask in human_span_set.mask_arg_map.items():
                try:
                    correspondent_llm_mask = llm_span_set.mask_arg_map[role]
                except KeyError:
                    print(f"LLM prediction of {doc_id} is missing {role}.")
                    continue

                if correspondent_llm_mask.sum().item() == 0 or human_span_mask.sum().item() == 0:
                    continue

                # Create positive example
                positive_sample = {
                    'doc_tokens': doc_tokenized,
                    'doc_attn_mask': doc_attn_mask,
                    'summary_tokens': summary_tokenized,
                    'summary_attn_mask': summary_attn_mask,
                    'candidate_span_mask': human_span_mask,
                    'llm_span_mask': correspondent_llm_mask,
                    'label': torch.tensor(1, dtype=torch.float)
                }

                save_sample = {
                    'id': example_id,
                    'doc_ref': doc_key,
                    'summary_ref': summary_key,
                    'candidate_span_mask': human_span_mask.tolist(),
                    'llm_span_mask': correspondent_llm_mask.tolist(),
                    'label': 1.0,
                    'role': role
                }

                if self.save_dir is not None:
                    self.jsonl_file.write(json.dumps(save_sample) + '\n')
                    self.jsonl_file.flush()

                self.examples.append(positive_sample)
                example_id += 1

                # Create negative examples
                for neg_idx in range(num_negative):
                    negative_span = self.get_negative_example(
                        spans=span_bank, 
                        doc=doc, 
                        strategy='random')

                    negative_sample = {
                        'doc_tokens': doc_tokenized,
                        'doc_attn_mask': doc_attn_mask,
                        'summary_tokens': summary_tokenized,
                        'summary_attn_mask': summary_attn_mask,
                        'candidate_span_mask': negative_span,
                        'llm_span_mask': correspondent_llm_mask,
                        'label': torch.tensor(0, dtype=torch.float)
                    }

                    save_negative = {
                        'id': example_id,
                        'doc_ref': doc_key,
                        'summary_ref': summary_key,
                        'candidate_span_mask': negative_span.tolist(),
                        'llm_span_mask': correspondent_llm_mask.tolist(),
                        'label': 0.0,
                        'role': role,
                        'neg_idx': neg_idx
                    }

                    if self.save_dir is not None:
                        self.jsonl_file.write(json.dumps(save_negative) + '\n')
                        self.jsonl_file.flush()

                    self.examples.append(negative_sample)
                    example_id += 1

            del human_span_set
            del llm_span_set
            del span_bank
            import gc
            gc.collect()

            if self.save_dir is not None and doc_id == len(documents) - 1:
                # Close the JSONL file
                if hasattr(self, 'jsonl_file'):
                    self.jsonl_file.close()

                # Update metadata to mark as complete
                meta_path = os.path.join(save_dir, 'meta_data.json')
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)

                metadata['is_complete'] = True
                metadata['completed_at'] = datetime.datetime.now().isoformat()

                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

        if self.save_dir is not None and hasattr(self, 'jsonl_file'):
            self.jsonl_file.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def _flatten_list(self, nested_list):
        """
        Flatten a nested list if it's nested, otherwise return as is.
        This handles both cases: [[1,2,3]] -> [1,2,3] and [1,2,3] -> [1,2,3]
        """
        # Check if this is a nested list
        if isinstance(nested_list, list) and len(nested_list) > 0 and isinstance(nested_list[0], list):
            # It's nested, so flatten it
            return nested_list[0]
        # Not nested, return as is
        return nested_list

    def get_negative_example(self, spans: CandSpanSet, doc: Document, strategy: str = 'random'):
        """
        return one negative example by selecting a span from the list of spans in the document
        according to an appointed strategy.
        """
        if strategy == 'random':
            return random.choice(spans.span_masks)
        elif strategy == 'marginal':
            # TODO: implement a simple edit distance check and return a list of candidates.
            return None

    def load_saved_samples(self, load_dir):
        """
        Load previously saved samples from a dataset directory.
        This method serves two purposes:
        1. Load a complete dataset for use in training/evaluation
        2. Resume processing by identifying which documents have been processed

        Args:
            load_dir: Directory containing saved dataset files

        Returns:
            List of examples loaded from the saved files
        """
        examples = []
        processed_doc_ids = set()
        last_example_id = -1

        # Check if this is a complete dataset by looking for a completion marker
        meta_path = os.path.join(load_dir, 'meta_data.json')
        is_complete = False
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    is_complete = metadata.get('is_complete', False)
            except json.JSONDecodeError:
                print(f"Warning: Couldn't parse metadata from {meta_path}")

        # Load document and summary maps
        print(f"Loading document and summary maps from {load_dir}...")
        doc_map = self._load_jsonl_to_dict(os.path.join(load_dir, 'doc_map.jsonl'), key_field='id')
        summary_map = self._load_jsonl_to_dict(os.path.join(load_dir, 'summary_map.jsonl'), key_field='id')
        print(f"Loaded {len(doc_map)} documents and {len(summary_map)} summaries")

        # Load dataset entries
        dataset_path = os.path.join(load_dir, 'dataset.jsonl')
        if os.path.exists(dataset_path):
            print(f"Loading dataset entries from {dataset_path}...")
            entry_count = 0
            with open(dataset_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_count += 1

                        doc_ref = entry.get('doc_ref', '')
                        if doc_ref.startswith('doc_'):
                            try:
                                doc_id = int(doc_ref.split('_')[1])
                                processed_doc_ids.add(doc_id)
                            except (IndexError, ValueError):
                                pass

                        if entry.get('id', 0) > last_example_id:
                            last_example_id = entry['id']
                    except json.JSONDecodeError:
                        print("Warning: Couldn't parse entry in dataset file")
                        continue

            print(f"Found {entry_count} entries in dataset file")

        print("Reconstructing examples from dataset entries...")
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        doc_ref = entry.get('doc_ref', '')
                        summary_ref = entry.get('summary_ref', '')

                        if doc_ref not in doc_map or summary_ref not in summary_map:
                            continue

                        doc_data = doc_map[doc_ref]
                        summary_data = summary_map[summary_ref]

                        sample = {
                            'doc_tokens': torch.tensor(doc_data['tokens'], dtype=torch.long),
                            'doc_attn_mask': torch.tensor(doc_data['attn_mask'], dtype=torch.long),
                            'summary_tokens': torch.tensor(summary_data['tokens'], dtype=torch.long),
                            'summary_attn_mask': torch.tensor(summary_data['attn_mask'], dtype=torch.long),
                            'human_span_mask': torch.tensor(entry['human_span_mask'], dtype=torch.float),
                            'llm_span_mask': torch.tensor(entry['llm_span_mask'], dtype=torch.float),
                            'label': torch.tensor(entry['label'], dtype=torch.float)
                        }

                        examples.append(sample)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Couldn't process entry: {e}")
                        continue

        print(f"Successfully loaded {len(examples)} complete examples")
        print(f"Found {len(processed_doc_ids)} processed document IDs")

        self.last_example_id = last_example_id
        self.processed_doc_ids = processed_doc_ids
        self.is_complete_dataset = is_complete

        return examples

    def _load_jsonl_to_dict(self, file_path, key_field='id'):
        """
        Helper method to load JSONL file into a dictionary.

        Args:
            file_path: Path to the JSONL file
            key_field: The field to use as the dictionary key

        Returns:
            Dictionary mapping key_field values to the corresponding entries
        """
        result = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if key_field in entry:
                            result[entry[key_field]] = entry
                    except json.JSONDecodeError:
                        print(f"Warning: Couldn't parse line in {file_path}")

        return result 


def llm_training_data_dump(
        dataset,  
        model, 
        setting,
        file_path,
        backend, 
        max_tokens
):
    # contexts = ['report', 'source']
    contexts = ['train']
    splits = ['dev', 'test']
    for c in contexts:
        for s in splits:
            output_path = os.path.join(file_path, f'{model}_{setting}_{c}_{s}.json')
            if s == 'dev':
                data = dataset[759: 1012]
            else:
                data = dataset[1012:]
            trainset = dataset[:759]

            if os.path.isfile(output_path):
                with open(output_path, 'r') as interrupted_f:
                    prev_content = interrupted_f.read()
                fixed_prev_content = prev_content + "]"
                interrupted_data = json.loads(fixed_prev_content)
                already_collected = len(interrupted_data)
            else:
                already_collected = 0

            if already_collected > 0:
                print(f'{c}, {s}, collected {already_collected} files, restoring.')

            with open(output_path, 'a') as f:
                if already_collected == 0:
                    f.write('[')
                for idx, document in enumerate(tqdm(data[already_collected:], desc=f'processing {s} documents')):
                    output_dict = parse_llm_predictions(get_llm_predictions(trainset, idx, document, model, c, setting, backend))
                    output_dict.update({'instance_id': document.instance_id})
                    json.dump(output_dict, f, indent=4)
                    if idx < len(dataset) - 1:
                        f.write(",\n")
                f.write(']')

            print(f"Processed data saved to {output_path}")


def main():
    paths = [
        '/data/cjin/retrieval-augmented-event-extraction/data/train.jsonl',
        '/data/cjin/retrieval-augmented-event-extraction/data/dev.jsonl',
        '/data/cjin/retrieval-augmented-event-extraction/data/test.jsonl']
    famus = []
    for path in paths:
        famus.extend(read_from_jsonl(path))

    print(len(famus))

    model_name = 'claude-3-5-sonnet-20241022'
    backend = CompletionAPIFactory.get_api(api_name='claude', api_key=CLAUDE_API_KEY)

    llm_doc_path = 'data/llm_draft_results/'
    llm_training_data_dump(
        dataset=famus, 
        model=model_name, 
        setting='FS',
        file_path=llm_doc_path, 
        backend=backend, 
        max_tokens=500,
    )
    # write_llm_spans_into_docs(famus, llm_doc_path)
    # tokenizer = AutoTokenizer.from_pretrained("/data/cjin/stella_en_400M_v5")
    # save_dir = 'data/dataset_v5'
    # dataset = SpanDataset(famus, tokenizer, save_dir)


if __name__ == "__main__":
    main()
