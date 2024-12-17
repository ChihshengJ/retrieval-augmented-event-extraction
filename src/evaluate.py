import os
import tqdm
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn
from torch import Tensor
import transformers
from transformers import AutoTokenizer
from typing import List, Tuple
import pprint
from sftp import SpanPredictor

from data_preprocessing.data import Document, InputDocument, DocumentSpan, SpanSet, read_from_jsonl
from llms.llm_backend import CompletionAPIFactory
from llms.llm_span import get_llm_prediction, parse_llm_prediction, write_llm_spans_into_docs
from model import Vectorizer
from ceaf_ree import score


def get_socres(test_set: List[Document], soft_score: bool, encoder: Vectorizer):
    pass




def load_test_set(file_path):
    pass


def main():
    test_set = load_test_set('../data/test.jsonl')
    # load_llm_generation('../data/llm_naive_prediction_for_all_documents_v2.json')
    paths = ['../data/train.jsonl', '../data/dev.jsonl', '../data/test.jsonl']
    famus = []
    length = 0
    for path in paths:
        famus.extend(read_from_jsonl(path))
        length = len(famus) - length
        print(length)
    write_llm_spans_into_docs(famus, '../data/llm_naive_prediction_for_all_documents_v2.json')
    print(famus[2024].frame)
    print(len(famus[2024].spans))
    print(famus[2024].roles)
    print(famus[2024].llm_roles)
    


if __name__ == "__main__":
    main()
