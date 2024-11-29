import os
import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn
from torch import Tensor
import transformers
from transformers import AutoTokenizer
from typing import List, Tuple
# import sftp

from data_preprocessing.data import Document, FieldObject, DocumentSpan, SpanSet, read_from_jsonl 
from llms import llm_backend, llm_span
from encoder import Vectorizer

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    OPENAI_API_KEY = api_key
else:
    OPENAI_API_KEY = None
    print("Warning: OPENAI_API_KEY is not set")


def main():
    test_set = read_from_jsonl('../data/test.jsonl')
    # get_span_database(test_set[0])
    back_end = llm_backend.CompletionAPIFactory.get_api(api_name='openai', api_key=OPENAI_API_KEY)
    output = llm_span.get_llm_prediction(d=test_set[40], model='gpt-3.5-turbo', back_end=back_end)
    print("LLM:", output)
    for span in test_set[40].roles:
        print("ano: ", span.argument, ": ", span.textual_span)
    # tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")
    # encoder = Vectorizer()
    # sp = SpanSet(test_set[0], tokenizer=tokenizer, encoder=encoder)



if __name__ == '__main__':
    main()



