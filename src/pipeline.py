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

from data_preprocessing.data import Document, InputDocument, DocumentSpan, SpanSet
from llms.llm_backend import CompletionAPIFactory
from llms.llm_span import get_llm_prediction, parse_llm_prediction
from model import Vectorizer

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    OPENAI_API_KEY = api_key
else:
    OPENAI_API_KEY = None
    print("Warning: OPENAI_API_KEY is not set")


def load_inputs(file_path: str, predictor: SpanPredictor) -> Document:
    """
    Given a json file with the text, event frame, and trigger, turn it into a Document object
    with all the span masks extracted
    """
    data = None
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line) 
    return InputDocument(data["Document"], data["Event"], data["Trigger"], predictor)


def retrieval_extraction(file_path: str, llm_backend, predictor: SpanPredictor, encoder: Vectorizer):
    test_example = load_inputs('../data/infertest/test_1.json', predictor)
    prediction = parse_llm_prediction(get_llm_prediction(test_example, model='gpt-4o', back_end=llm_backend))


    return None 


def main():
    predictor = None
    predictor = SpanPredictor.from_path(
        '/data/svashishtha/spanfinder/model/model.tar.gz',
        cuda_device=1,
    )
    llm_backend = CompletionAPIFactory.get_api(api_name='openai', api_key=OPENAI_API_KEY)
    # prediction = llm_span.get_llm_prediction(d=test_exmaple, model='gpt-4o', back_end=back_end)

    


if __name__ == '__main__':
    main()



