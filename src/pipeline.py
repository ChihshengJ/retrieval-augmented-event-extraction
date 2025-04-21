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
from typing import List, Tuple, Dict
import pprint
from sftp import SpanPredictor

from data_preprocessing.data import Document, InputDocument, DocumentSpan, CandSpanSet, LlmSpanSet
from data_preprocessing.data_utils import frame_to_llm_prompt_info_dct
from llms.llm_backend import CompletionAPIFactory
from llms.llm_span import get_llm_predictions, parse_llm_predictions, write_llm_spans_into_one_doc
from model import Vectorizer
from new_model import SpanRanker


api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    OPENAI_API_KEY = api_key
else:
    OPENAI_API_KEY = None
    print("Warning: OPENAI_API_KEY is not set")


def load_inputs(file_path: str, predictor: SpanPredictor) -> Document:
    """
    Given a json file with the text, event frame, and a trigger, initiate a Document object
    with all the span masks extracted
    """
    data = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
    return InputDocument(data["Document"], data["Event"], data["Trigger"], predictor)


def get_frame_info(d: Document) -> Dict:
    return frame_to_llm_prompt_info_dct(d.frame)


def get_predictions(model, d: Document, info_dict: Dict, pred: Dict, tokenizer: AutoTokenizer, max_seq_length) -> Tuple[List[DocumentSpan], List[float]]:
    span_bank = CandSpanSet(d, tokenizer, max_seq_length)
    llm_pred = LlmSpanSet(d, tokenizer, max_seq_length)
    doc_tokenized = span_bank.ids
    doc_attn_mask = span_bank.attention_mask
    summary_tokenized = llm_pred.llm_ids
    summary_attn_mask = llm_pred.llm_attn_mask

    for role in info_dict['event_roles']:
        for span_mask in span_bank.span_masks:
            batch = {
                'doc_tokens': doc_tokenized,
                'doc_attn_mask': doc_attn_mask,
                'summary_tokens': summary_tokenized,
                'summary_attn_mask': summary_attn_mask,
                'candidate_span_mask': span_mask,
                'llm_pred_mask': llm_pred,
            }
            score = model(batch)



def get_top_candidate(spans: List[DocumentSpan], scores: List[float]) -> DocumentSpan:
    top_candidate = scores.index(max(scores))
    return spans[top_candidate]


def retrieval_extraction(file_path: str, llm_backend, predictor: SpanPredictor, tokenizer: AutoTokenizer, max_seq_length: int):
    """
    Implementation of the pipeline. Given a json file with the text, event frame, 
    and a event trigger, return a predicted span for each role.
    """

    # Initiate a Document object with framenet settings and a span bank
    doc = load_inputs(file_path, predictor)

    # Retrieve_framenet_info_dict
    info_dict = get_frame_info(doc)

    # Get LLM predictions for all the roles
    llm_predictions = parse_llm_predictions(
        get_llm_predictions(
            doc=doc,
            info_dict=info_dict,
            model='gpt-4o',
            back_end=llm_backend
        )
    )

    write_llm_spans_into_one_doc(doc, llm_predictions)

    # Run Model through each role
    model = SpanRanker()
    tokenizer = None
    predictions = rank_spans(doc, info_dict, tokenizer, max_seq_length)

    return predictions


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



