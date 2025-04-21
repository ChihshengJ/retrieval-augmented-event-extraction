import json
from typing import List, Tuple
from dataclasses import dataclass
import torch
from torch import Tensor
from transformers import AutoTokenizer
import sftp
import re
from sftp import SpanPredictor

import src.model
from src.model import Vectorizer
from .data_utils import find_candidate_spans, sentence_token_span_to_doc_spans


class FieldObject:
    """
    A class used to make the values of a jsonl object accessible through attributes, 
    it would automatically replace the hyphens in the keys into underscores.

    """

    def __init__(self, dictionary: dict):
        for k, v in dictionary.items():
            self._original_keys = dictionary.keys()
            safe_key = k.replace('-', '_')
            if isinstance(v, dict):
                v = FieldObject(v)
            elif isinstance(v, list):
                v = [FieldObject(item) if isinstance(v, dict) else item for item in v]
            setattr(self, safe_key, v)

    def __getattr__(self, key):
        original_key = key.replace("_", "-")
        if original_key in self._original_keys:
            return self.__dict__[key]
        raise AttributeError(f"Attribute '{key}' not found")

    def __repr__(self):
        return f"<FieldObject {self.__dict__}>"


@dataclass
class DocumentSpan:
    """
    A class holding a span.
    """
    textual_span: str
    char_start_idx: int
    char_end_idx: int
    token_start_idx: int
    token_end_idx: int
    argument: str

    def __init__(self, span: List):
        if len(span) <= 1:
            self.textual_span = None
            self.char_start_idx = None
            self.char_end_idx = None
            self.token_start_idx = None
            self.token_end_idx = None
            self.argument = span[0]
        else:
            self.textual_span = span[0]
            self.char_start_idx = span[1]
            self.char_end_idx = span[2]
            self.token_start_idx = span[3]
            self.token_end_idx = span[4]
            self.argument = None if len(span) == 5 else span[5]


@dataclass
class Document:
    """
    A generic class for a documentation.
    Attr:
        instance_id: inherent from the FAMuS dataset
        instance_id_raw_lome_predictor: inherent from the FAMuS daraset
        frame: the event frame from the FrameNet
        doctext: the original document
        doctext: pre-tokenized document, a list of strings
        spans: a list of Span objects extracted by Span-finder
        trigger: the event trigger in the original docuemnt as a Span object
        roles: the role annotations of the event against FrameNet
        is_platinum: a Bool value represent whether the annotation is a platinum annotation
    """
    instance_id: str
    instance_id_raw_lome_predictor: str
    frame: str
    report: str
    report_tok: List[str]
    source: str
    source_tok: List[str]
    report_spans: List[DocumentSpan]
    source_spans: List[DocumentSpan]
    trigger: DocumentSpan
    report_roles: List[DocumentSpan]
    source_roles: List[DocumentSpan]
    llm_roles: List[DocumentSpan]
    is_platinum: bool


class InputDocument(Document):
    def __init__(self, d: str, s: str, event: str, trigger: str, predictor: SpanPredictor):
        self.instance_id = None
        self.instance_id_raw_lome_predictor = None
        self.frame = event
        self.report = d
        self.report_tok = None
        self.source = s
        self.source_tok = None
        self.report_spans = [DocumentSpan(span) for span in self.set_spans(self.report, predictor)]
        self.source_spans = [DocumentSpan(span) for span in self.set_spans(self.source, predictor)]
        self.trigger = DocumentSpan([trigger, 0, 0, 0, 0, 0])
        self.report_roles = None
        self.source_roles = None
        self.llm_roles = None
        self.is_platinum = False

    def set_spans(self, t: str, predictor: SpanPredictor):
        sentences = self._split_into_sentences(t)
        tokenized_sentences = [self._naive_tokenizer(sentence) for sentence in sentences]
        all_spans_pre_aligned = find_candidate_spans(tokenized_sentences, predictor, [])
        all_spans = []
        for span in all_spans_pre_aligned:
            all_spans.append(sentence_token_span_to_doc_spans(span, tokenized_sentences))
        return all_spans

    def get_spans(self):
        return self.report_spans, self.source_spans

    def _naive_tokenizer(self, sentence: str) -> List[str]:
        pattern = r"\d+(?:[\.,]\d+)*|[\w]+|[^\s\w]"
        return re.findall(pattern, sentence)

    def _split_into_sentences(self, passage: str) -> List[str]:
        pattern = r'(?<!\b\w)(?<!\b[A-Za-z]\.)(?<!\b[A-Za-z]{2}\.)\s*(?<=[.!?])\s+'
        return re.split(pattern, passage)


class SpanSet:
    def __init__(self, d: Document, tokenizer: AutoTokenizer, max_seq_length: int):
        """
        A class that contains the tokenized spans and span masks for a single document
        """
        self.data = d
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # self.ids, self.attention_mask, self.span_masks = self._set_inputs(d.doctext, d.spans, max_seq_length)

    def _set_inputs(
            self,
            doctext: str,
            spans: List[DocumentSpan],
            max_seq_length: int
    ) -> Tuple[Tensor, List[Tensor]]:
        raise NotImplementedError

    def _get_span_masks(self, s: DocumentSpan, offset_map: Tensor):
        if s.textual_span is None or s.char_start_idx is None:
            return torch.zeros((1, offset_map.shape[1]), dtype=torch.float)
        # print(s.char_start_idx, s.char_end_idx)
        span_mask = torch.zeros((1, offset_map.shape[1]), dtype=torch.float)
        start_idx, end_idx = s.char_start_idx, s.char_end_idx + 1
        real_start_idx, real_end_idx = 0, 0
        for i, token in enumerate(offset_map[0, :-1, :]):
            real_start_idx = i if token[0].item() == start_idx else real_start_idx
            real_end_idx = i if token[1].item() == end_idx else real_end_idx
        span_mask[0, real_start_idx: real_end_idx + 1] = 1
        # print(span_mask[0, :25])
        # print('tokens:', span_mask.sum().item())
        return span_mask


class CandSpanSet(SpanSet):
    def __init__(self, d: Document, tokenizer: AutoTokenizer, max_seq_length: int):
        """
        A class that contains the tokenized spans and span masks for a single document
        """
        super().__init__(d, tokenizer, max_seq_length)
        self.r_ids, self.r_attn_mask, self.r_span_masks = self._set_inputs(d.report, d.report_spans, max_seq_length)
        self.s_ids, self.s_attn_mask, self.s_span_masks = self._set_inputs(d.source, d.source_spans, max_seq_length)

    def _set_inputs(
            self,
            doctext: str,
            spans: List[DocumentSpan],
            max_seq_length: int
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        private method.
        args:
        d: Document, the original document that you wanna get the span masks out of.
        tokenizer: tokenizer of your choice

        Returns:
        Tokenized original document as ids: Tensor
        Span masks shaped as (num_of_spans, length_of_tokenized_document): Tensor
        """
        tokenized_doctext = self.tokenizer(
            text=doctext,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=max_seq_length,
            return_offsets_mapping=True)
        doctext_offset_map = tokenized_doctext['offset_mapping']

        span_masks = [self._get_span_masks(span, doctext_offset_map) for span in spans]
        # print(span_masks[0].shape)
        return tokenized_doctext['input_ids'], tokenized_doctext['attention_mask'], span_masks


class AnnotationSpanSet(SpanSet):
    def __init__(self, d: Document, tokenizer: AutoTokenizer, max_seq_length: int):
        super().__init__(d, tokenizer, max_seq_length)
        self.r_ids, self.r_attn_mask, self.r_anno_masks = self._set_inputs(d.report, d.report_spans, max_seq_length)
        self.s_ids, self.s_attn_mask, self.s_anno_masks = self._set_inputs(d.source, d.source_spans, max_seq_length)
        self.r_mask_arg_map, self.s_mask_arg_map = self._set_mask_arg_map()

    def _set_inputs(
            self,
            doctext: str,
            spans: List[DocumentSpan],
            max_seq_length: int
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        private method.
        args:
        d: Document, the original document that you wanna get the span masks out of.
        tokenizer: tokenizer of your choice

        Returns:
        Tokenized original document as ids: Tensor
        Span masks shaped as (num_of_spans, length_of_tokenized_document): Tensor
        """
        tokenized_doctext = self.tokenizer(
            text=doctext,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=max_seq_length,
            return_offsets_mapping=True)
        doctext_offset_map = tokenized_doctext['offset_mapping']

        span_masks = [self._get_span_masks(span, doctext_offset_map) for span in spans]
        # print(span_masks[0].shape)
        return tokenized_doctext['input_ids'], tokenized_doctext['attention_mask'], span_masks

    def _set_mask_arg_map(self):
        arguments = [span.argument for span in self.data.report_roles]
        r_argmap = {}
        for idx, arg in enumerate(arguments):
            mask = self.r_anno_masks[idx]
            r_argmap.update({arg: mask})
        arguments = [span.argument for span in self.data.source_roles]
        s_argmap = {}
        for idx, arg in enumerate(arguments):
            mask = self.s_anno_masks[idx]
            s_argmap.update({arg: mask})

        return r_argmap, s_argmap


class LlmSpanSet(SpanSet):
    def __init__(self, d: Document, tokenizer: AutoTokenizer, max_seq_length: int):
        super().__init__(d, tokenizer, max_seq_length)
        self.llm_ids, self.llm_attn_mask, self.llm_span_masks = self._set_inputs(d.llm_roles[-1].textual_span, d.llm_roles[:-1], max_seq_length)
        self.mask_arg_map = self._set_mask_arg_map()

    def _set_inputs(
        self,
        doctext: str,
        spans: List[DocumentSpan],
        max_seq_length: int
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        private method.
        args:
        d: Document, the original document that you wanna get the span masks out of.
        tokenizer: tokenizer of your choice

        Returns:
        Tokenized original document as ids: Tensor
        Span masks shaped as (num_of_spans, length_of_tokenized_document): Tensor
        """
        tokenized_doctext = self.tokenizer(
            text=doctext,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=max_seq_length,
            return_offsets_mapping=True)
        doctext_offset_map = tokenized_doctext['offset_mapping']
        span_masks = [self._get_span_masks(span, doctext_offset_map) for span in spans]
        return tokenized_doctext['input_ids'], tokenized_doctext['attention_mask'], span_masks

    def _set_mask_arg_map(self):
        arguments = [span.argument for span in self.data.llm_roles[:-1]]
        argmap = {}
        for idx, arg in enumerate(arguments):
            mask = self.llm_span_masks[idx]
            argmap.update({arg: mask})
        return argmap


def read_from_jsonl(file_path: str) -> List[Document]:
    with open(file_path, 'r') as f:
        data = [FieldObject(json.loads(line)) for line in f]
    # pprint.pp(data[0].report_dict.role_annotations)
    # for argument in list(data[0].report_dict.role_annotations._original_keys)[:-1]:
    #     print(getattr(data[0].report_dict.role_annotations, argument))
    combined_dataset = []

    for line in data:
        d = Document(
            instance_id=line.instance_id,
            instance_id_raw_lome_predictor=line.instance_id_raw_lome_predictor,
            frame=line.frame,
            report=line.report_dict.doctext,
            source=line.source_dict.doctext,
            report_tok=line.report_dict.doctext_tok,
            source_tok=line.source_dict.doctext_tok,
            report_spans=[DocumentSpan(span) for span in line.report_dict.all_spans],
            source_spans=[DocumentSpan(span) for span in line.source_dict.all_spans],
            trigger=DocumentSpan(line.report_dict.frame_trigger_span),
            report_roles=[
                DocumentSpan([argument])
                if getattr(line.report_dict.role_annotations, argument) == []
                else DocumentSpan(getattr(line.report_dict.role_annotations, argument)[0])
                for argument in list(line.report_dict.role_annotations._original_keys)[:-1]
            ],
            source_roles=[
                DocumentSpan([argument])
                if getattr(line.source_dict.role_annotations, argument) == []
                else DocumentSpan(getattr(line.source_dict.role_annotations, argument)[0])
                for argument in list(line.source_dict.role_annotations._original_keys)[:-1]
            ],
            llm_roles=None,
            is_platinum=line.bool_platinum,
        )
        combined_dataset.append(d)
    return combined_dataset
