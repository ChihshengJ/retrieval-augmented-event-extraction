import json
from typing import List, Tuple
from dataclasses import dataclass
import torch
from torch import Tensor
from transformers import AutoTokenizer

from model import Vectorizer


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
    A generic class for a docutmentation.
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
    doctext: str
    doctext_tok: List[str]
    spans: List[DocumentSpan]
    trigger: DocumentSpan
    roles: List[DocumentSpan]
    llm_roles: List[DocumentSpan]
    is_platinum: bool
    is_report: bool


class SpanSet: 
    def __init__(self, d: Document, tokenizer: AutoTokenizer):
        """
        A class that contains the tokenized spans and span masks for a single document
        """
        self.data = d
        self.tokenizer = tokenizer
        self.ids, self.attention_mask, self.span_masks = self._set_inputs(d.doctext, d.spans)

    def _set_inputs(self, doctext: str, spans: List[DocumentSpan]) -> Tuple[Tensor, List[Tensor]]:
        """
        private method.
        args:
        d: Document, the original document that you wanna get the span masks out of.
        tokenizer: tokenizer of your choice

        Returns:
        Tokenized original document as ids: Tensor
        Span masks shaped as (num_of_spans, length_of_tokenized_document): Tensor
        """
        tokenized_doctext = self.tokenizer(text=doctext, return_tensors='pt', truncation=True, padding='max_length', return_offsets_mapping=True)
        doctext_offset_map = tokenized_doctext['offset_mapping']
        # print(doctext_offset_map.shape, tokenized_doctext['input_ids'].shape)
        # tokens = [self.tokenizer.convert_ids_to_tokens(id) for id in tokenized_doctext['input_ids'].tolist()]
        # print(doctext)
        # print(doctext_offset_map[0, :25])

        span_masks = [self._get_span_mask(span, doctext_offset_map) for span in spans]
        # print(span_masks[0].shape)
        return tokenized_doctext['input_ids'], tokenized_doctext['attention_mask'], span_masks

    def _get_span_mask(self, s: DocumentSpan, offset_map: Tensor):
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


class AnnotationSpanSet(SpanSet):
    def __init__(self, d: Document, tokenizer: AutoTokenizer):
        super().__init__(d, tokenizer)
        _, _, self.annotation_span_masks = self._set_inputs(d.doctext, d.roles)
        self.mask_arg_map = self._set_mask_arg_map()

    def _set_mask_arg_map(self):
        arguments = [span.argument for span in self.data.roles]
        map = {}
        for idx, arg in enumerate(arguments):
            mask = self.annotation_span_masks[idx]
            map.update({arg: mask})
        return map


class LlmSpanSet(SpanSet):
    def __init__(self, d: Document, tokenizer: AutoTokenizer):
        super().__init__(d, tokenizer)
        # print(d.llm_roles[-1].textual_span)
        self.llm_ids, self.llm_attn_mask, self.llm_span_masks = self._set_inputs_llm(d.llm_roles[-1].textual_span, d.llm_roles[:-1])
        # print(self.llm_span_masks)
        self.mask_arg_map = self._set_mask_arg_map()

    def _set_mask_arg_map(self):
        arguments = [span.argument for span in self.data.llm_roles[:-1]]
        map = {}
        for idx, arg in enumerate(arguments):
            mask = self.llm_span_masks[idx]
            map.update({arg: mask})
        return map

    def _set_inputs_llm(self, doctext: str, spans: List[DocumentSpan]) -> Tuple[Tensor, List[Tensor]]:
        """
        private method.
        args:
        d: Document, the original document that you wanna get the span masks out of.
        tokenizer: tokenizer of your choice

        Returns:
        Tokenized original document as ids: Tensor
        Span masks shaped as (num_of_spans, length_of_tokenized_document): Tensor
        """
        tokenized_doctext = self.tokenizer(text=doctext, return_tensors='pt', truncation=True, padding='max_length', return_offsets_mapping=True)
        doctext_offset_map = tokenized_doctext['offset_mapping']
        span_masks = [self._get_span_mask_llm(span, doctext_offset_map) for span in spans]
        return tokenized_doctext['input_ids'], tokenized_doctext['attention_mask'], span_masks

    def _get_span_mask_llm(self, s: DocumentSpan, offset_map: Tensor):
        if s.textual_span is None or s.char_start_idx is None:
            return torch.zeros((1, offset_map.shape[1]), dtype=torch.float)
        # print(s.char_start_idx, s.char_end_idx)
        span_mask = torch.zeros((1, offset_map.shape[1]), dtype=torch.float)
        start_idx, end_idx = s.char_start_idx, s.char_end_idx
        real_start_idx, real_end_idx = 0, 0
        for i, token in enumerate(offset_map[0, :-1, :]): 
            real_start_idx = i if token[0].item() == start_idx else real_start_idx
            real_end_idx = i if token[1].item() == end_idx else real_end_idx
        span_mask[0, real_start_idx: real_end_idx + 1] = 1
        # print(span_mask[0, :25])
        # print('tokens:', span_mask.sum().item())
        return span_mask 


def read_from_jsonl(file_path: str) -> List[Document]:
    with open(file_path, 'r') as f:
        data = [FieldObject(json.loads(line)) for line in f]
    # pprint.pp(data[0].report_dict.role_annotations)
    # for argument in list(data[0].report_dict.role_annotations._original_keys)[:-1]:
    #     print(getattr(data[0].report_dict.role_annotations, argument))
    combined_dataset = []

    for line in data:
        report = Document(
            instance_id=line.instance_id,
            instance_id_raw_lome_predictor=line.instance_id_raw_lome_predictor,
            frame=line.frame,
            doctext=line.report_dict.doctext,
            doctext_tok=line.report_dict.doctext_tok,
            spans=[DocumentSpan(span) for span in line.report_dict.all_spans],
            trigger=DocumentSpan(line.report_dict.frame_trigger_span),
            roles=[DocumentSpan([argument])
                   if getattr(line.report_dict.role_annotations, argument) == []
                   else DocumentSpan(getattr(line.report_dict.role_annotations, argument)[0]) 
                   for argument in list(line.report_dict.role_annotations._original_keys)[:-1]],
            llm_roles=None,
            is_platinum=line.bool_platinum,
            is_report=True
        )
        combined_dataset.append(report)
        source = Document(
            instance_id=line.instance_id,
            instance_id_raw_lome_predictor=line.instance_id_raw_lome_predictor,
            frame=line.frame,
            doctext=line.source_dict.doctext,
            doctext_tok=line.source_dict.doctext,
            spans=[DocumentSpan(span) for span in line.source_dict.all_spans],
            trigger=DocumentSpan(line.report_dict.frame_trigger_span),
            roles=[DocumentSpan([argument])
                   if getattr(line.source_dict.role_annotations, argument) == []
                   else DocumentSpan(getattr(line.source_dict.role_annotations, argument)[0]) 
                   for argument in list(line.source_dict.role_annotations._original_keys)[:-1]],
            llm_roles=None,
            is_platinum=line.bool_platinum,
            is_report=False
        )
        combined_dataset.append(source)
    return combined_dataset

