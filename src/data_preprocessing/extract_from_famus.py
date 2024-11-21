import json
from typing import List
from dataclasses import dataclass
from torch import Tensor
import pprint


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
    textual_span: str
    char_start_idx: int
    char_end_idx: int
    token_start_idx: int
    token_end_idx: int

    def __init__(self, span: List):
        self.textual_span = span[0]
        self.char_start_idx = span[1]
        self.char_end_idx = span[2]
        self.token_start_idx = span[3]
        self.token_end_idx = span[4]


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
    roles: FieldObject
    is_platinum: bool


def read_from_json(file_path: str) -> List[Document]:
    with open(file_path, 'r') as f:
        data = [FieldObject(json.loads(line)) for line in f]
    # pprint.pp(data[0].report_dict.all_spans)
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
            roles=line.report_dict.role_annotations,
            is_platinum=line.bool_platinum
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
            roles=line.source_dict.role_annotations,
            is_platinum=line.bool_platinum
        )
        combined_dataset.append(source)

    return combined_dataset

