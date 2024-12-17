from typing import List, Dict, Tuple
import os
import json

from .llm_backend import CompletionAPIFactory, CompletionAPI
from data_preprocessing.data import Document, DocumentSpan
from data_preprocessing.data_utils import frame_to_llm_prompt_info_dct, fuzzy_find

SYSTEM_PROMPT = """
You are a system that generates high quality role annotations of a text describing an event based on given event arguments.
The following inputs are given to you:
1. Event Type: A Frame name from the FrameNet ontology (eg: Hiring, Arrest, etc.)
2. Event Definition: Definition of the event type along with an optional example.
3. Event keywords: A span in the document that pinpoint the event. There could be no keywords.
3. Roles: All roles (or participants) of the event type (or frame) followed with its definition.
4. Document: A document from which the roles are to be extracted.
You should output the exact spans from the document for each role in the order they are listed in the "roles" section.
Then, use the exact spans you predicted that are not N/A to generate a 1 - 2 sentence summary describing the event.
If you think there are multiple candidates for a span, please choose the most informative one.
please answer with the following format:
"Role1: span1
Role2: span2
...
Summary: 1-2 sentence summary of the event"
Note that you can leave an N/A if you are not certain whether there is any span representing the role in the document.
"""

PROMPT_TEMPLATE = """
Event type: {event_type},
Event definition: {event_definition}
Event keywords: {event_trigger}
Roles: 
{roles}
Document: {document},
Answer:
"""

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    OPENAI_API_KEY = api_key
else:
    OPENAI_API_KEY = None
    print("Warning: OPENAI_API_KEY is not set")


default_backend = CompletionAPIFactory.get_api(api_name='openai', api_key=OPENAI_API_KEY)

# stanza initialization
# stanza.download('en')
# stanza_pipeline = stanza.Pipeline(lang='en', processors='tokenize')


def get_frame_info(d: Document) -> Dict:
    return frame_to_llm_prompt_info_dct(d.frame)


def get_llm_prediction(d: Document, model: str = 'gpt-3.5-turbo', back_end: CompletionAPI = default_backend, max_tokens: int = 500) -> str:
    frame_info = get_frame_info(d)
    system_prompt = SYSTEM_PROMPT
    prompt = PROMPT_TEMPLATE.format(
        event_type=frame_info['event_type'],
        event_definition=frame_info['event_definition'],
        event_trigger=d.trigger.textual_span,
        roles=frame_info['event_roles'],
        document=d.doctext
    )
    # print('prompt:', prompt)
    output = back_end.get_completion(prompt=prompt, 
                                     system_prompt=system_prompt,
                                     model=model,
                                     max_tokens=max_tokens)
    return output


def parse_llm_prediction(pred: str):
    data_dict = {}
    for line in pred.splitlines():
        if ":" in line:
            key, value = line.split(":", 1) 
            data_dict[key.strip()] = value.strip()
        else: 
            continue
    return data_dict


def write_llm_spans_into_docs(dataset: List[Document], file_path: str):
    """Need to make sure that dataset and llm responses are aligned"""
    max_len = len(dataset)
    with open(file_path, 'r') as f:
        data = json.load(f)
        for idx, line in enumerate(data):
            if idx < max_len:
                write_llm_spans_into_one_doc(dataset[idx], line)
            else:
                break


def write_llm_spans_into_one_doc(d: Document, llm_dict: Dict):
    llm_roles = []
    for argument, span in list(llm_dict.items())[:-1]:
        start_idx, end_idx = fuzzy_find(llm_dict['Summary'], span)
        if span == 'N/A' or span == llm_dict['Summary']:
            start_idx, end_idx = None, None
        span_list = [span, start_idx, end_idx, None, None, argument]
        span = DocumentSpan(span_list)
        llm_roles.append(span)
    d.llm_roles = llm_roles






