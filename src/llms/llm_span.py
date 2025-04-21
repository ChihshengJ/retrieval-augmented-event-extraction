from typing import List, Dict, Tuple
import os
import json
import re

from .llm_backend import CompletionAPIFactory, CompletionAPI
import src.data_processing.data
import src.data_processing.data_utils
from src.data_processing.data import Document, DocumentSpan
from src.data_processing.data_utils import frame_to_llm_prompt_info_dict, fuzzy_find

SYSTEM_PROMPT = """
You are a system that generates high quality role annotations of one or two documents describing an event, based on given event roles.
The following inputs are given to you:
1. Event Type: A Frame name from the FrameNet ontology (eg: Hiring, Arrest, etc.)
2. Event Definition: Definition of the event type along with an optional example.
3. Event keywords: A span in the document that pinpoint the event. There could be no keywords.
4. Roles: All roles (or participants) of the event type (or frame) followed with its definition.
5. Report: A document that provides a description of the event, usually shorter thant the Source.
6. Source: (Optional) A document that potentialy provides more detail to the event presented in the Report.
You should output the exact text spans from either the Report (if Source is None) or the Source (if both are present) for each role in the order they are listed in the "roles" section.
Note that if a Source is present (not None), please only extract roles from the Source with regard to the event described in the Report. 
If there are multiple candidates for one role, return the most informative one, i.e. "March 1st" is more informative than "Thursday", "Michael Jackson" is more informative than "the pop star".
Then, use the exact spans you predicted that are not N/A to generate a 1 - 2 sentence summary describing the event.
If there are multiple candidates for a span, please choose the most informative one.
Please answer with the following format (the Role1 and Role2 are placeholders):
"Role1: span1
Role2: span2
...
Summary: 1-2 sentence summary of the event"
Note that you can leave an N/A if you are not certain whether there is any span representing the role in the document.
"""

FEW_SHOT_PREFIX = """
Event type: {event_type},
Event definition: {event_definition}
Event keywords: {event_trigger}
Roles: 
{roles}
Report: {report}
Source: {source},
Answer: {answers}
"""

PROMPT_TEMPLATE = """
Event type: {event_type},
Event definition: {event_definition}
Event keywords: {event_trigger}
Roles: 
{roles}
Report: {report}
Source: {source},
Answer:
"""


def get_llm_predictions(
        trainset: List[Document],
        idx: int,
        d: Document,
        model: str,
        context: str,
        setting: str = 'ZS',
        back_end: CompletionAPI = None,
        max_tokens: int = 500
) -> str:
    """
    Generate LLM predictions for a single document.
    """

    info_dict = frame_to_llm_prompt_info_dict(d.frame)

    roles = re.sub(r"<fe[xn].*?>|</fe[xn]>", '', info_dict['event_roles'])

    system_prompt = SYSTEM_PROMPT if setting == 'ZS' else SYSTEM_PROMPT + "\nThree examples are shown in the following text:"
    if setting != 'ZS':
        index = idx * 3
        few_shot_prompt = ""
        for i in range(index, index + 3):
            if context == 'report':
                few_shot_roles = [{span.argument: span.textual_span} for span in trainset[i].report_roles]
                ans = "\n"
                for role in few_shot_roles:
                    ans += str(role).strip("{}").replace("'", '') + '\n'

            else:
                few_shot_roles = [{span.argument: span.textual_span} for span in trainset[i].source_roles]
                ans = "\n"
                for role in few_shot_roles:
                    ans += str(role).strip("{}").replace("'", '') + '\n'
            few_shot_prompt += FEW_SHOT_PREFIX.format(
                event_type=info_dict['event_type'],
                event_definition=info_dict['event_definition'],
                event_trigger=trainset[i].trigger.textual_span,
                roles=roles,
                report=trainset[i].report,
                source=None if context == 'report' else trainset[i].source,
                answers=ans
            )
        prompt = few_shot_prompt + PROMPT_TEMPLATE.format(
            event_type=info_dict['event_type'],
            event_definition=info_dict['event_definition'],
            event_trigger=d.trigger.textual_span,
            roles=roles,
            report=d.report,
            source=None if context == 'report' else d.source
        )
    else:
        prompt = PROMPT_TEMPLATE.format(
            event_type=info_dict['event_type'],
            event_definition=info_dict['event_definition'],
            event_trigger=d.trigger.textual_span,
            roles=roles,
            report=d.report,
            source=None if context == 'report' else d.source
        )
    # print('prompt:', prompt)
    output = back_end.get_completion(prompt=prompt,
                                     system_prompt=system_prompt,
                                     model=model,
                                     max_tokens=max_tokens)
    return output


def parse_llm_predictions(pred: str) -> Dict:
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
