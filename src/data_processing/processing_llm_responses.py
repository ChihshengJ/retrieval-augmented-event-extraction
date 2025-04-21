import os
import json
import re

import src.data_processing.data_utils
from src.data_processing.data_utils import fuzzy_find


def span_match(text, pred):
    start_char_idx, end_char_idx = fuzzy_find(text, pred)
    text_result = text[start_char_idx: end_char_idx]
    # print(f'input span: {pred}')
    # print(f'find match: {text_result}')
    return text_result


def llm_responses_in_jsonl(model_name, splits, contexts, settings, famus, match_span=False):
    assert len(famus) == len(splits)
    for se in settings:
        data = []
        for split in splits:
            s = []
            for context in contexts:
                file_path = f'/data/cjin/retrieval-augmented-event-extraction/data/llm_draft_results/{model_name}_{se}_{context}_{split}.json'
                print(f'reading from {file_path}')
                with open(file_path, 'r') as f:
                    curr_cont = json.load(f)
                s.append(curr_cont)
            data.append(s)
            # data = [[dev report, dev source], [test report, test source]]

        os.makedirs('/data/cjin/retrieval-augmented-event-extraction/data/llm_responses/', exist_ok=True)
        for i, s in enumerate(splits):
            for j, c in enumerate(contexts):
                d = data[i][j]
                info = famus[i]
                if match_span:
                    output_path = f'/data/cjin/retrieval-augmented-event-extraction/data/llm_responses/{c}_{s}_{se}_{model_name}_span-matched.jsonl'
                else:
                    output_path = f'/data/cjin/retrieval-augmented-event-extraction/data/llm_responses/{c}_{s}_{se}_{model_name}.jsonl'
                print(f'save output to {output_path}')
                with open(output_path, 'w') as f:
                    for n, pred_roles in enumerate(d):
                        doc_id = info[n]['instance_id']
                        doctext = info[n][f'{c}_dict']['doctext']
                        json_object = {doc_id: None}
                        role_lst = []
                        fixed_pred_roles = {key.split(". ", 1)[1] if ". " in key else key: value for key, value in pred_roles.items()}
                        for role in list(info[n][f'{c}_dict']['role_annotations'].keys()):
                            if role == 'role-spans-indices-in-all-spans':
                                continue
                            try:
                                span = fixed_pred_roles[role]
                            except KeyError:
                                span = ''
                                print(f'no span for {role} at {c} {s} {n}')
                                print(list(fixed_pred_roles.keys()))
                            if span in ('N/A', 'None'):
                                span = ''
                            if match_span:
                                span = span_match(doctext, span)
                            role_lst.append({role: [[span]]})
                        json_object[doc_id] = role_lst
                        json.dump(json_object, f)
                        f.write("\n")


def main():
    splits = ['dev', 'test']
    contexts = ['report', 'source']
    settings = ['ZS', 'FS']

    model_name = 'claude-3-5-sonnet-20241022'
    # model_name = 'gpt-4o'

    paths = [
        # '/data/cjin/retrieval-augmented-event-extraction/data/train.jsonl',
        '/data/cjin/retrieval-augmented-event-extraction/data/dev.jsonl',
        '/data/cjin/retrieval-augmented-event-extraction/data/test.jsonl'
    ]

    famus = []
    for idx, _ in enumerate(splits):
        with open(paths[idx], 'r') as f:
            data = [json.loads(line) for line in f]
        famus.append(data)
        # famus = [[dev, test]]

    llm_responses_in_jsonl(model_name, splits, contexts, settings, famus, match_span=True)



if __name__ == '__main__':
    main()
