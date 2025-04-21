import json
import numpy as np
from tqdm import tqdm
import argparse

import src.iterx
from src.iterx.metrics.famus.iterx_famus import IterXFAMuSMetric


def compute_ceafe_rme_scores(gold_file, predictions,
                             ignore_no_template_doc=False,
                             sanitize_special_chars=False,
                             scorer_type='phi-3-levenshtein'):
    # Soft Match
    iterx_famus = IterXFAMuSMetric({gold_file: gold_file},
                                   scorer_type=scorer_type,
                                   ignore_no_template_doc=ignore_no_template_doc,
                                   sanitize_special_chars=sanitize_special_chars)
    iterx_famus(predictions,
                gold_file,
                normalize_role=False)
    return iterx_famus.get_metric(reset=True)['iterx_famus_slot_f1']


def print_compute_ceafe_rme_scores(
    gold_file,
    predictions,
    ignore_no_template_doc=True,
    sanitize_special_chars=True
):
    # Exact Match
    iterx_famus = IterXFAMuSMetric({gold_file: gold_file},
                                 scorer_type='phi-3',
                                 ignore_no_template_doc=ignore_no_template_doc,
                                 sanitize_special_chars=sanitize_special_chars)
    iterx_famus(predictions,
                gold_file,
                normalize_role=False)

    exact_match_dict = iterx_famus.get_metric(reset=True)

    metrics_string = "(CEAF_RME_phi-3), P, R, F1, (CEAF_RME_phi-a), P, R, F1: \n"
    metrics_string += f"{exact_match_dict['iterx_famus_slot_p']*100:.2f} & "
    metrics_string += f"{exact_match_dict['iterx_famus_slot_r']*100:.2f} & "
    metrics_string += f"{exact_match_dict['iterx_famus_slot_f1']*100:.2f} & "

    # Soft Match
    iterx_famus = IterXFAMuSMetric({gold_file: gold_file},
                                     scorer_type='phi-3-levenshtein',
                                     ignore_no_template_doc=ignore_no_template_doc,
                                     sanitize_special_chars=sanitize_special_chars)
    iterx_famus(predictions,
                gold_file,
                normalize_role=False)
    soft_match_dict = iterx_famus.get_metric(reset=True)

    metrics_string += f"{soft_match_dict['iterx_famus_slot_p']*100:.2f} & "
    metrics_string += f"{soft_match_dict['iterx_famus_slot_r']*100:.2f} & "
    metrics_string += f"{soft_match_dict['iterx_famus_slot_f1']*100:.2f} & "

    print(metrics_string)


def chatgpt_response_to_iterx_format(chatgpt_predictions):
    """
    Convert a chatgpt response to iterx format
    """
    result = {}
    for prediction in chatgpt_predictions:
        # print(list(prediction.keys()))
        famus_id = list(prediction.keys())[0]
        incident_type = famus_id.split('-frame-', 1)[-1]

        # Initialize the dictionary for this famus_id if it doesn't exist
        if famus_id not in result:
            result[famus_id] = [{'incident_type': incident_type}]

        for role_dict in prediction[famus_id]:
            result[famus_id][0].update(role_dict)

    # Remove incident_type from the dicts that have no other keys
    for famus_id in result:
        if len(result[famus_id][0]) == 1:  # Only 'incident_type' is present
            result[famus_id] = []

    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate results')

    parser.add_argument('--model_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--context', type=str)
    parser.add_argument('--setting', type=str)
    parser.add_argument('--post', type=str, default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    gold_files = {'report':
                  {'dev': "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/iterx_format/report_data/mixed_spans/dev.jsonl", 
                   'test': "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/iterx_format/report_data/mixed_spans/test.jsonl"},
                  'source': 
                  {'dev': "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/iterx_format/source_data/mixed_spans/dev.jsonl", 
                   'test': "/data/svashishtha/FAMuS/data/cross_doc_role_extraction/iterx_format/source_data/mixed_spans/test.jsonl"}
                  }

    context = args.context
    split = args.split
    setting = args.setting
    model_name = args.model_name
    post = args.post
    print(context, split, setting, post)
    if post == '':
        llm_path = f"/data/cjin/retrieval-augmented-event-extraction/data/llm_responses/{context}_{split}_{setting}_{model_name}.jsonl"
    else:
        llm_path = f"/data/cjin/retrieval-augmented-event-extraction/data/llm_responses/{context}_{split}_{setting}_{model_name}_{post}.jsonl"
    with open(llm_path, 'r') as f:
        chatgpt_predictions = [json.loads(line) for line in f]

    preds = chatgpt_response_to_iterx_format(chatgpt_predictions)

    print_compute_ceafe_rme_scores(gold_files[context][split], preds)


if __name__ == '__main__':
    main()
