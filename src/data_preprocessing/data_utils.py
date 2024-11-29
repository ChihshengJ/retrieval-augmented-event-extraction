from nltk.corpus import framenet as fn
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import List, Tuple
import re


# This part is for extracting the frame information from the FrameNet corpus.
def frame_to_def(frame: str) -> str: 
    """
    definition of a frame (take only the first sentence)
    """
    # doc = stanza_pipeline(fn.frame(frame)['definition'])
    # sents = [sentence.text for sentence in doc.sentences]
    # # For small errors where sentence boundary splits the first sentence
    # # Example: Frame: 'Arriving'
    # if len(sents[0].split()) < 5:
    #     return sents[0] + " " + sents[1]
    # return sents[0]
    doc = fn.frame(frame)['definition']
    import re
    match = re.match(r"^(.*?)'", doc)
    if match:
        return match.group(1)
    else:
        print('problem, ', doc)
        return None


def frame_to_core_roles(frame: str):
    """
    extract a list of all core roles
    """
    # Added extra roles that are generally frequent 
    extra_roles = ['Time', 'Place']
    extra_valid_roles = [role for role in extra_roles if role in fn.frame(frame).FE]
    core_roles = [role for role, dcts in fn.frame(frame).FE.items() 
                  if dcts['coreType'] == 'Core' or dcts['coreType'] == "Core-Unexpressed"]
    extra_valid_roles = [role for role in extra_valid_roles if role not in core_roles]
    return core_roles + extra_valid_roles


def frame_to_role_def_example(frame: str, role: str):
    """
    Given a frame and its role, get the role definition and role example
    """
    import re
    soup = BeautifulSoup(fn.frame(frame).FE[role]['definitionMarkup'], features="lxml")
    # a role can used inside <fex> tag with either its name or its abbrev
    role_names = [role, fn.frame(frame).FE[role]['abbrev']]
    # get definition of role:
    string_def = " "
    for child in list(soup.find("def-root").children):
        if "<ex>" not in str(child):
            string_def += str(child)
        else:
            break
    string_def = string_def.strip() 

    #replace "FE" by role in the definition of a role
    string_def = re.sub(r' FE ', r' role ', string_def)
    # Get the first example of the role
    examples = soup.findAll("ex")
    if examples:
        example = examples[0]
        # remove the <fex> tag for other roles
        example_str = ""
        for child in example.children:
            if isinstance(child, Tag):
                if child.name == "fex" and child.get("name") not in role_names:
                    # only get contents of roles that are not this role
                    string_contents = [str(c) for c in child.contents]
                    example_str += " ".join(string_contents) 
                elif child.name != "fex":
                    # only get contents of other tags (such as <t> tag)
                    string_contents = [str(c) for c in child.contents]
                    example_str += " ".join(string_contents) 
                else:
                    example_str += str(child)
            else:
                example_str += str(child)
    else:
        example_str = ""

    return (string_def, simplify_fex_tag(example_str))


def simplify_fex_tag(string):
    import re 
    string = re.sub(r"<fex[^>]*>.*?</fex>", r'', string)
    return string


def make_double_quotes_single(string):
    return string.replace(r'"', "'")


def frame_to_info_dct(frame: str):
    """
    Given an input frame, extract the required info (for rams2 annotation) as JSON dicts
    """
    core_roles = frame_to_core_roles(frame)
    dct = {"frameDefinitions": [make_double_quotes_single(frame_to_def(frame))],
           'roles': core_roles, 
           'roleDefinitions': [make_double_quotes_single(frame_to_role_def_example(frame, role)[0]) for role in core_roles], 
           'roleExamples': [make_double_quotes_single(str(frame_to_role_def_example(frame, role)[1])) for role in core_roles], 
           } 
    return dct


def frame_to_llm_prompt_info_dct(frame_name):
    """
    Given a frame name, return a dictionary with the following keys:
    - event_type: frame_name
    - event_definition: frame definition with example
    - event_roles: frame roles with definitions and examples
    """
    info_dct = frame_to_info_dct(frame_name)
    # Event Roles
    event_def_roles = ""
    for idx, (role, role_def, role_example) in enumerate(zip(info_dct['roles'],
                                                         info_dct['roleDefinitions'], 
                                                         info_dct['roleExamples'])): 
        event_def_roles += f"{idx+1}. {role}: {role_def}"
        event_def_roles += f"\n{role_example}\n"

    return {'event_type': frame_name,
            'event_definition': info_dct['frameDefinitions'][0],
            'event_roles': event_def_roles}


# This part is for generate spans from the LLM summaries.
def tokenize(string: str):
    "This is just a naive implementation, it can be swapped"
    return string.lower().split()


def strip_punctuation(token):
    return re.sub(r'[^\w\s]', '', token)


def fuzzy_find(summary, span, match_score=2, mismatch_penalty=-1, gap_penalty=-1) -> Tuple[int, int]:
    """
    An axuiliary function used for finding the indices of a span in a summary.
    args:
        summary: str, a summary in string
        span: str, a span in string
        match_score: int, default to 2
        mismatch_penalty: int, default to -1
        gap_penalty: int, default to -1
    returns:
        (start_idx, end_idx): character indices of the span in the summary
    """
    summary_tokens = tokenize(summary)
    span_tokens = tokenize(span)
    (start_idx, end_idx), aligned_seq1, aligned_seq2 = smith_waterman(
        summary_tokens, span_tokens, match_score, mismatch_penalty, gap_penalty
    )
    if start_idx == end_idx:
        return None, None

    start_char_idx, end_char_idx = tokens_to_char_indices(
        summary, summary_tokens, start_idx, end_idx
    )

    return start_char_idx, end_char_idx


def token_similarity(token1, token2):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, token1.lower(), token2.lower()).ratio()


def smith_waterman(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-1, similarity_threshold=0.5):
    m, n = len(seq1), len(seq2)
    score_matrix = [[0] * (n + 1) for _ in range(m + 1)]
    max_score = 0
    max_pos = None

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            token1 = strip_punctuation(seq1[i - 1]).lower()
            token2 = strip_punctuation(seq2[j - 1]).lower()
            similarity = token_similarity(token1, token2)
            if similarity >= similarity_threshold:
                score = match_score * similarity
            else:
                score = mismatch_penalty
            diag = score_matrix[i - 1][j - 1] + score
            up = score_matrix[i - 1][j] + gap_penalty
            left = score_matrix[i][j - 1] + gap_penalty
            score_matrix[i][j] = max(0, diag, up, left)
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                max_pos = (i, j)

    if max_pos is None:
        return (0, 0), [], []

    i, j = max_pos
    aligned_seq1 = []
    aligned_seq2 = []
    while i > 0 and j > 0 and score_matrix[i][j] > 0:
        token1 = strip_punctuation(seq1[i - 1]).lower()
        token2 = strip_punctuation(seq2[j - 1]).lower()
        if token1 == token2:
            aligned_seq1.insert(0, seq1[i - 1])
            aligned_seq2.insert(0, seq2[j - 1])
            i -= 1
            j -= 1
        elif score_matrix[i - 1][j] + gap_penalty == score_matrix[i][j]:
            i -= 1
        elif score_matrix[i][j - 1] + gap_penalty == score_matrix[i][j]:
            j -= 1
        else:
            i -= 1
            j -= 1

    start_idx_seq1 = i
    end_idx_seq1 = max_pos[0]

    return (start_idx_seq1, end_idx_seq1), aligned_seq1, aligned_seq2


def tokens_to_char_indices(text, tokens, start_token_idx, end_token_idx):
    spans = list(re.finditer(r'\S+', text))
    if start_token_idx >= len(spans):
        return len(text), len(text)
    start_char_idx = spans[start_token_idx].start()
    if end_token_idx - 1 >= len(spans):
        end_char_idx = len(text)
    else:
        end_char_idx = spans[end_token_idx - 1].end()
    return start_char_idx, end_char_idx

