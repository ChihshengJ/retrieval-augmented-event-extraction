from transformers import AutoTokenizer

from llms.llm_span import get_llm_prediction, parse_llm_prediction, write_llm_spans_into_docs
from llms.llm_backend import CompletionAPIFactory
from .data import Document, AnnotationSpanSet, LlmSpanSet, read_from_jsonl


paths = ['../data/test.jsonl']
test_f = []
for path in paths:
    test_f.extend(read_from_jsonl(path))
print(len(test_f))
llm_doc_path = '../data/llm_naive_prediction_for_all_documents_v2.json' 
write_llm_spans_into_docs(test_f, llm_doc_path)
tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

for doc in test_f[:3]:
    print(doc.doctext)
    llm_res = LlmSpanSet(doc, tokenizer)



