from typing import List, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name_or_path = "BAAI/bge-reranker-base"
cache_dir = None
max_length = 512

sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]] = \
         [("hello world", "nice to meet you"), ("head north", "head south")]
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          cache_dir=cache_dir)
# XLMRobertaForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                           cache_dir=cache_dir)
model = model.to("cuda")
model.eval()

inputs = tokenizer(
    sentence_pairs,
    padding=True,
    truncation=True,
    return_tensors='pt',
    max_length=max_length,
).to("cuda")

all_scores = []
with torch.no_grad():
    logits = model(**inputs, return_dict=True).logits
    scores = logits.view(-1, ).float()
    all_scores.extend(scores.cpu().numpy().tolist())
print(all_scores)
