# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example of using the OpenAI entrypoint's rerank API which is compatible with
Qwen and Cohere https://github.com/QwenLM/Qwen3-Embedding

run: vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
"""

import json

import requests

url = "http://127.0.0.1:8000/rerank"

headers = {"accept": "application/json", "Content-Type": "application/json"}

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
instruction = "Given a web search query, retrieve relevant passages that answer the query"
query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

query = "What is the capital of China?"
documents = [
        "The capital of China is Beijing.",
        "The capital of France is Paris.",
        "Horses and cows are both animals",
]

data = {
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": query_template.format(prefix=prefix, instruction=instruction, query=query),
    "documents": [document_template.format(doc=doc, suffix=suffix) for doc in documents]
}

def main():
    response = requests.post(url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()