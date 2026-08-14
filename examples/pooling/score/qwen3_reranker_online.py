# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
What is the difference between the official original version and one
that has been converted into a sequence classification model?

Qwen3-Reranker is a language model that doing reranker by using the
logits of "no" and "yes" tokens.
This requires computing logits for all 151,669 tokens in the vocabulary,
making it inefficient and incompatible with vLLM's score() API.

A conversion method has been proposed to transform the original model into a
sequence classification model. This converted model:
1. Is significantly more efficient
2. Fully supports vLLM's score() API
3. Simplifies initialization parameters
Reference: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
Reference: https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/convert_model_to_seq_cls.py

For the converted model, initialization would simply be:
    vllm serve tomaarsen/Qwen3-Reranker-0.6B-seq-cls --runner pooling --chat-template examples/pooling/score/template/qwen3_reranker.jinja

This example demonstrates loading the ORIGINAL model with special overrides
to make it compatible with vLLM's score API.
    vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' --chat-template examples/pooling/score/template/qwen3_reranker.jinja
"""

import json

import requests

# URL of the vLLM server's score endpoint
# Default vLLM server runs on localhost port 8000
url = "http://127.0.0.1:8000/score"

# HTTP headers for the request
headers = {"accept": "application/json", "Content-Type": "application/json"}

# Example queries & documents
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Request payload for the score API
data = {
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "queries": queries,
    "documents": documents,
}


def main():
    """Main function to send a score request to the vLLM server.

    This function sends a POST request to the /score endpoint with
    the query and documents, then prints the relevance scores.
    """
    # Send POST request to the vLLM server's score endpoint
    response = requests.post(url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request successful!")
        # Pretty print the JSON response containing relevance scores
        # The response includes scores for each document's relevance to the query
        print(json.dumps(response.json(), indent=2))
    else:
        # Handle request failure
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
