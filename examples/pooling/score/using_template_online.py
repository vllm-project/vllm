# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Example of using the rerank API with template.

This script demonstrates how to interact with a vLLM server running
a reranking model via the REST API.
Before running this script, start the vLLM server with one of the
supported reranking models using the commands below.

note:
    Some reranking models require special configuration overrides to work correctly
    with vLLM's score API.
    Reference: https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/qwen3_reranker_online.py
    Reference: https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/convert_model_to_seq_cls.py

run:
    vllm serve BAAI/bge-reranker-v2-gemma --hf_overrides '{"architectures": ["GemmaForSequenceClassification"],"classifier_from_token": ["Yes"],"method": "no_post_processing"}' --chat-template examples/pooling/score/template/bge-reranker-v2-gemma.jinja
    vllm serve tomaarsen/Qwen3-Reranker-0.6B-seq-cls --chat-template examples/pooling/score/template/qwen3_reranker.jinja
    vllm serve mixedbread-ai/mxbai-rerank-base-v2 --hf_overrides '{"architectures": ["Qwen2ForSequenceClassification"],"classifier_from_token": ["0", "1"], "method": "from_2_way_softmax"}' --chat-template examples/pooling/score/template/mxbai_rerank_v2.jinja
    vllm serve nvidia/llama-nemotron-rerank-1b-v2 --runner pooling --trust-remote-code --chat-template examples/pooling/score/template/nemotron-rerank.jinja
    vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' --chat-template examples/pooling/score/template/qwen3_reranker.jinja
"""

import json

import requests

# URL of the vLLM server's rerank endpoint
# Default vLLM server runs on localhost port 8000
url = "http://127.0.0.1:8000/rerank"

# HTTP headers for the request
headers = {"accept": "application/json", "Content-Type": "application/json"}

# Example query & documents
query = "how much protein should a female eat?"
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    "Calorie intake should not fall below 1,200 a day in women or 1,500 a day in men, except under the supervision of a health professional.",
]

# Request payload for the rerank API
data = {
    "model": "nvidia/llama-nemotron-rerank-1b-v2",  # Model to use for reranking
    "query": query,  # The query to score documents against
    "documents": documents,  # List of documents to be scored
}


def main():
    """Main function to send a rerank request to the vLLM server.

    This function sends a POST request to the /rerank endpoint with
    the query and documents, then prints the relevance scores.
    """
    # Send POST request to the vLLM server's rerank endpoint
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
