# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""
Example of using the rerank API with template.

run:
    vllm serve nvidia/llama-nemotron-rerank-1b-v2 --runner pooling --trust-remote-code --chat-template examples/pooling/score/template/nemotron-rerank.jinja
"""

import json

import requests

url = "http://127.0.0.1:8000/rerank"

headers = {"accept": "application/json", "Content-Type": "application/json"}

query = "how much protein should a female eat?"
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    "Calorie intake should not fall below 1,200 a day in women or 1,500 a day in men, except under the supervision of a health professional.",
]

data = {
    "model": "nvidia/llama-nemotron-rerank-1b-v2",
    "query": query,
    "documents": documents,
}


def main():
    response = requests.post(url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
