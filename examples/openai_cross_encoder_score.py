"""Examples Python client Score for Cross Encoder Models
"""

import argparse
import json
import pprint

import requests


def post_http_request(prompt: json, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3")
    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/v1/score"

    model_name = args.model

    text_1 = "What is the capital of France?"
    text_2 = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Prompt for text_1 is string and text_2 is a list:")
    pprint.pprint(prompt)
    print("Score Response:")
    pprint.pprint(score_response.data)

    text_1 = [
        "What is the capital of Brazil?", "What is the capital of France?"
    ]
    text_2 = [
        "The capital of Brazil is Brasilia.", "The capital of France is Paris."
    ]
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Prompt for text_1 and text_2 are lists:")
    pprint.pprint(prompt)
    print("Score Response:")
    pprint.pprint(score_response.data)

    text_1 = "What is the capital of Brazil?"
    text_2 = "The capital of Brazil is Brasilia."
    prompt = {"model": model_name, "text_1": text_1, "text_2": text_2}
    score_response = post_http_request(prompt=prompt, api_url=api_url)
    print("Prompt for text_1 and text_2 are strings:")
    pprint.pprint(prompt)
    print("Score Response:")
    pprint.pprint(score_response.data)