# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/boltuix/NeuroBERT-NER

"""
Example online usage of Pooling API for Named Entity Recognition (NER).

Run `vllm serve <model> --runner pooling`
to start up the server in vLLM. e.g.

vllm serve boltuix/NeuroBERT-NER
"""

import argparse

import requests
import torch


def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="boltuix/NeuroBERT-NER")

    return parser.parse_args()


def main(args):
    from transformers import AutoConfig, AutoTokenizer

    api_url = f"http://{args.host}:{args.port}/pooling"
    model_name = args.model

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    label_map = config.id2label

    # Input text
    text = "Barack Obama visited Microsoft headquarters in Seattle on January 2025."
    prompt = {"model": model_name, "input": text}

    pooling_response = post_http_request(prompt=prompt, api_url=api_url)

    # Run inference
    output = pooling_response.json()["data"][0]
    logits = torch.tensor(output["data"])
    predictions = logits.argmax(dim=-1)
    inputs = tokenizer(text, return_tensors="pt")

    # Map predictions to labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_map[p.item()] for p in predictions]
    assert len(tokens) == len(predictions)

    # Print results
    for token, label in zip(tokens, labels):
        if token not in tokenizer.all_special_tokens:
            print(f"{token:15} → {label}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
