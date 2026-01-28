# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import httpx
from transformers import AutoTokenizer

GEN_ENDPOINT = "http://localhost:8000/inference/v1/generate"
DUMMY_API_KEY = "empty"
MODEL_NAME = "Qwen/Qwen3-0.6B"

transport = httpx.HTTPTransport()
headers = {"Authorization": f"Bearer {DUMMY_API_KEY}"}
client = httpx.Client(
    transport=transport,
    base_url=GEN_ENDPOINT,
    timeout=600,
    headers=headers,
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How many countries are in the EU?"},
]


def main(client):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    payload = {
        "model": MODEL_NAME,
        "token_ids": token_ids,
        "sampling_params": {"max_tokens": 24, "temperature": 0.2, "detokenize": False},
        "stream": False,
    }
    resp = client.post(GEN_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(data)
    print("-" * 50)
    print("Token generation results:")
    res = tokenizer.decode(data["choices"][0]["token_ids"])
    print(res)
    print("-" * 50)


if __name__ == "__main__":
    main(client)
