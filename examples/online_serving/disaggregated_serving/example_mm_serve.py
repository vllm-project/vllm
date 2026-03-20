# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated multimodal serving: render → generate round-trip.

Demonstrates the two-phase disaggregated flow:
  1. /v1/chat/completions/render  – preprocesses a multimodal chat request
     into token IDs and serialized tensor features.
  2. /inference/v1/generate       – runs inference on the preprocessed tokens.

The render response is passed *directly* to generate with only
``sampling_params`` added, showing that the two endpoints compose with
zero client-side transformation.

Launch the server first:

    vllm serve Qwen/Qwen3-VL-2B-Instruct \
        --dtype bfloat16 --max-model-len 4096 --enforce-eager

Then run this script:

    python example_mm_serve.py
"""

import io

import pybase64 as base64
import requests
from PIL import Image
from transformers import AutoTokenizer

BASE_URL = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


def make_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def main():
    # -- Step 1: Create a test image (solid red) -------------------------
    image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    data_url = make_data_url(image)
    print("Created 224x224 red test image")

    # -- Step 2: Render (preprocess) -------------------------------------
    render_payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": "What color is this image? Answer in one word.",
                    },
                ],
            }
        ],
    }

    print("\n--- Render ---")
    render_resp = requests.post(
        f"{BASE_URL}/v1/chat/completions/render", json=render_payload
    )
    render_resp.raise_for_status()
    render_data = render_resp.json()

    print(f"Response keys: {list(render_data.keys())}")
    print(f"Number of token_ids: {len(render_data['token_ids'])}")

    features = render_data.get("features")
    if features and features.get("kwargs_data"):
        print(f"kwargs_data modalities: {list(features['kwargs_data'].keys())}")
        for modality, items in features["kwargs_data"].items():
            print(
                f"  {modality}: {len(items)} item(s), "
                f"first item keys: {list(items[0].keys()) if items else '(empty)'}"
            )
    else:
        print("WARNING: no kwargs_data in render response")

    # -- Step 3: Generate (inference) ------------------------------------
    # Pass the render output directly — only add sampling_params.
    generate_payload = render_data
    generate_payload["sampling_params"] = {
        "max_tokens": 20,
        "temperature": 0.0,
    }

    print("\n--- Generate ---")
    gen_resp = requests.post(f"{BASE_URL}/inference/v1/generate", json=generate_payload)
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()

    # -- Step 4: Decode & print ------------------------------------------
    output_ids = gen_data["choices"][0]["token_ids"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"Output token count: {len(output_ids)}")
    print(f"Generated text: {text!r}")

    if "red" in text.lower():
        print("\nModel correctly identified the red image.")
    else:
        print(f"\nWARNING: Expected 'red' in output, got: {text!r}")


if __name__ == "__main__":
    main()
