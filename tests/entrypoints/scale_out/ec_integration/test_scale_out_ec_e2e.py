# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for the scale-out EC connector flow (render -> encode -> prefill).

Mimics the bash-script style of ``tests/v1/kv_connector/nixl_integration``:
the instances are started by ``run_scale_out_ec_e2e_test.sh`` and this file is
the HTTP client that drives the disaggregated request path and compares its
output against a single-instance baseline.

Topology under test (started by the bash script):

    render (GPU-less)   -- /v1/chat/completions/render, /derender
    encode (EC producer) -- /inference/v1/generate
    prefill (EC consumer) -- /inference/v1/generate

The client renders each multimodal chat request once, sends the full
``kwargs_data`` to the encode instance, then sends metadata-only features
(``mm_metadata`` without ``kwargs_data``) plus the encode response's
``ec_transfer_params`` to the prefill instance, and finally derenders the
output token ids. The derendered text must match the baseline exactly.

Usage (from the bash script, or manually):
    # Baseline mode (saves outputs):
    python test_scale_out_ec_e2e.py \
        --mode baseline \
        --service_url http://localhost:19603 \
        --model_name Qwen/Qwen3-VL-2B-Instruct \
        --baseline_file /tmp/vllm_scale_out_ec_baseline.txt

    # Disagg mode (compares outputs):
    python test_scale_out_ec_e2e.py \
        --mode disagg \
        --render_url http://localhost:19600 \
        --encode_url http://localhost:19601 \
        --prefill_url http://localhost:19602 \
        --model_name Qwen/Qwen3-VL-2B-Instruct \
        --baseline_file /tmp/vllm_scale_out_ec_baseline.txt
"""

import argparse
import json
import sys
import time

import requests

from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url

MAX_OUTPUT_LEN = 128
REQUEST_TIMEOUT = 300

image_1 = ImageAsset("stop_sign").pil_image.resize((1280, 720))
image_2 = ImageAsset("cherry_blossom").pil_image.resize((1280, 720))
image_1_url = encode_image_url(image_1)
image_2_url = encode_image_url(image_2)

# Each prompt is a chat completion request body; the same body is sent to the
# baseline instance, to /render, and to /derender (as chat_request).
SAMPLE_PROMPTS: list[dict] = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_1_url}},
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ],
        "description": "single image",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_2_url}},
                    {"type": "image_url", "image_url": {"url": image_1_url}},
                    {"type": "text", "text": "Describe these 2 images in detail."},
                ],
            }
        ],
        "description": "two images",
    },
    {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "description": "text only",
    },
]


def chat_request(prompt: dict) -> dict:
    return {
        "model": None,  # filled by the caller
        "messages": prompt["messages"],
        "max_tokens": MAX_OUTPUT_LEN,
        "temperature": 0.0,
        "seed": 42,
    }


def post_json(url: str, payload: dict) -> requests.Response:
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(
            f"POST {url} failed with {response.status_code}: {response.text}"
        )
    return response


def run_baseline(args) -> None:
    outputs = []
    for prompt in SAMPLE_PROMPTS:
        request = chat_request(prompt) | {"model": args.model_name}
        response = post_json(f"{args.service_url}/v1/chat/completions", request).json()
        outputs.append(response["choices"][0]["message"]["content"])
        print(f"[baseline] {prompt['description']}: {outputs[-1][:80]!r}")

    with open(args.baseline_file, "w") as f:
        json.dump(outputs, f)
    print(f"Baseline saved to {args.baseline_file}")


def _assert_mm_features(features: dict) -> None:
    kwargs_data = features.get("kwargs_data") or {}
    mm_metadata = features.get("mm_metadata")
    assert any(item is not None for items in kwargs_data.values() for item in items), (
        f"render response has no kwargs_data items: {features}"
    )
    assert mm_metadata, "render response is missing mm_metadata"
    for modality, hashes in features["mm_hashes"].items():
        assert modality in mm_metadata, f"mm_metadata missing modality {modality}"
        assert len(mm_metadata[modality]) == len(hashes)


def run_disagg(args) -> None:
    with open(args.baseline_file) as f:
        baseline = json.load(f)

    failures = []
    for prompt, expected in zip(SAMPLE_PROMPTS, baseline, strict=True):
        request = chat_request(prompt) | {"model": args.model_name}
        got = run_disagg_one(args, request)
        if got != expected:
            failures.append(
                f"{prompt['description']}:\n"
                f"  baseline: {expected!r}\n  disagg:  {got!r}"
            )
        else:
            print(f"[disagg] {prompt['description']}: matched baseline")

    if failures:
        print("Output mismatch:")
        for failure in failures:
            print(failure)
        sys.exit(1)
    print("All scale-out EC outputs matched the baseline")


def run_disagg_one(args, request: dict) -> str:
    # 1. Render: chat request -> token ids + multimodal features (GPU-less).
    rendered = post_json(
        f"{args.render_url}/v1/chat/completions/render", request
    ).json()
    token_ids = rendered["token_ids"]
    features = rendered.get("features")
    request_id = rendered["request_id"]

    # 2. Encode: full kwargs_data so the vision encoder can run. The
    #    response carries ec_transfer_params for the prefill instance.
    ec_transfer_params = None
    if features is not None:
        _assert_mm_features(features)
        encode_payload = {
            "request_id": request_id,
            "token_ids": token_ids,
            "features": {
                "mm_hashes": features["mm_hashes"],
                "mm_placeholders": features["mm_placeholders"],
                "kwargs_data": features["kwargs_data"],
            },
            "sampling_params": {"max_tokens": 1, "temperature": 0.0},
        }
        encode_response = post_json(
            f"{args.encode_url}/inference/v1/generate", encode_payload
        ).json()
        ec_transfer_params = encode_response.get("ec_transfer_params")
        assert ec_transfer_params, (
            f"encode response has no ec_transfer_params: {encode_response}"
        )

    # 3. Prefill: metadata-only features. kwargs_data is dropped; the
    #    embeddings arrive through the EC connector.
    if features is not None:
        prefill_features = {
            "mm_hashes": features["mm_hashes"],
            "mm_placeholders": features["mm_placeholders"],
            "mm_metadata": features["mm_metadata"],
        }
        _check_metadata_only_rejected_without_ec(args, token_ids, prefill_features)
    else:
        prefill_features = None

    prefill_payload = {
        "request_id": request_id,
        "token_ids": token_ids,
        "sampling_params": {"max_tokens": MAX_OUTPUT_LEN, "temperature": 0.0},
    }
    if prefill_features is not None:
        prefill_payload["features"] = prefill_features
        prefill_payload["ec_transfer_params"] = ec_transfer_params
    prefill_response = post_json(
        f"{args.prefill_url}/inference/v1/generate", prefill_payload
    ).json()
    output_token_ids = prefill_response["choices"][0]["token_ids"]
    assert output_token_ids, f"prefill returned no token ids: {prefill_response}"

    # 4. Derender: token ids -> chat completion response (GPU-less).
    derendered = post_json(
        f"{args.render_url}/v1/chat/completions/derender",
        {
            "model": args.model_name,
            "generate_response": prefill_response,
            "prompt_tokens": len(token_ids),
            "chat_request": request,
        },
    ).json()
    return derendered["choices"][0]["message"]["content"]


def _check_metadata_only_rejected_without_ec(
    args, token_ids: list[int], prefill_features: dict
) -> None:
    """mm_metadata without kwargs_data must be rejected without EC params."""
    payload = {
        "token_ids": token_ids,
        "features": prefill_features,
        "sampling_params": {"max_tokens": 1, "temperature": 0.0},
    }
    response = requests.post(
        f"{args.prefill_url}/inference/v1/generate",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    assert response.status_code in (400, 422), (
        "metadata-only features without ec_transfer_params should be rejected, "
        f"got {response.status_code}: {response.text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "disagg"], required=True)
    parser.add_argument("--service_url", help="baseline single-instance URL")
    parser.add_argument("--render_url", help="GPU-less render server URL")
    parser.add_argument("--encode_url", help="EC producer (encode) URL")
    parser.add_argument("--prefill_url", help="EC consumer (prefill) URL")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--baseline_file", required=True)
    args = parser.parse_args()

    start = time.time()
    if args.mode == "baseline":
        assert args.service_url, "--service_url is required in baseline mode"
        run_baseline(args)
    else:
        assert args.render_url, "--render_url is required in disagg mode"
        assert args.encode_url, "--encode_url is required in disagg mode"
        assert args.prefill_url, "--prefill_url is required in disagg mode"
        run_disagg(args)
    print(f"Mode {args.mode} finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
