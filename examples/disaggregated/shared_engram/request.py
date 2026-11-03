# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send one manual NIXL prefill-to-decode request to the launch.sh workers."""

import argparse
import json
from urllib.request import Request, urlopen


def post(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8100")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8200")
    parser.add_argument("--model", default="deepseek-v41-shared")
    parser.add_argument(
        "--prompt", default="Explain why a CUDA graph needs stable addresses."
    )
    args = parser.parse_args()
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": 1,
        "temperature": 0,
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        },
    }
    prefill = post(f"{args.prefill_url}/v1/completions", payload)
    transfer = prefill.get("kv_transfer_params")
    if not transfer:
        raise RuntimeError("prefill did not return KV transfer parameters")
    decode = post(
        f"{args.decode_url}/v1/completions",
        {**payload, "max_tokens": 32, "kv_transfer_params": transfer},
    )
    usage = decode.get("usage") or {}
    if usage.get("completion_tokens") != 32:
        raise RuntimeError(f"expected 32 output tokens, got {usage}")
    choice = decode["choices"][0]
    print(choice["text"])
    print(
        json.dumps(
            {
                "usage": usage,
                "finish_reason": choice.get("finish_reason"),
            }
        )
    )


if __name__ == "__main__":
    main()
