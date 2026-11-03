# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the deterministic ModelB completion oracle over HTTP."""

import argparse
import json
import urllib.error
import urllib.request


def request_json(
    url: str, payload: dict | None = None, headers: dict[str, str] | None = None
) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}; response body: {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request-id", default="modelb-fixed-oracle-001")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=None,
        help="Pin the request to one data-parallel rank for oracle repeatability.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = (
        {"X-data-parallel-rank": str(args.dp_rank)}
        if args.dp_rank is not None
        else None
    )
    models = request_json(f"{base_url}/v1/models", headers=headers)
    model = models["data"][0]["id"]
    tokenize = request_json(
        f"{base_url}/tokenize",
        {
            "model": model,
            "prompt": "State the capital of France in one word.",
            "add_special_tokens": True,
            "return_token_strs": True,
        },
        headers=headers,
    )
    payload = {
        "model": model,
        "prompt": tokenize["tokens"],
        "add_special_tokens": False,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 17,
        "logprobs": 20,
        "prompt_logprobs": 1,
        "return_token_ids": True,
        "request_id": args.request_id,
    }
    completion = request_json(f"{base_url}/v1/completions", payload, headers=headers)
    print(
        json.dumps(
            {
                "request": payload,
                "tokenize": tokenize,
                "completion": completion,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
