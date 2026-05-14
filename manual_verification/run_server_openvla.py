# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pybase64 as base64
import torch
from openvla_check_config import (
    CASES_PATH,
    HF_ARTIFACTS_PATH,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SERVER_RESULT_PATH,
)

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"


def request_payload(case: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(case["image_path"])
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "What action should the robot take to "
                            f"{case['instruction']}?"
                        ),
                    },
                ],
            }
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "return_token_ids": True,
    }


def post_json(payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def first_mismatch(
    reference: list[int],
    candidate: list[int],
) -> int | None:
    for idx, (reference_token, candidate_token) in enumerate(zip(reference, candidate)):
        if reference_token != candidate_token:
            return idx
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def main() -> None:
    cases = json.loads(CASES_PATH.read_text())["cases"]
    hf_artifacts = torch.load(
        HF_ARTIFACTS_PATH,
        map_location="cpu",
        weights_only=False,
    )
    hf_outputs = {item["case_id"]: item for item in hf_artifacts["outputs"]}

    results = []
    for case in cases:
        response = post_json(request_payload(case))
        choice = response["choices"][0]
        token_ids = choice["token_ids"]
        hf_token_ids = hf_outputs[case["case_id"]]["generated_token_ids"]
        token_ids_match = token_ids == hf_token_ids

        results.append(
            {
                "case_id": case["case_id"],
                "episode_index": case["episode_index"],
                "frame_index": case["frame_index"],
                "image_path": case["image_path"],
                "instruction": case["instruction"],
                "hf_generated_token_ids": hf_token_ids,
                "server_generated_token_ids": token_ids,
                "server_generated_text": choice["message"]["content"],
                "finish_reason": choice["finish_reason"],
                "usage": response.get("usage"),
                "token_ids_match": token_ids_match,
                "first_mismatch_index": None
                if token_ids_match
                else first_mismatch(hf_token_ids, token_ids),
            }
        )

    num_matches = sum(item["token_ids_match"] for item in results)
    report = {
        "server_url": SERVER_URL,
        "model": MODEL_ID,
        "num_cases": len(results),
        "num_token_exact_matches": num_matches,
        "num_token_mismatches": len(results) - num_matches,
        "token_exact_match_rate": num_matches / len(results) if results else 0.0,
        "cases": results,
    }

    SERVER_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SERVER_RESULT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
