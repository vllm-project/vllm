# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify an RLHF inference-weight update with Weight Checker.

Start a development server with real weights in another terminal:

    VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --port 8000

Then run:

    python examples/rl/weight_checker.py --base-url http://localhost:8000

The example follows the Weight Checker lifecycle:

    checksum -> reset -> reload/transfer -> checksum -> compare

The server keeps no baseline state, so the caller holds the first checksum
and sends it back when comparing. That keeps the check valid when requests
are load-balanced over several API server processes.

For a standalone demonstration, the existing ``collective_rpc`` development
endpoint reloads the inference weights from the configured checkpoint. In a
real RLHF system, replace ``reload_inference_weights`` with the trainer's
normal weight-transfer operation.
"""

import argparse
from typing import Any

import requests


def post(base_url: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """POST to a development endpoint and return its JSON response."""
    response = requests.post(f"{base_url}{path}", timeout=900, **kwargs)
    response.raise_for_status()
    if not response.content:
        return {}
    return response.json()


def check_weights(
    base_url: str, action: str, baseline: dict[str, str] | None = None
) -> dict[str, Any]:
    """Run one Weight Checker action, passing a baseline when comparing."""
    payload: dict[str, Any] = {"action": action}
    if baseline is not None:
        payload["baseline"] = baseline
    return post(base_url, "/weight_checker", json=payload)


def reload_inference_weights(base_url: str) -> None:
    """Reload the server's original inference weights from its checkpoint."""
    post(
        base_url,
        "/collective_rpc",
        json={"method": "reload_weights", "timeout": 900},
    )


def verify_weight_update(base_url: str) -> None:
    """Run a complete reset, reload, and byte-for-byte verification cycle."""
    print("[1/4] Computing the original checksums and saving the baseline...")
    original = check_weights(base_url, "checksum")["checksums"]
    print(f"      hashed {len(original)} tensors")

    print("[2/4] Resetting inference weights...")
    reset = check_weights(base_url, "reset")
    assert reset["status"] == "reset"

    print("      Reloading inference weights from the original checkpoint...")
    reload_inference_weights(base_url)

    print("[3/4] Computing checksums of the reloaded inference weights...")
    check_weights(base_url, "checksum")

    print("[4/4] Comparing the current weights with the original baseline...")
    comparison = check_weights(base_url, "compare", original)
    if not comparison["match"]:
        mismatches = comparison["mismatches"]
        preview = "\n".join(f"  - {name}" for name in mismatches[:10])
        raise RuntimeError(
            f"Weight verification failed with {len(mismatches)} mismatches:\n{preview}"
        )

    print("Weight verification passed: all inference weights match.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of a vLLM server running in development mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify_weight_update(args.base_url.rstrip("/"))
