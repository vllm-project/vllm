# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify an RLHF inference-weight update with Weight Checker.

Start a development server with real weights in another terminal:

    VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --port 8000

Then run:

    python examples/rl/weight_checker.py --base-url http://localhost:8000

The example follows the Weight Checker lifecycle:

    checksum -> pause -> reset -> reload/transfer -> checksum -> compare -> resume

The server keeps no baseline state, so the caller holds the first checksum
and sends it back when comparing. That keeps the check valid when requests
are load-balanced over several API server processes.

``reset`` overwrites the weights in place, so it is bracketed by ``/pause`` and
``/resume``: serving between the reset and the reload would generate from random
weights, and the prefix cache would keep those blocks. See
docs/features/weight_checker.md for the full rationale.

Pass ``--extra-url`` one or more times to also check that the replicas agree
with each other, which is a different question from whether they match a
baseline: it compares their checksums against each other and reports the ranks
whose digests differ.

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
    print("[1/6] Computing the original checksums and saving the baseline...")
    original = check_weights(base_url, "checksum")["checksums"]
    print(f"      hashed {len(original)} tensors")

    print("[2/6] Pausing generation so no request reads random weights...")
    post(base_url, "/pause", params={"mode": "abort"})

    print("[3/6] Resetting inference weights...")
    reset = check_weights(base_url, "reset")
    assert reset["status"] == "reset"

    print("      Reloading inference weights from the original checkpoint...")
    reload_inference_weights(base_url)
    # /pause clears the prefix cache for mode=abort, so this is a safety net
    # for callers whose pause configuration kept the blocks.
    reset_cache = post(
        base_url, "/reset_prefix_cache", params={"reset_running_requests": True}
    )
    if not reset_cache["success"]:
        raise RuntimeError("Could not clear the blocks cached from random weights")

    print("[4/6] Computing checksums of the reloaded inference weights...")
    check_weights(base_url, "checksum")

    print("[5/6] Comparing the current weights with the original baseline...")
    comparison = check_weights(base_url, "compare", original)
    if not comparison["match"]:
        mismatches = comparison["mismatches"]
        preview = "\n".join(f"  - {name}" for name in mismatches[:10])
        raise RuntimeError(
            f"Weight verification failed with {len(mismatches)} mismatches:\n{preview}"
        )

    print("[6/6] Resuming generation...")
    post(base_url, "/resume")

    print("Weight verification passed: all inference weights match.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of a vLLM server running in development mode.",
    )
    parser.add_argument(
        "--extra-url",
        action="append",
        default=[],
        metavar="URL",
        help="Another API server to include in a replica consistency check.",
    )
    return parser.parse_args()


def check_replicas(base_url: str, extra_urls: list[str]) -> None:
    """Ask one server whether the replicas reached so far agree.

    Each URL is queried for its own checksums, and the answers are sent back to
    the first server, which is the one that owns the comparison. A single
    report agrees with itself, so the returned ``ranks`` is what tells you the
    check covered the ranks you expected.
    """
    reports = [check_weights(base_url, "checksum")["checksums"]]
    reports += [
        check_weights(url, "checksum")["checksums"] for url in extra_urls
    ]

    result = post(
        base_url,
        "/weight_checker",
        json={"action": "consistency", "checksums": reports},
    )
    print(
        f"compared {result['reports']} reports covering {len(result['ranks'])} "
        f"rank prefixes"
    )
    if not result["consistent"]:
        preview = "\n".join(f"  - {name}" for name in result["mismatches"][:10])
        raise RuntimeError(
            f"Replicas disagree on {len(result['mismatches'])} tensors:\n{preview}"
        )
    print("Replica consistency passed: every report agrees.")


if __name__ == "__main__":
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    verify_weight_update(base_url)
    if args.extra_url:
        check_replicas(base_url, [url.rstrip("/") for url in args.extra_url])
