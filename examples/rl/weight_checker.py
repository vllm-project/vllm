# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify an RLHF inference-weight update with Weight Checker.

Start a development server with real weights in another terminal:

    VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --port 8000

Then run:

    python examples/rl/weight_checker.py --base-url http://localhost:8000

The example follows the Weight Checker lifecycle:

    checksum -> pause -> reset -> reload/transfer -> compare -> resume

The server keeps no baseline state, so the caller holds the first checksum
and sends it back when comparing. That keeps the check valid when requests
are load-balanced over several API server processes.

``reset`` overwrites the weights in place, so the pause brackets it: serving
between the reset and the resume would generate from random weights, and the
prefix cache would keep those blocks. The pause is a caller-side decision
rather than something the endpoint enforces. See
docs/features/weight_checker.md for the rationale.

Pass ``--extra-url`` one or more times to also check that other replicas hold
the same weights. Each one is checked against the same reference baseline
rather than against each other: a checksum key carries the data-parallel rank,
so two replicas hold the same weights under different keys and a pairwise
comparison would call every tensor a mismatch.

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


def rank_prefixes(checksums: dict[str, str]) -> list[str]:
    """Return the rank prefixes a checksum response covers.

    A key is ``dp{..}:pp{..}:pcp{..}:tp{..}:ep{..}:{tensor name}``, so the
    prefix is the first five fields. Splitting on the first five colons keeps a
    tensor name containing a colon intact.
    """
    return sorted({":".join(key.split(":", 5)[:5]) for key in checksums})


def check_weights(
    base_url: str,
    action: str,
    baseline: dict[str, str] | None = None,
    extra_checksums: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run one Weight Checker action, with a baseline and extra reports."""
    payload: dict[str, Any] = {"action": action}
    if baseline is not None:
        payload["baseline"] = baseline
    if extra_checksums:
        payload["checksums"] = extra_checksums
    return post(base_url, "/weight_checker", json=payload)


def reload_inference_weights(base_url: str) -> None:
    """Reload the server's original inference weights from its checkpoint."""
    post(
        base_url,
        "/collective_rpc",
        json={"method": "reload_weights", "timeout": 900},
    )


def verify_weight_update(base_url: str) -> dict[str, str]:
    """Run a complete reset, reload, and byte-for-byte verification cycle.

    The pause is the caller's choice, not a requirement of the endpoint: every
    Weight Checker action works either way. It is here because it keeps the
    random weights from being served, and their blocks from being cached.

    Returns:
        The baseline checksums, which a later replica check can reuse.
    """
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

    # No checksum first: compare hashes the current weights itself, so asking
    # for them here would repeat the work and discard the result.
    print("[4/6] Comparing the current weights with the original baseline...")
    comparison = check_weights(base_url, "compare", original)
    if not comparison["match"]:
        mismatches = comparison["mismatches"]
        preview = "\n".join(f"  - {name}" for name in mismatches[:10])
        raise RuntimeError(
            f"Weight verification failed with {len(mismatches)} mismatches:\n{preview}"
        )

    print("[5/6] Resuming generation...")
    post(base_url, "/resume")

    print("[6/6] Weight verification passed: all inference weights match.")
    return original


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
        help=(
            "Another API server to check against the same baseline. Repeatable."
        ),
    )
    return parser.parse_args()


def check_replicas(extra_urls: list[str], bundle: dict[str, str]) -> None:
    """Check every replica against one baseline.

    Comparing replicas against each other does not work: their checksum keys
    carry their own data-parallel rank, so two replicas hold the same weights
    under different keys and a pairwise comparison would report every tensor as
    a mismatch. The shared reference has to be one baseline, which each replica
    checks itself against.
    """
    ranks = rank_prefixes(bundle)
    print(f"Reference baseline covers {len(bundle)} tensors over {len(ranks)} ranks")

    for url in extra_urls:
        other = check_weights(url, "checksum")["checksums"]
        if rank_prefixes(other) != ranks:
            raise RuntimeError(
                f"{url} covers different ranks than {ranks}, so it cannot be "
                "checked against this baseline"
            )

        result = check_weights(url, "compare", bundle)
        if not result["match"]:
            preview = "\n".join(f"  - {name}" for name in result["mismatches"][:10])
            raise RuntimeError(
                f"{url} disagrees with the baseline on "
                f"{len(result['mismatches'])} tensors:\n{preview}"
            )
        print(f"{url}: matches the baseline over {len(result['ranks'])} ranks")

    print(f"All {len(extra_urls) + 1} replicas hold the same weights.")


def collect_update_digests(base_url: str, weight_version: str) -> dict[str, str]:
    """Finish a weight update and get this instance's digests back.

    The `checksum` option rides on the finish call, so the digests arrive with
    the commit rather than from a separate `checksum` that would hash every
    weight a second time.
    """
    finished = post(
        base_url,
        "/finish_weight_update",
        json={"weight_version": weight_version, "checksum": True},
    )
    digests = finished.get("checksums")
    if not digests:
        raise RuntimeError("finish_weight_update returned no digests")
    print(f"collected {len(digests)} digests for {weight_version}")
    return digests


if __name__ == "__main__":
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    extra_urls = [url.rstrip("/") for url in args.extra_url]
    bundle = verify_weight_update(base_url)
    if extra_urls:
        check_replicas(extra_urls, bundle)
