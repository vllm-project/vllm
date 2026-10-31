# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""COPY_TO consumer: bridge a `vllm.copy_to` hint to the offloading connector.

The hint carries an explicit block-hash target (submission-time snapshot,
per the #59708 selector semantics). The offloading connector's request-free
prefetch primitive owns feasibility, reservation, transfer and publication;
this bridge only validates the payload shape and maps the synchronous
outcome onto the hint result vocabulary.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.prefetch import (
    PrefetchSubmitOutcome,
    PrefetchSubmitResult,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_hints.actions import CopyTo, KvHintResult


def apply_copy_to(connector: object, payload: CopyTo, request: object) -> KvHintResult:
    """Execute a COPY_TO hint against the bound offloading connector."""
    request_free_prefetch = getattr(connector, "request_free_prefetch", None)
    if request_free_prefetch is None:
        return KvHintResult(
            "unsupported", reason="No offloading connector with prefetch support"
        )
    if payload.src.tier != "cpu" or payload.dst.tier != "gpu":
        return KvHintResult(
            "unsupported", reason="Only local cpu -> gpu COPY_TO is supported"
        )
    if payload.src.engine_id is not None or payload.dst.engine_id is not None:
        return KvHintResult(
            "unsupported", reason="Cross-engine COPY_TO is not supported"
        )
    try:
        block_hashes = [BlockHash(bytes.fromhex(h)) for h in payload.target]
    except ValueError:
        return KvHintResult("rejected", reason="Target contains a non-hex block hash")
    result: PrefetchSubmitResult = request_free_prefetch(block_hashes)
    return _map_outcome(result)


def _map_outcome(result: PrefetchSubmitResult) -> KvHintResult:
    if result.outcome is PrefetchSubmitOutcome.ACCEPTED:
        return KvHintResult(
            "applied",
            reason=f"submitted prefetch {result.prefetch_id} "
            f"({result.num_blocks_to_load} blocks)",
        )
    if result.outcome is PrefetchSubmitOutcome.ALREADY_AVAILABLE:
        return KvHintResult(
            "applied", reason=f"{result.num_blocks_available} blocks already cached"
        )
    if result.outcome is PrefetchSubmitOutcome.DEFERRED:
        return KvHintResult("deferred", reason="Prefetch not feasible right now")
    if result.outcome is PrefetchSubmitOutcome.MISSING:
        return KvHintResult(
            "rejected", reason="Target does not resolve to offloaded content"
        )
    return KvHintResult(
        "unsupported", reason="Request-free prefetch is disabled or unknown"
    )
