# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helpers for OBJ tier performance and stress tests.

Covers: S3 config from env, tier/buffer factory, key/job helpers,
drain, throughput math, and cleanup.
"""

import gc
import os
import time
import uuid

import numpy as np
import torch

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult
from vllm.v1.kv_offload.tiering.obj.tier import ObjSecondaryTier

# ---------------------------------------------------------------------------
# S3 credentials
# ---------------------------------------------------------------------------

S3_ENV_VARS = (
    "VLLM_TEST_S3_BUCKET",
    "VLLM_TEST_S3_ENDPOINT",
    "VLLM_TEST_S3_ACCESS_KEY",
    "VLLM_TEST_S3_SECRET_KEY",
)


def get_s3_config() -> dict:
    """Return S3 connection params read from environment variables."""
    return {
        "bucket": os.environ.get("VLLM_TEST_S3_BUCKET", ""),
        "endpoint_override": os.environ.get("VLLM_TEST_S3_ENDPOINT", ""),
        "access_key": os.environ.get("VLLM_TEST_S3_ACCESS_KEY", ""),
        "secret_key": os.environ.get("VLLM_TEST_S3_SECRET_KEY", ""),
        "scheme": os.environ.get("VLLM_TEST_S3_SCHEME", "http"),
    }


def s3_config_available() -> bool:
    return all(os.environ.get(v) for v in S3_ENV_VARS)


# ---------------------------------------------------------------------------
# Tier and buffer factories
# ---------------------------------------------------------------------------

# Unique prefix per process so parallel test runs don't collide in S3.
_SESSION_PREFIX = f"perf/{uuid.uuid4().hex[:8]}"


def make_obj_tier(
    key_prefix: str = _SESSION_PREFIX,
    lookup_mode: str = "dict",
    **kwargs,
) -> ObjSecondaryTier:
    cfg = get_s3_config()
    return ObjSecondaryTier(
        **cfg,
        model_name="perf_model",
        gpu_block_size=16,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype="float32",
        lookup_mode=lookup_mode,
        key_prefix=key_prefix,
        **kwargs,
    )


def make_tier_with_buffer(
    num_blocks: int,
    elements_per_block: int,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> tuple[ObjSecondaryTier, torch.Tensor]:
    """Create an ObjSecondaryTier with a registered primary CPU buffer."""
    tier = make_obj_tier(**kwargs)
    tensor = torch.zeros((num_blocks, elements_per_block), dtype=dtype)
    tier.set_primary_view(memoryview(tensor.numpy()))
    return tier, tensor


# ---------------------------------------------------------------------------
# Key / job helpers
# ---------------------------------------------------------------------------

_CTX = ReqContext()


def unique_key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return JobMetadata(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Drain
# ---------------------------------------------------------------------------

def drain(
    tier: ObjSecondaryTier,
    timeout: float = 120.0,
    poll_interval: float = 0.01,
) -> list[JobResult]:
    """Poll get_finished() until all pending jobs complete or timeout."""
    results: list[JobResult] = []
    deadline = time.perf_counter() + timeout
    while tier._pending_jobs:
        results.extend(tier.get_finished())
        if not tier._pending_jobs:
            break
        if time.perf_counter() > deadline:
            raise TimeoutError(
                f"drain timed out after {timeout}s, "
                f"{len(tier._pending_jobs)} jobs still pending"
            )
        time.sleep(poll_interval)
    return results


# ---------------------------------------------------------------------------
# Throughput math
# ---------------------------------------------------------------------------

# Per-token KV cache bytes for known models.
# Formula: layers × kv_heads × head_dim × 2 (K+V) × dtype_bytes (bf16=2)
KV_BYTES_PER_TOKEN = {
    "Qwen/Qwen3-0.6B": 28 * 8 * 128 * 2 * 2,
    "Qwen/Qwen2.5-1.5B-Instruct": 28 * 2 * 128 * 2 * 2,
    "Qwen/Qwen2.5-3B-Instruct": 36 * 2 * 128 * 2 * 2,
    "Qwen/Qwen2.5-7B-Instruct": 28 * 4 * 128 * 2 * 2,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 32 * 8 * 128 * 2 * 2,
    "meta-llama/Llama-3.2-1B-Instruct": 16 * 8 * 64 * 2 * 2,
    "meta-llama/Meta-Llama-3.1-70B": 80 * 8 * 128 * 2 * 2,
}


def total_bytes(num_blocks: int, elements_per_block: int, dtype: torch.dtype) -> int:
    return num_blocks * elements_per_block * dtype.itemsize  # type: ignore[attr-defined]


def bytes_to_gbs(byte_count: int, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return (byte_count / elapsed) / (1 << 30)


def format_gbs(gbs: float) -> str:
    return f"{gbs:.3f} GB/s" if gbs > 0 else "N/A"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def del_tier_and_cleanup(tier: ObjSecondaryTier) -> None:
    """Shut down the tier and release memory."""
    try:
        tier.shutdown()
        del tier
        gc.collect()
    except Exception as exc:
        print(f"[WARN] Cleanup failed: {exc}")
