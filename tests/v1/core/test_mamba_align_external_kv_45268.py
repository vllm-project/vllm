# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for issue #45268.

Hybrid (FullAttention + Mamba) models running with ``mamba_cache_mode="align"``
and ``enable_prefix_caching`` used to hit a hard ``AssertionError`` in the
scheduler whenever a KV connector reported a prefix-cache hit:

    File ".../vllm/v1/core/sched/scheduler.py", line 286,
        in _mamba_block_aligned_split
    assert num_external_computed_tokens == 0, (
    AssertionError: External KV connector is not verified yet

This killed the EngineCore (``EngineDeadError`` on the next request). It was
reported on ``qwen3.6-27b-fp8`` (TP=2/PP=2 + ``--kv-offloading-backend native
--enable-sleep-mode``) but the minimal trigger is connector-agnostic: any
external (CPU) prefix-cache hit on a mamba-align hybrid. ``ai21labs/Jamba-tiny-dev``
was the published minimal model used to confirm it without sleep mode, via
``warmup -> reset_prefix_cache -> resend``.

PR #42554 removed the assert and gated ``_mamba_block_aligned_split`` on
``not load_kv_async``. This test locks that fix in at the scheduler layer
(CPU-only, no GPU / no model download) by driving a mamba-align hybrid scheduler
through a connector-reported external hit and asserting ``schedule()`` does not
crash -- for both the asynchronous-load path (OffloadingConnector / NIXL) and
the synchronous-hit path the old assert was guarding.
"""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
NUM_BLOCKS = 10000


def _build_mamba_align_scheduler(matched_tokens: int, is_async: bool) -> Scheduler:
    """FA + Mamba hybrid scheduler, mamba_cache_mode='align', prefix caching on,
    with a MockKVConnector reporting ``matched_tokens`` external-hit tokens."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
        watermark=0.0,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    # Drives Scheduler.need_mamba_block_aligned_split.
    cache_config.mamba_cache_mode = "align"

    kv_transfer_config = KVTransferConfig(
        kv_connector="MockKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "matched_tokens": matched_tokens,
            "is_async": is_async,
        },
    )

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(pipeline_parallel_size=1),
        kv_transfer_config=kv_transfer_config,
    )

    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_fa"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(["layer_mamba"], mamba_spec),
        ],
    )
    cache_config.num_gpu_blocks = NUM_BLOCKS
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


@pytest.mark.parametrize("is_async", [True, False])
def test_mamba_align_external_kv_hit_does_not_crash(is_async: bool):
    """A connector-reported external prefix-cache hit on a mamba-align hybrid
    must not crash the scheduler (regression for #45268).

    Pre-#42554 the scheduler hit
    ``assert num_external_computed_tokens == 0`` inside
    ``_mamba_block_aligned_split`` -> EngineDeadError. ``matched_tokens`` is set
    to 3 full blocks so the external hit is non-zero and block-aligned, matching
    the reported repro (CPU prefix-cache hit beyond the local GPU cache).
    """
    matched_tokens = 3 * BLOCK_SIZE  # 48: non-zero, block-aligned external hit
    scheduler = _build_mamba_align_scheduler(matched_tokens, is_async)

    # Sanity: the mamba-align code path under test is actually engaged.
    assert scheduler.need_mamba_block_aligned_split

    request = create_requests(num_requests=1, num_tokens=64, block_size=BLOCK_SIZE)[0]
    scheduler.add_request(request)

    # The historical bug: this call raised AssertionError and killed EngineCore.
    output = scheduler.schedule()

    scheduled = output.num_scheduled_tokens.get(request.request_id, 0)
    if is_async:
        # Async load (OffloadingConnector / NIXL): no local work is scheduled
        # this step while the remote KV transfer is set up.
        assert scheduled == 0
    else:
        # Synchronous hit: forward progress is scheduled, block-aligned to a
        # multiple of block_size (no partial-block mamba checkpoint).
        assert scheduled > 0
        assert scheduled % BLOCK_SIZE == 0


def test_mamba_block_aligned_split_accepts_external_tokens():
    """Direct unit-level guard: ``_mamba_block_aligned_split`` must accept a
    non-zero ``num_external_computed_tokens`` without asserting (the removed
    pre-#42554 assert), and return a block-aligned chunk."""
    scheduler = _build_mamba_align_scheduler(matched_tokens=0, is_async=False)
    request = create_requests(num_requests=1, num_tokens=128, block_size=BLOCK_SIZE)[0]

    # 48 external + 0 local computed; large num_new_tokens to be aligned down.
    num_new_tokens = scheduler._mamba_block_aligned_split(
        request,
        num_new_tokens=80,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=3 * BLOCK_SIZE,
    )
    assert num_new_tokens % BLOCK_SIZE == 0
    assert num_new_tokens > 0
