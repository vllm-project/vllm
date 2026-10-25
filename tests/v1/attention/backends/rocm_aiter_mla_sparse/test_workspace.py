# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workspace and memory of ROCM_AITER_MLA_SPARSE.

Every persistent buffer the builder allocates must be read by a path the
configuration can take, and a fixed workload's peak memory stays in budget.

Per-forward allocations in the hot path (justified):
- the attention output (``[tokens, heads, kv_lora_rank]``), one per layer;
- ``fit_kpool_indices_to_aiter``'s fitted copy, only when the top-k buffer is
  wider than topk (kpool indexers);
- the sink rescale temporaries, only on sink layers.
The q concat buffer comes from the workspace manager, not a fresh allocation.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_mla_sparse.sparse_harness import (
    make_case,
)
from tests.v1.attention.model_shapes import DEEPSEEK_V3_2, GLM_5_3_FLASH
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms.rocm import on_gfx950

DSV32 = DEEPSEEK_V3_2.shard(8)
GLM_FLASH = GLM_5_3_FLASH.shard(8)

# Buffers every configuration reads.
ALWAYS = {
    "req_id_per_token_buffer",
    "qo_indptr",
    "paged_kv_last_page_len",
    "paged_kv_indices",
    "paged_kv_indptr",
    "_context_lens_cpu",
}
WORK = {
    "_mla_work_meta_data",
    "_mla_work_indptr",
    "_mla_work_info_set",
    "_mla_reduce_indptr",
    "_mla_reduce_final_map",
    "_mla_reduce_partial_map",
}
PREFILL = {"chunked_prefill_workspace"}


def _tensors(builder) -> set[str]:
    return {
        name for name, value in vars(builder).items() if isinstance(value, torch.Tensor)
    }


SHARED_WORKSPACE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="SparseMLACommonMetadataBuilder (shared with the CUDA sparse "
    "backends) allocates chunked_prefill_workspace even when no layer has a "
    "dense-prefill backend; dots3_note reads it without one, so the fix "
    "belongs in a separate PR",
)

CONFIGS = [
    pytest.param(DSV32, {}, ALWAYS | WORK | PREFILL, id="default"),
    pytest.param(DSV32, {"kv_cache_dtype": "fp8"}, ALWAYS | WORK | PREFILL, id="fp8"),
    # Sink layers have no dense prefill and never read the work metadata.
    pytest.param(
        DSV32, {"use_sinks": True}, ALWAYS, id="sinks", marks=SHARED_WORKSPACE
    ),
    # bf16 NoPE always runs vLLM's Triton kernels.
    pytest.param(GLM_FLASH, {}, ALWAYS | PREFILL, id="nope_bf16"),
]


@pytest.mark.parametrize("shape, kwargs, used", CONFIGS)
def test_every_builder_buffer_is_read_by_the_config(env, shape, kwargs, used):
    case = make_case(shape, [64], **kwargs)
    allocated = _tensors(case.builder)
    assert allocated <= used, f"unused buffers: {sorted(allocated - used)}"
    assert WORK & used <= allocated


@pytest.mark.skipif(not on_gfx950(), reason="AITER Triton sparse MLA is gfx950")
def test_aiter_triton_path_allocates_no_work_metadata(env, monkeypatch):
    monkeypatch.setattr(rocm_aiter_ops, "_TRITON_SPARSE_MLA", True)
    case = make_case(DSV32, [64])
    assert not (_tensors(case.builder) & WORK)


def test_nope_fp8_allocates_no_work_metadata(env):
    """NoPE fp8 is forced onto AITER Triton, which needs no work metadata."""
    case = make_case(GLM_FLASH, [64], kv_cache_dtype="fp8")
    assert case.layer.impl.use_aiter_sparse_mla
    assert not (_tensors(case.builder) & WORK)


# Peak bytes above the cache/weights for one decode step of 128 requests at
# 4k context and one 4k prefill (DeepSeek V3.2, TP8, top-k 2048). Measured on
# MI355: 12.7 MiB and 277 MiB; the budgets leave about 20% headroom.
PEAK_BUDGET = {"decode": 16 << 20, "prefill": 320 << 20}


@pytest.mark.parametrize("phase", list(PEAK_BUDGET))
def test_peak_memory_budget(env, phase):
    if phase == "decode":
        query_lens, seq_lens = [1] * 128, [4096] * 128
    else:
        query_lens, seq_lens = [4096], [4096]
    case = make_case(DSV32, seq_lens, topk=2048, max_num_batched_tokens=8192)
    batch = case.batch(query_lens, seq_lens)
    query = case.query(batch.num_tokens)
    indices = case.indices(batch)
    case.run(batch, query, indices)  # warm kernels and caches
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    case.run(batch, query, indices)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    assert peak <= PEAK_BUDGET[phase], f"peak {peak / 2**20:.1f} MiB"
