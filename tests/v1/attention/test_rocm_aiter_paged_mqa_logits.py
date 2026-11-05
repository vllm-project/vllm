# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("Requires ROCm.", allow_module_level=True)

pytest.importorskip("aiter.paged_mqa_logits")

from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
)

from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

# The DeepSeek-V3.2 / GLM-5 indexer: 32 heads of 128, 64-token pages.
HEADS, DIM, BLOCK = 32, 128, 64


@pytest.fixture
def aiter_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    "batch,next_n,context",
    [(16, 1, 32768), (16, 3, 8192), (256, 3, 8192)],
)
@torch.inference_mode()
def test_aiter_paged_mqa_logits_matches_gluon_call(
    aiter_enabled, batch, next_n, context
):
    """aiter.paged_mqa_logits picks its kernel and knobs per shape; inside each
    row's causal bound it must give the logits of the Gluon call it replaces."""
    assert sparse_mod._aiter_paged_mqa_logits() is not None
    torch.manual_seed(0)
    fp8 = current_platform.fp8_dtype()
    pages = context // BLOCK

    q = torch.randn(batch, next_n, HEADS, DIM, device="cuda").to(fp8)
    # Each page holds BLOCK x DIM fp8 keys followed by BLOCK fp32 scales.
    keys = torch.randn(batch * pages, BLOCK * DIM, device="cuda").to(fp8)
    scales = torch.rand(batch * pages, BLOCK, device="cuda") + 0.5
    kv_cache = torch.cat([keys.view(torch.uint8), scales.view(torch.uint8)], dim=1)
    kv_cache = kv_cache.view(batch * pages, BLOCK, 1, DIM + 4)
    weights = torch.randn(batch * next_n, HEADS, device="cuda")
    # The (B, next_n) table vLLM builds with MTP: row j of a sequence sees
    # context - next_n + 1 + j tokens.
    rows = torch.arange(next_n, device="cuda", dtype=torch.int32)
    seq_lens = (context - next_n + 1 + rows).repeat(batch, 1)
    block_tables = torch.randperm(batch * pages, device="cuda", dtype=torch.int32).view(
        batch, pages
    )

    out = sparse_mod.rocm_fp8_paged_mqa_logits(
        q, kv_cache, weights, seq_lens, block_tables, None, context
    ).clone()
    ref = torch.zeros(batch * next_n, context, device="cuda")
    deepgemm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        ref,
        seq_lens[:, -1].contiguous(),
        block_tables,
        context,
        ChunkK=256,
        Preshuffle=True,
        KVBlockSize=BLOCK,
        WavePerEU=2,
    )

    valid = torch.arange(context, device="cuda") < seq_lens.reshape(-1, 1)
    torch.testing.assert_close(out[valid], ref[valid], rtol=1e-5, atol=1e-4)
