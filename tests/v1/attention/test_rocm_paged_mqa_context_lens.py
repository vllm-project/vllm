# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.platforms import current_platform

if not current_platform.is_rocm() or not is_aiter_found_and_supported():
    pytest.skip("Requires ROCm with AITER.", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

HEADS, DIM, BLOCK = 32, 128, 64


@torch.inference_mode()
def test_per_row_table_matches_per_sequence_lengths(monkeypatch):
    """With MTP, native decode hands the paged MQA logits a (B, next_n) table,
    [b, j] = L_b - next_n + 1 + j. AITER reads its last column, so the logits
    must equal those of the [B] per-sequence lengths (#59420)."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    if sparse_mod.paged_mqa_logits_module() is None:
        pytest.skip("AITER paged MQA logits are not installed.")
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))
    try:
        torch.manual_seed(0)
        fp8 = current_platform.fp8_dtype()
        lens = torch.tensor([8192, 1500, 3000, 700], device="cuda", dtype=torch.int32)
        batch, next_n, max_model_len = lens.numel(), 3, 8192
        pages = max_model_len // BLOCK

        q = torch.randn(batch, next_n, HEADS, DIM, device="cuda").to(fp8)
        keys = torch.randn(batch * pages, BLOCK * DIM, device="cuda").to(fp8)
        scales = torch.rand(batch * pages, BLOCK, device="cuda") + 0.5
        kv_cache = torch.cat(
            [keys.view(torch.uint8), scales.view(torch.uint8)], dim=1
        ).view(batch * pages, BLOCK, 1, DIM + 4)
        weights = torch.randn(batch * next_n, HEADS, device="cuda")
        block_tables = torch.randperm(
            batch * pages, device="cuda", dtype=torch.int32
        ).view(batch, pages)
        rows = torch.arange(next_n, device="cuda", dtype=torch.int32)
        table = lens[:, None] - next_n + 1 + rows

        def logits(context_lens):
            return sparse_mod.rocm_fp8_paged_mqa_logits(
                q, kv_cache, weights, context_lens, block_tables, None, max_model_len
            ).clone()

        out = logits(table)
        ref = logits(lens)

        valid = torch.arange(max_model_len, device="cuda") < table.reshape(-1, 1)
        assert torch.equal(out[valid], ref[valid])
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()
