# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.rocm_aiter_mla_prefill import context_row_indices


def test_dcp_context_indices_skip_padding_and_respect_chunk_starts():
    chunk = SimpleNamespace(
        padded_local_seq_lens=[3, 2],
        local_context_lens_allranks=[[7, 6], [2, 1]],
        local_starts=[5, 0],
        num_local_context_tokens=5,
        num_context_tokens=6,
    )
    rows = context_row_indices(chunk, torch.device("cpu"))
    assert rows.tolist() == [0, 1, 5, 3, 4, 8]
    assert rows.dtype == torch.int32


def test_dcp_context_indices_allow_no_local_tokens_for_a_rank():
    chunk = SimpleNamespace(
        padded_local_seq_lens=[3],
        local_context_lens_allranks=[[2, 9]],
        local_starts=[4],
        num_local_context_tokens=3,
        num_context_tokens=3,
    )
    assert context_row_indices(chunk, torch.device("cpu")).tolist() == [3, 4, 5]


def test_dcp_context_indices_reject_inconsistent_token_total():
    chunk = SimpleNamespace(
        padded_local_seq_lens=[3],
        local_context_lens_allranks=[[1, 1]],
        local_starts=[0],
        num_local_context_tokens=3,
        num_context_tokens=3,
    )
    with pytest.raises(AssertionError):
        context_row_indices(chunk, torch.device("cpu"))


def test_compressed_gather_preserves_bytes_and_workspace_partition(monkeypatch):
    from vllm.v1.attention.ops import rocm_aiter_mla_prefill as prefill

    workspace = torch.full((12, 4), -1, dtype=torch.bfloat16)
    cache = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
    chunk = SimpleNamespace(
        num_local_context_tokens=2,
        local_context_lens_allranks=[[2, 2]],
        padded_local_cu_seq_lens=torch.tensor([0, 2]),
        num_requests=1,
        starts=torch.tensor([0]),
    )

    def local_gather(src, dst, *args):
        assert src.dtype == dst.dtype == torch.uint8
        dst.copy_(src)

    def allgather(dst, src):
        assert dst.dtype == src.dtype == torch.uint8
        assert dst.shape == (4, 4)
        assert src.shape == (2, 4)
        dst[:2].copy_(src)
        dst[2:].copy_(src + 16)

    monkeypatch.setattr(prefill.ops, "cp_gather_cache", local_gather)
    gathered = prefill.gather_compressed_context(
        cache, workspace, torch.empty(1, 1), chunk, allgather, torch.float8_e4m3fn
    )
    torch.testing.assert_close(
        gathered.view(torch.uint8), torch.cat((cache, cache + 16))
    )
    # Retain the original logical capacity even though the byte view is larger.
    assert gathered.data_ptr() - workspace.data_ptr() == 4 * 4
    torch.testing.assert_close(workspace.view(torch.uint8).reshape(-1, 4)[:2], cache)


@pytest.mark.parametrize(
    "unsupported",
    ["packed_cache", "e5m2", "quantized_weight", "bias", "no_indices"],
)
def test_dcp_prefill_unsupported_formats_use_original_path(monkeypatch, unsupported):
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("ROCm MLA backend")
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        AiterMLAImpl,
        MLACommonImpl,
    )

    impl = object.__new__(AiterMLAImpl)
    impl.kv_cache_dtype = {"packed_cache": "fp8_ds_mla", "e5m2": "fp8_e5m2"}.get(
        unsupported, "fp8"
    )
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = impl.v_head_dim = 128
    impl.qk_rope_head_dim = 64
    impl.num_heads = 1
    impl.kv_b_proj = SimpleNamespace(
        weight=torch.empty(256, 512, dtype=torch.bfloat16),
        weight_scale=torch.ones(1) if unsupported == "quantized_weight" else None,
        bias=torch.ones(256) if unsupported == "bias" else None,
    )
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(q_data_type=torch.bfloat16),
        dcp_context_row_indices=None if unsupported == "no_indices" else [],
    )
    expected = (torch.ones(1), torch.zeros(1))
    monkeypatch.setattr(
        MLACommonImpl,
        "_context_parallel_compute_prefill_context",
        lambda *args: expected,
    )
    assert (
        impl._context_parallel_compute_prefill_context(
            torch.empty(1, 1, 192, dtype=torch.bfloat16),
            torch.empty(1, 1536, 576, dtype=torch.uint8),
            metadata,
            torch.ones(1),
            8,
        )
        is expected
    )
