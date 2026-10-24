# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA indexer profiling run must reserve the decode logits, which are
sized by max_model_len and allocated outside the CUDA graph pools in eager and
piecewise steps."""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer

MiB = 1024 * 1024


def _profiled_logits_bytes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sm100: bool,
    topk_backend: str,
    dcp_world_size: int,
    max_decode_tokens: int,
    max_model_len: int,
) -> int:
    """Run the indexer's dummy (profiling) path and return the largest
    single allocation it makes, the logits reservation."""
    sizes: list[int] = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        sizes.append(out.numel() * out.element_size())
        return out

    monkeypatch.setattr(torch, "empty", recording_empty)
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    monkeypatch.setattr(
        sparse_indexer,
        "_gather_workspace_shapes",
        lambda *a: (((1,), torch.uint8), ((1,), torch.uint8)),
    )
    monkeypatch.setattr(
        sparse_indexer,
        "current_workspace_manager",
        lambda: SimpleNamespace(get_simultaneous=lambda *s: None),
    )
    monkeypatch.setattr(
        sparse_indexer, "_reserve_row_shard_exchange_transient", lambda *a: None
    )
    monkeypatch.setattr(sparse_indexer.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    monkeypatch.setattr(
        sparse_indexer.current_platform,
        "is_device_capability_family",
        lambda family: sm100 and family == 100,
    )
    sparse_indexer.sparse_attn_indexer(
        hidden_states=torch.empty(4, 4),
        k_cache_prefix="k",
        kv_cache=torch.empty(4, 4),
        q_quant=torch.empty(4, 4),
        q_scale=None,
        k=None,
        weights=torch.empty(4),
        quant_block_size=128,
        scale_fmt=None,
        topk_tokens=8,
        head_dim=64,
        max_model_len=max_model_len,
        total_seq_lens=512,
        topk_indices_buffer=torch.empty((8, 8), dtype=torch.int32),
        skip_k_cache_insert=True,
        use_pcp=False,
        pcp_shard_decode_requests=False,
        dense_mha_metadata_layer_name="",
        dcp_world_size=dcp_world_size,
        topk_backend=topk_backend,
        max_decode_tokens=max_decode_tokens,
    )
    return max(sizes)


@pytest.mark.parametrize(
    ("sm100", "topk_backend", "dcp_world_size", "rows", "expected"),
    [
        # BF16 logits plus the FP32 upcast the default top-k reads; rows are
        # padded to whole 1024-column multiples, as DeepGEMM pads them.
        (True, "auto", 1, 96, 96 * 40_960 * (2 + 4)),
        # DeepSelect ranks the BF16 logits directly.
        (True, "deep_select", 1, 96, 96 * 40_960 * 2),
        # The DCP merge needs FP32 even with DeepSelect.
        (True, "deep_select", 2, 96, 96 * 40_960 * (2 + 4)),
        # DeepGEMM emits FP32 logits elsewhere; nothing is upcast.
        (False, "auto", 1, 96, 96 * 40_960 * 4),
        # Decode logits below the prefill chunks' budget keep the budget.
        (True, "auto", 1, 1, 1 * MiB),
    ],
)
def test_profiling_run_reserves_decode_logits(
    monkeypatch: pytest.MonkeyPatch,
    sm100: bool,
    topk_backend: str,
    dcp_world_size: int,
    rows: int,
    expected: int,
) -> None:
    reserved = _profiled_logits_bytes(
        monkeypatch,
        sm100=sm100,
        topk_backend=topk_backend,
        dcp_world_size=dcp_world_size,
        max_decode_tokens=rows,
        max_model_len=40_000,
    )
    assert reserved == expected
