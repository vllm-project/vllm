# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    _prepare_gdn_prefill_initial_state,
    _resolve_gdn_prefill_backend,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends import gdn_attn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder


def _make_config(
    *,
    backend: str = "aiter_flydsl",
    head_k_dim: int = 128,
    head_v_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    return SimpleNamespace(
        additional_config={"gdn_prefill_backend": backend},
        model_config=SimpleNamespace(
            dtype=dtype,
            hf_text_config=SimpleNamespace(
                linear_key_head_dim=head_k_dim,
                linear_value_head_dim=head_v_dim,
            ),
        ),
    )


def test_cli_accepts_aiter_flydsl_gdn_prefill_backend():
    with patch.object(qwen_gdn_linear_attn.current_platform, "device_type", "cpu"):
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--gdn-prefill-backend", "aiter_flydsl"])
    assert args.gdn_prefill_backend == "aiter_flydsl"


@pytest.mark.parametrize(
    "available,head_k_dim,head_v_dim,dtype,expected",
    [
        (True, 128, 128, torch.bfloat16, "aiter_flydsl"),
        (False, 128, 128, torch.bfloat16, "triton"),
        (True, 64, 128, torch.bfloat16, "triton"),
        (True, 128, 64, torch.bfloat16, "triton"),
        (True, 128, 128, torch.float16, "triton"),
    ],
)
def test_resolve_aiter_flydsl_gdn_prefill_backend(
    available: bool,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
    expected: str,
):
    config = _make_config(
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        dtype=dtype,
    )
    with (
        patch.object(
            qwen_gdn_linear_attn.current_platform, "is_rocm", return_value=True
        ),
        patch.object(
            qwen_gdn_linear_attn.rocm_aiter_ops,
            "is_gdn_flydsl_prefill_available",
            return_value=available,
        ),
    ):
        requested, active = _resolve_gdn_prefill_backend(config)

    assert requested == "aiter_flydsl"
    assert active == expected


def test_aiter_flydsl_dispatch_arguments(monkeypatch: pytest.MonkeyPatch):
    q = torch.empty(1, 8, 2, 128, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, 8, 4, 128, dtype=torch.bfloat16)
    g = torch.empty(1, 8, 4, dtype=torch.float32)
    beta = torch.empty_like(g)
    state_pool = torch.empty(7, 4, 128, 128, dtype=torch.float32)
    state_indices = torch.tensor([2, 5], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    prefill_metadata = object()
    expected_o = torch.empty_like(v)
    captured = {}

    def fake_aiter(**kwargs):
        captured.update(kwargs)
        return expected_o, state_pool

    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "_aiter_flydsl_chunk_gated_delta_rule",
        fake_aiter,
    )
    output, final_state = ChunkGatedDeltaRule.forward_aiter_flydsl(
        None,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state_pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        prefill_metadata=prefill_metadata,
        initial_state_indices=state_indices,
    )

    assert output is expected_o
    assert final_state is state_pool
    assert captured["use_chunk_flydsl"] is True
    assert captured["use_prepare_flydsl"] is True
    assert captured["state_dtype"] is torch.float32
    assert captured["prefill_metadata"] is prefill_metadata
    assert captured["initial_state_indices"] is state_indices
    assert captured["inplace_final_state"] is True
    assert captured["use_qk_l2norm_in_kernel"] is False


def test_aiter_metadata_is_built_once_from_host_sequence_lengths(monkeypatch):
    builder = object.__new__(GDNAttentionMetadataBuilder)
    builder.gdn_prefill_backend = "aiter_flydsl"
    cu_seqlens = torch.tensor([0, 70, 200], dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.clone()
    expected_metadata = object()
    captured = {}

    def fake_build(seq_lens_cpu, *, cu_seqlens):
        captured["seq_lens_cpu"] = seq_lens_cpu
        captured["cu_seqlens"] = cu_seqlens
        return expected_metadata

    monkeypatch.setattr(
        gdn_attn,
        "_build_aiter_flydsl_prefill_metadata",
        fake_build,
    )
    chunk_indices, chunk_offsets, metadata = builder._build_chunk_metadata(
        cu_seqlens,
        cu_seqlens_cpu,
        torch.device("cpu"),
    )

    assert chunk_indices is None
    assert chunk_offsets is None
    assert metadata is expected_metadata
    assert captured == {
        "seq_lens_cpu": [70, 130],
        "cu_seqlens": cu_seqlens,
    }


def test_indexed_prefill_state_pool_zeroes_only_fresh_slots():
    state_pool = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 1, 2, 2)
    state_pool_before = state_pool.clone()
    state_indices = torch.tensor([3, 1, 4], dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True])

    initial_state, initial_state_indices = _prepare_gdn_prefill_initial_state(
        state_pool,
        state_indices,
        has_initial_state,
        use_indexed_state_pool=True,
    )

    assert initial_state is state_pool
    assert initial_state_indices is state_indices
    assert torch.count_nonzero(state_pool[1]) == 0
    assert torch.equal(state_pool[3], state_pool_before[3])
    assert torch.equal(state_pool[4], state_pool_before[4])
    assert torch.equal(state_pool[0], state_pool_before[0])
    assert torch.equal(state_pool[2], state_pool_before[2])
