# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Kimi-K3's ROCm AITER prefill integration."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.model_executor.warmup import kimi_k3_triton_warmup
from vllm.models.kimi_k3.amd.ops import kda_prefill
from vllm.platforms import current_platform


def _requires_aiter_kda_prefill() -> None:
    if not current_platform.is_rocm():
        pytest.skip("Kimi-K3 AITER prefill requires ROCm")


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    return (difference.norm() / (expected.float().norm() + 1e-6)).item()


@pytest.mark.parametrize(
    ("requested", "aiter_enabled", "expected"),
    [
        ("auto", True, "flashkda"),
        ("auto", False, "triton"),
        ("triton", True, "triton"),
        ("flashkda", False, "flashkda"),
    ],
)
def test_resolve_kda_prefill_backend(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    aiter_enabled: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(
        kda_prefill,
        "is_fused_kda_chunk_supported",
        lambda: False,
    )
    monkeypatch.setattr(
        kda_prefill.rocm_aiter_ops,
        "is_enabled",
        lambda: aiter_enabled,
    )

    actual = kda_prefill.resolve_kda_prefill_backend(requested)

    assert actual == expected


def test_kda_conv1d_weight_loader_populates_prefill_and_decode_copies() -> None:
    dims = [8, 8, 8]
    tp_size = 2
    tp_rank = 1
    local_dim = dims[0] // tp_size
    width = 4

    param = torch.nn.Parameter(torch.empty(3 * local_dim, 1, width))
    decode_weight = torch.empty(3, width, local_dim)
    prefill_weight = torch.empty(3 * local_dim, width, dtype=torch.bfloat16)
    loader = kda_prefill.make_kda_conv1d_weight_loader(
        dims,
        tp_size,
        tp_rank,
        decode_weight,
        prefill_weight,
    )

    expected_shards = []
    for shard_id in range(3):
        loaded = torch.arange(
            shard_id * 100,
            shard_id * 100 + dims[shard_id] * width,
            dtype=torch.float32,
        ).view(dims[shard_id], 1, width)
        loader(param, loaded, shard_id)
        expected_shards.append(loaded[local_dim:].squeeze(1))

    expected = torch.cat(expected_shards)
    torch.testing.assert_close(param.squeeze(1), expected)
    torch.testing.assert_close(prefill_weight, expected.to(torch.bfloat16))
    for shard_id, expected_shard in enumerate(expected_shards):
        torch.testing.assert_close(decode_weight[shard_id], expected_shard.T)


def test_aiter_fused_qkv_conv_uses_vllm_metadata_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "aiter.ops.causal_conv1d_fwd_split_qkv"
    fake_module = ModuleType(module_name)
    metadata = object()
    captured: dict[str, Any] = {}

    def fake_conv(**kwargs):
        captured.update(kwargs)
        x = kwargs["x"]
        projection_size = kwargs["k_dim"]
        output = torch.empty(x.shape[1], projection_size, dtype=x.dtype)
        return output, output.clone(), output.clone()

    fake_module.causal_conv1d_split_qkv_hip_fn = fake_conv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    kda_prefill.aiter_causal_conv1d_prefill(
        x=torch.empty(16, 96, dtype=torch.bfloat16),
        weight=torch.empty(96, 4, dtype=torch.bfloat16),
        bias=None,
        conv_state=torch.empty(1, 96, 3, dtype=torch.bfloat16),
        query_start_loc=torch.tensor([0, 16], dtype=torch.int32),
        projection_size=32,
        cache_indices=torch.zeros(1, dtype=torch.int32),
        has_initial_state=torch.zeros(1, dtype=torch.bool),
        metadata=metadata,
    )

    assert captured["block_m"] == 8
    assert captured["metadata"] is metadata


def test_aiter_prefill_warmup_covers_fused_conv_and_flashkda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_heads, head_dim, num_tokens = 2, 128, 96
    projection_size = num_heads * head_dim
    state = torch.empty(1, num_heads, head_dim, head_dim)
    prefill_weight = torch.empty(
        3 * projection_size,
        4,
        dtype=torch.bfloat16,
    )
    layer = SimpleNamespace(
        kda_prefill_backend="flashkda",
        kv_cache=(torch.empty(0), state),
        local_num_heads=num_heads,
        head_dim=head_dim,
        conv_size=4,
        prefill_conv1d_weight=prefill_weight,
        conv1d=SimpleNamespace(bias=None),
        A_log=torch.empty(num_heads),
        dt_bias=torch.empty(projection_size),
        gate_lower_bound=-5.0,
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_conv(**kwargs):
        calls.append(("conv", kwargs))
        output = torch.empty(num_tokens, projection_size, dtype=torch.bfloat16)
        return output, output.clone(), output.clone()

    def fake_kda(**kwargs):
        calls.append(("kda", kwargs))
        return torch.empty_like(kwargs["q"]), torch.empty_like(kwargs["initial_state"])

    monkeypatch.setattr(kda_prefill, "aiter_causal_conv1d_prefill", fake_conv)
    monkeypatch.setattr(kda_prefill, "aiter_kda_prefill", fake_kda)

    kimi_k3_triton_warmup._warm_aiter_kda_prefill(
        cast(Any, layer),
        torch.bfloat16,
        num_tokens,
    )

    assert [name for name, _ in calls] == ["conv", "kda"]
    conv_args = calls[0][1]
    assert conv_args["x"].shape == (num_tokens, 3 * projection_size)
    assert conv_args["conv_state"].shape == (1, 3 * projection_size, 3)
    assert conv_args["weight"] is prefill_weight
    assert conv_args["metadata"] is None
    kda_args = calls[1][1]
    assert kda_args["q"].shape == (1, num_tokens, num_heads, head_dim)
    assert kda_args["initial_state"].shape == (1, num_heads, head_dim, head_dim)


def test_aiter_prefill_warmup_uses_bounded_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_heads, head_dim = 2, 128
    projection_size = num_heads * head_dim
    warmup_tokens = kimi_k3_triton_warmup._AITER_KDA_PREFILL_WARMUP_TOKENS
    state = torch.empty(1, num_heads, head_dim, head_dim)
    layer = SimpleNamespace(
        kda_prefill_backend="flashkda",
        kv_cache=(torch.empty(0), state),
        local_num_heads=num_heads,
        head_dim=head_dim,
        conv_size=4,
        prefill_conv1d_weight=torch.empty(
            3 * projection_size,
            4,
            dtype=torch.bfloat16,
        ),
        conv1d=SimpleNamespace(bias=None),
        A_log=torch.empty(num_heads),
        dt_bias=torch.empty(projection_size),
        gate_lower_bound=-5.0,
    )

    def fake_conv(**kwargs):
        assert kwargs["x"].shape[0] == warmup_tokens
        output = torch.empty(warmup_tokens, projection_size, dtype=torch.bfloat16)
        return output, output.clone(), output.clone()

    def fake_kda(**kwargs):
        return torch.empty_like(kwargs["q"]), torch.empty_like(kwargs["initial_state"])

    monkeypatch.setattr(kda_prefill, "aiter_causal_conv1d_prefill", fake_conv)
    monkeypatch.setattr(kda_prefill, "aiter_kda_prefill", fake_kda)

    kimi_k3_triton_warmup._warm_aiter_kda_prefill(
        cast(Any, layer),
        torch.bfloat16,
        max_num_batched_tokens=warmup_tokens * 4,
    )


@torch.inference_mode()
def test_aiter_fused_qkv_conv_matches_vllm_prefill_conv() -> None:
    _requires_aiter_kda_prefill()
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    torch.manual_seed(11)
    device = "cuda"
    num_heads, head_dim, width = 2, 128, 4
    projection_size = num_heads * head_dim
    packed_size = 3 * projection_size
    query_start_loc = torch.tensor([0, 7, 12], device=device, dtype=torch.int32)
    cache_indices = torch.tensor([2, 1], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device=device)
    x = torch.randn(12, packed_size, device=device, dtype=torch.bfloat16)
    weight = torch.randn(packed_size, width, device=device, dtype=torch.bfloat16)
    conv_state = torch.randn(
        4, packed_size, width - 1, device=device, dtype=torch.bfloat16
    )

    expected_state = conv_state.clone()
    expected = []
    for x_part, weight_part, state_part in zip(
        x.split(projection_size, dim=-1),
        weight.split(projection_size),
        expected_state.split(projection_size, dim=1),
        strict=True,
    ):
        expected.append(
            causal_conv1d_fn(
                x_part.transpose(0, 1),
                weight_part,
                None,
                activation="silu",
                conv_states=state_part,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                metadata=None,
            ).transpose(0, 1)
        )

    actual_state = conv_state.clone()
    actual = kda_prefill.aiter_causal_conv1d_prefill(
        x=x,
        weight=weight,
        bias=None,
        conv_state=actual_state,
        query_start_loc=query_start_loc,
        projection_size=projection_size,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        metadata=None,
    )

    for actual_part, expected_part in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_part, expected_part, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)


@torch.inference_mode()
def test_aiter_flashkda_preserves_vllm_state_layout() -> None:
    _requires_aiter_kda_prefill()
    from vllm.models.kimi_k3.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )

    torch.manual_seed(17)
    device = "cuda"
    num_heads, head_dim, num_tokens = 12, 128, 96
    cu_seqlens = torch.tensor([0, 48, 96], device=device, dtype=torch.int32)
    shape = (1, num_tokens, num_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)
    raw_gate = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    raw_beta = torch.randn(
        1, num_tokens, num_heads, device=device, dtype=torch.bfloat16
    )
    A_log = torch.randn(num_heads, device=device, dtype=torch.float32).abs() * 0.5
    dt_bias = (
        torch.randn(num_heads * head_dim, device=device, dtype=torch.float32) * 0.1
    )
    initial_state = (
        torch.randn(
            2,
            num_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    )

    expected_out, expected_state = chunk_kda_with_fused_gate(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        raw_g=raw_gate.clone(),
        raw_beta=raw_beta.clone(),
        A_log=A_log,
        g_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=initial_state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    actual_out, actual_state = kda_prefill.aiter_kda_prefill(
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    assert _relative_error(actual_out, expected_out) < 2e-2
    assert _relative_error(actual_state, expected_state) < 2e-2
