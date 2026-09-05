# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    _get_packed_decode_launch_config,
)


@pytest.fixture(autouse=True)
def clear_packed_decode_launch_config_cache():
    _get_packed_decode_launch_config.cache_clear()
    yield
    _get_packed_decode_launch_config.cache_clear()


@pytest.mark.parametrize(
    (
        "batch_size",
        "num_key_heads",
        "num_value_heads",
        "key_dim",
        "value_dim",
        "is_sm120",
        "expected_bv",
    ),
    [
        (1, 16, 16, 128, 128, True, 16),
        (24, 16, 16, 128, 128, True, 16),
        (25, 16, 16, 128, 128, True, 32),
        (16, 8, 16, 128, 128, True, 32),
        (16, 16, 8, 128, 128, True, 32),
        (16, 16, 16, 64, 128, True, 32),
        (16, 16, 16, 128, 64, True, 32),
        (16, 16, 16, 128, 128, False, 32),
    ],
)
def test_packed_decode_launch_config(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    num_key_heads: int,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
    is_sm120: bool,
    expected_bv: int,
):
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: is_sm120,
    )

    config = _get_packed_decode_launch_config(
        batch_size,
        num_key_heads,
        num_value_heads,
        key_dim,
        value_dim,
        device_index=0,
    )

    assert config == (expected_bv, 1, 3)


def test_packed_decode_launch_config_cache_is_per_device(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[int, int]] = []

    def is_sm120(capability: int, device_id: int = 0) -> bool:
        calls.append((capability, device_id))
        return device_id == 1

    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        is_sm120,
    )
    shape = (16, 16, 16, 128, 128)

    assert _get_packed_decode_launch_config(*shape, 0) == (32, 1, 3)
    assert _get_packed_decode_launch_config(*shape, 1) == (16, 1, 3)
    assert _get_packed_decode_launch_config(*shape, 1) == (16, 1, 3)
    assert calls == [(120, 0), (120, 1)]


DEVICE = current_platform.device_type

requires_accelerator = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Gated delta rule Triton kernels require a CUDA-alike or XPU device.",
)


@requires_accelerator
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
@pytest.mark.parametrize(
    ("batch_size", "num_key_heads", "num_value_heads", "simulate_sm120"),
    [
        pytest.param(16, 4, 8, False, id="fallback-b16"),
        pytest.param(32, 4, 8, False, id="fallback-b32"),
        pytest.param(16, 16, 16, True, id="sm120-optimized"),
    ],
)
def test_fused_recurrent_packed_decode_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    strided_mixed_qkv: bool,
    batch_size: int,
    num_key_heads: int,
    num_value_heads: int,
    simulate_sm120: bool,
):
    torch.manual_seed(0)

    # Cover generic grouped-value fallback and Qwen3.5-0.8B's optimized shape.
    B = batch_size
    H = num_key_heads
    HV = num_value_heads
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    if simulate_sm120:
        monkeypatch.setattr(
            current_platform,
            "is_device_capability",
            lambda capability, device_id=0: capability == 120,
        )
        assert _get_packed_decode_launch_config(B, H, HV, K, V, 0) == (16, 1, 3)

    device = torch.device(DEVICE)

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn((B, qkv_dim + 64), device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)

    a = torch.randn((B, HV), device=device, dtype=dtype)
    b = torch.randn((B, HV), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases). Index 0 is
    # reserved as NULL_BLOCK_ID (CUDA graph padding), so valid slots start at 1.
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    ssm_state_indices[-3:] = -1

    state0 = torch.randn((B + 1, HV, V, K), device=device, dtype=dtype)
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).unsqueeze(1)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    # Output rows for PAD_SLOT_ID entries are never written (uninitialized in
    # both paths), so compare only the valid rows.
    valid = ssm_state_indices > 0
    torch.testing.assert_close(out_packed[valid], out_ref[valid], rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


@requires_accelerator
def test_packed_decode_keeps_beta_in_fp32():
    device = torch.device(DEVICE)
    dtype = torch.bfloat16

    mixed_qkv = torch.ones((1, 3), device=device, dtype=dtype)
    a = torch.zeros((1, 1), device=device, dtype=dtype)
    b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    params = torch.zeros((1,), device=device, dtype=dtype)
    ssm_state_indices = torch.ones((1,), device=device, dtype=torch.int32)

    state_packed = torch.zeros((2, 1, 1, 1), device=device, dtype=torch.float32)
    out_packed = torch.empty((1, 1, 1, 1), device=device, dtype=dtype)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
    )

    expected_beta = torch.sigmoid(b.float()).squeeze()
    torch.testing.assert_close(
        state_packed[1, 0, 0, 0], expected_beta, rtol=1e-6, atol=1e-6
    )


@requires_accelerator
def test_packed_decode_supports_large_batch_head_grid():
    B, H, HV, K, V = 1024, 8, 64, 1, 1
    device = torch.device(DEVICE)
    gates = torch.empty((B, HV), device=device)
    params = torch.empty((HV,), device=device)
    out = torch.empty((B, 1, HV, V), device=device)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.empty((B, 2 * H * K + HV * V), device=device),
        a=gates,
        b=gates,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=torch.empty((1, HV, V, K), device=device),
        out=out,
        ssm_state_indices=torch.zeros((B,), device=device, dtype=torch.int32),
    )

    assert torch.count_nonzero(out).item() == 0
