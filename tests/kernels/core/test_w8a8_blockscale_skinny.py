# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

import vllm._aiter_ops  # noqa: F401
import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

_PAYLOAD_ENV_BY_NAME = {
    "wo_b": "VLLM_W8A8_BLOCKSCALE_SKINNY_WO_B_PAYLOAD",
    "wqa_wkv": "VLLM_W8A8_BLOCKSCALE_SKINNY_WQA_WKV_PAYLOAD",
}
_LEGACY_DEFAULT_PAYLOAD_ENV = "VLLM_W8A8_BLOCKSCALE_SKINNY_PAYLOAD"
_EXPECTED_N_BY_NAME = {
    "wo_b": 4096,
    "wqa_wkv": 1536,
}


def _require_payload(payload_name: str) -> Path:
    env_names = [_PAYLOAD_ENV_BY_NAME[payload_name]]
    if payload_name == "wo_b":
        env_names.append(_LEGACY_DEFAULT_PAYLOAD_ENV)

    for env_name in env_names:
        payload = os.getenv(env_name)
        if not payload:
            continue
        payload_path = Path(payload)
        if not payload_path.is_file():
            pytest.skip(f"Missing payload file from {env_name}: {payload_path}")
        return payload_path

    joined_envs = ", ".join(env_names)
    pytest.skip(f"Set one of [{joined_envs}] to a real runtime payload")


def _load_payload(
    payload_name: str, tokens: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(_require_payload(payload_name), map_location="cpu")
    weight = payload["weight"].cuda()
    weight_scale = payload["weight_scale"].cuda()
    assert int(weight.shape[0]) == _EXPECTED_N_BY_NAME[payload_name]

    if "x_fp8" in payload and "x_scale" in payload:
        x_fp8 = payload["x_fp8"].cuda()
        x_scale = payload["x_scale"].cuda()
    else:
        x_bf16 = payload["x_bf16"].cuda()
        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x_bf16, transpose_scale=True)

    if tokens is not None:
        x_fp8, x_scale = _slice_or_repeat_token_rows(x_fp8, x_scale, tokens)

    assert int(x_fp8.shape[1]) == 4096
    return x_fp8, weight, x_scale, weight_scale


def _slice_or_repeat_token_rows(
    x_fp8: torch.Tensor, x_scale: torch.Tensor, tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    base_tokens = int(x_fp8.shape[0])
    if tokens <= base_tokens:
        return (
            x_fp8[:tokens].contiguous(),
            _slice_scale_tokens(x_scale, tokens, base_tokens),
        )

    repeats = math.ceil(tokens / base_tokens)
    x_fp8 = x_fp8.repeat((repeats, 1))[:tokens].contiguous()
    x_scale = _repeat_scale_tokens(x_scale, tokens, base_tokens, repeats)
    return x_fp8, x_scale


def _slice_scale_tokens(
    x_scale: torch.Tensor, tokens: int, base_tokens: int
) -> torch.Tensor:
    if x_scale.dim() != 2:
        raise AssertionError(f"expected rank-2 x_scale, got {x_scale.shape}")
    if int(x_scale.shape[0]) == base_tokens:
        return x_scale[:tokens].contiguous()
    if int(x_scale.shape[1]) == base_tokens:
        return x_scale[:, :tokens].contiguous()
    raise AssertionError(
        f"unable to determine token dimension for x_scale shape={x_scale.shape}, "
        f"base_tokens={base_tokens}"
    )


def _repeat_scale_tokens(
    x_scale: torch.Tensor, tokens: int, base_tokens: int, repeats: int
) -> torch.Tensor:
    if x_scale.dim() != 2:
        raise AssertionError(f"expected rank-2 x_scale, got {x_scale.shape}")
    if int(x_scale.shape[0]) == base_tokens:
        return x_scale.repeat((repeats, 1))[:tokens].contiguous()
    if int(x_scale.shape[1]) == base_tokens:
        return x_scale.repeat((1, repeats))[:, :tokens].contiguous()
    raise AssertionError(
        f"unable to determine token dimension for x_scale shape={x_scale.shape}, "
        f"base_tokens={base_tokens}"
    )


def _run_candidate(x_fp8, weight, x_scale, weight_scale) -> torch.Tensor:
    out = torch.empty(
        (x_fp8.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x_fp8.device
    )
    cu_count = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    return ops.wvSplitKQBlockScale(
        weight,
        x_fp8,
        x_scale,
        weight_scale,
        out,
        cu_count,
        True,
    )


def _run_baseline(x_fp8, weight, x_scale, weight_scale) -> torch.Tensor:
    return rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
        x_fp8, weight, x_scale, weight_scale, output_dtype=torch.bfloat16
    )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize("tokens", range(1, 9))
def test_w8a8_blockscale_skinny_matches_aiter_real_payload(
    payload_name: str, tokens: int
):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=tokens)
    baseline = _run_baseline(x_fp8, weight, x_scale, weight_scale)
    candidate = _run_candidate(x_fp8, weight, x_scale, weight_scale)

    assert torch.isfinite(candidate).all()
    torch.testing.assert_close(
        candidate.float(),
        baseline.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
def test_w8a8_blockscale_skinny_repeated_launch_determinism(payload_name: str):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name)
    ref = _run_candidate(x_fp8, weight, x_scale, weight_scale)
    for _ in range(5):
        out = _run_candidate(x_fp8, weight, x_scale, weight_scale)
        torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize("misalign_target", ["activation", "weight"])
def test_w8a8_blockscale_skinny_rejects_misaligned_row_stride(
    payload_name: str, misalign_target: str
):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=6)

    if misalign_target == "activation":
        base = torch.empty(
            (x_fp8.shape[0], x_fp8.shape[1] + 1),
            dtype=x_fp8.dtype,
            device=x_fp8.device,
        )
        x_fp8 = base.as_strided(x_fp8.shape, (x_fp8.shape[1] + 1, 1))
        msg = "activation row stride must be a multiple of 16 elements"
    else:
        base = torch.empty(
            (weight.shape[0], weight.shape[1] + 1),
            dtype=weight.dtype,
            device=weight.device,
        )
        weight = base.as_strided(weight.shape, (weight.shape[1] + 1, 1))
        msg = "weight row stride must be a multiple of 16 elements"

    out = torch.empty(
        (x_fp8.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    cu_count = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    with pytest.raises(RuntimeError, match=msg):
        ops.wvSplitKQBlockScale(
            weight,
            x_fp8,
            x_scale,
            weight_scale,
            out,
            cu_count,
            True,
        )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize("misalign_target", ["activation", "weight"])
def test_w8a8_blockscale_skinny_rejects_misaligned_base_pointer(
    payload_name: str, misalign_target: str
):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=6)

    if misalign_target == "activation":
        base = torch.empty(
            (x_fp8.shape[0], x_fp8.shape[1] + 16),
            dtype=x_fp8.dtype,
            device=x_fp8.device,
        )
        x_fp8 = base[:, 1 : 1 + x_fp8.shape[1]]
        assert x_fp8.stride(0) % 16 == 0
        assert x_fp8.data_ptr() % 16 != 0
        msg = "activation data pointer must be 16-byte aligned"
    else:
        base = torch.empty(
            (weight.shape[0], weight.shape[1] + 16),
            dtype=weight.dtype,
            device=weight.device,
        )
        weight = base[:, 1 : 1 + weight.shape[1]]
        assert weight.stride(0) % 16 == 0
        assert weight.data_ptr() % 16 != 0
        msg = "weight data pointer must be 16-byte aligned"

    out = torch.empty(
        (x_fp8.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    cu_count = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    with pytest.raises(RuntimeError, match=msg):
        ops.wvSplitKQBlockScale(
            weight,
            x_fp8,
            x_scale,
            weight_scale,
            out,
            cu_count,
            True,
        )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize("bad_target", ["activation", "weight"])
def test_w8a8_blockscale_skinny_rejects_noncontiguous_k_dimension(
    payload_name: str, bad_target: str
):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=6)

    if bad_target == "activation":
        base = torch.empty(
            (x_fp8.shape[0], x_fp8.shape[1] * 2),
            dtype=x_fp8.dtype,
            device=x_fp8.device,
        )
        x_fp8 = base[:, ::2]
        msg = "activation must have contiguous K dimension"
    else:
        base = torch.empty(
            (weight.shape[0], weight.shape[1] * 2),
            dtype=weight.dtype,
            device=weight.device,
        )
        weight = base[:, ::2]
        msg = "weight must have contiguous K dimension"

    out = torch.empty(
        (x_fp8.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    cu_count = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    with pytest.raises(RuntimeError, match=msg):
        ops.wvSplitKQBlockScale(
            weight,
            x_fp8,
            x_scale,
            weight_scale,
            out,
            cu_count,
            True,
        )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize("bad_target", ["activation_scale", "weight_scale"])
def test_w8a8_blockscale_skinny_rejects_invalid_scale_layout(
    payload_name: str, bad_target: str
):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=6)

    if bad_target == "activation_scale":
        bad_scale = torch.empty(
            (x_fp8.shape[0] + 1, x_scale.shape[1]),
            dtype=x_scale.dtype,
            device=x_scale.device,
        )
        msg = r"activation_scale must be \[tokens, K/128\] or \[K/128, tokens\]"
        activation_scale = bad_scale
        weight_scale_arg = weight_scale
    else:
        bad_scale = torch.empty(
            (weight_scale.shape[0] + 1, weight_scale.shape[1]),
            dtype=weight_scale.dtype,
            device=weight_scale.device,
        )
        msg = r"weight_scale must be \[N/128, K/128\] or \[K/128, N/128\]"
        activation_scale = x_scale
        weight_scale_arg = bad_scale

    out = torch.empty(
        (x_fp8.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    cu_count = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    with pytest.raises(RuntimeError, match=msg):
        ops.wvSplitKQBlockScale(
            weight,
            x_fp8,
            activation_scale,
            weight_scale_arg,
            out,
            cu_count,
            True,
        )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize(
    ("guard_case", "msg"),
    [
        (
            "bpreshuffle_false",
            "wvSplitKQBlockScale first version supports bpreshuffle only",
        ),
        ("tokens_9", r"activation token count must be in \[1, 8\]"),
        ("k_2048", "logical K must be 4096"),
        ("n_1024", r"first version supports bpreshuffle logical N in \{4096, 1536\}"),
    ],
)
def test_w8a8_blockscale_skinny_rejects_unsupported_flag_or_shape(
    payload_name: str, guard_case: str, msg: str
):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=6)

    bpreshuffle = True
    if guard_case == "bpreshuffle_false":
        bpreshuffle = False
    elif guard_case == "tokens_9":
        x_fp8, x_scale = _slice_or_repeat_token_rows(x_fp8, x_scale, 9)
    elif guard_case == "k_2048":
        x_fp8 = x_fp8[:, :2048].contiguous()
        weight = weight[:, :2048].contiguous()
    elif guard_case == "n_1024":
        weight = weight[:1024].contiguous()
        weight_scale = weight_scale[:8].contiguous()
    else:
        raise AssertionError(f"unexpected guard_case={guard_case}")

    out = torch.empty(
        (x_fp8.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    cu_count = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    with pytest.raises(RuntimeError, match=msg):
        ops.wvSplitKQBlockScale(
            weight,
            x_fp8,
            x_scale,
            weight_scale,
            out,
            cu_count,
            bpreshuffle,
        )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize("cu_count", [0, -1])
def test_w8a8_blockscale_skinny_rejects_nonpositive_cu_count(
    payload_name: str, cu_count: int
):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=6)
    out = torch.empty(
        (x_fp8.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=x_fp8.device,
    )
    with pytest.raises(RuntimeError, match="CuCount must be positive"):
        ops.wvSplitKQBlockScale(
            weight,
            x_fp8,
            x_scale,
            weight_scale,
            out,
            cu_count,
            True,
        )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize("tokens", [1, 6, 8])
def test_w8a8_blockscale_skinny_transposed_scale_views(payload_name: str, tokens: int):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=tokens)
    x_scale_t = x_scale.t()
    weight_scale_t = weight_scale.t()

    assert x_scale_t.shape == (x_scale.shape[1], x_scale.shape[0])
    assert weight_scale_t.shape == (weight_scale.shape[1], weight_scale.shape[0])
    if min(x_scale.shape) > 1:
        assert not x_scale_t.is_contiguous()
    assert not weight_scale_t.is_contiguous()

    baseline = _run_baseline(x_fp8, weight, x_scale, weight_scale)
    candidate = _run_candidate(x_fp8, weight, x_scale_t, weight_scale_t)
    torch.testing.assert_close(
        candidate.float(),
        baseline.float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("payload_name", ["wo_b", "wqa_wkv"])
@pytest.mark.parametrize("tokens", [1, 2, 4, 6, 8])
def test_w8a8_blockscale_skinny_cuda_graph_replay(payload_name: str, tokens: int):
    x_fp8, weight, x_scale, weight_scale = _load_payload(payload_name, tokens=tokens)

    cu_count = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    out = torch.empty(
        (x_fp8.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x_fp8.device
    )

    # Warm JIT/extension path before capture.
    ops.wvSplitKQBlockScale(weight, x_fp8, x_scale, weight_scale, out, cu_count, True)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ops.wvSplitKQBlockScale(
            weight,
            x_fp8,
            x_scale,
            weight_scale,
            out,
            cu_count,
            True,
        )

    ref = out.clone()
    for _ in range(5):
        g.replay()
        torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Payload-free synthetic tests.
#
# The tests above require a captured runtime payload (real pre-shuffled weights)
# and are skipped unless VLLM_W8A8_BLOCKSCALE_SKINNY_*_PAYLOAD is set, so they do
# not run in default CI. The tests below construct their own inputs and a pure
# torch reference, so they exercise the kernel's numerics and guards on any
# gfx1151 node without external files.
#
# The kernel reads weights through PyTorch strides (see bpreshuffle_ptr), so a
# logically contiguous [N, K] weight makes it compute a plain strided
# block-scaled GEMM that we can mirror in torch. This is enough to cover the
# scale-layout handling (including the square wo_b case) and the output numerics;
# it deliberately does not assert anything about the physical pre-shuffle
# permutation, which is validated by the payload-based tests.
# ---------------------------------------------------------------------------

_K = 4096
_K_GROUPS = _K // 128


def _skip_if_not_gfx1151() -> None:
    from vllm.platforms.rocm import on_gfx1151

    if not on_gfx1151():
        pytest.skip("wvSplitKQBlockScale is implemented for gfx1151 only")


def _make_synthetic_inputs(
    n: int, tokens: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (weight, activation, activation_scale, weight_scale) for an
    N x K = (n, 4096) block-scaled fp8 GEMM with `tokens` rows of activation.

    Scales are canonical: activation_scale is [tokens, K/128] and weight_scale
    is [N/128, K/128], both contiguous float32.
    """
    torch.manual_seed(seed)
    fp8_dtype = current_platform.fp8_dtype()
    device = "cuda"

    # Keep magnitudes small so the fp8 round-trip stays well inside e4m3 range.
    weight = (torch.randn(n, _K, device=device) * 0.25).to(fp8_dtype)
    activation = (torch.randn(tokens, _K, device=device) * 0.25).to(fp8_dtype)

    weight_scale = (
        torch.rand(n // 128, _K_GROUPS, device=device, dtype=torch.float32) * 0.5 + 0.5
    )
    activation_scale = (
        torch.rand(tokens, _K_GROUPS, device=device, dtype=torch.float32) * 0.5 + 0.5
    )
    return weight, activation, activation_scale, weight_scale


def _reference_blockscale_gemm(
    weight: torch.Tensor,
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch reference matching the kernel's math.

    The kernel upconverts fp8 -> bf16 before the dot product, so dequantize
    through bf16 here too, then apply the per-128 block scales.
    """
    n, k = weight.shape
    tokens = activation.shape[0]
    k_groups = k // 128

    # Dequantize through bf16 (matching the kernel's fp8 -> bf16 upconvert),
    # then apply the per-128 block scales and matmul in fp32.
    w = weight.to(torch.bfloat16).float().view(n, k_groups, 128)
    a = activation.to(torch.bfloat16).float().view(tokens, k_groups, 128)

    # weight_scale is [N/128, K/128]: each scale row covers 128 weight rows.
    w_scale_per_row = weight_scale.repeat_interleave(128, dim=0)  # [N, K/128]
    w = w * w_scale_per_row.view(n, k_groups, 1)
    a = a * activation_scale.view(tokens, k_groups, 1)

    w = w.reshape(n, k)
    a = a.reshape(tokens, k)
    return (a @ w.t()).to(torch.bfloat16)


def _run_op(weight, activation, activation_scale, weight_scale) -> torch.Tensor:
    out = torch.empty(
        (activation.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=activation.device,
    )
    cu_count = torch.cuda.get_device_properties(activation.device).multi_processor_count
    ops.wvSplitKQBlockScale(
        weight, activation, activation_scale, weight_scale, out, cu_count, True
    )
    return out


@pytest.mark.parametrize("n", [4096, 1536])
@pytest.mark.parametrize("tokens", [1, 2, 6, 8])
def test_w8a8_blockscale_skinny_synthetic_matches_reference(n: int, tokens: int):
    _skip_if_not_gfx1151()
    weight, activation, a_scale, w_scale = _make_synthetic_inputs(n, tokens)

    out = _run_op(weight, activation, a_scale, w_scale)
    ref = _reference_blockscale_gemm(weight, activation, a_scale, w_scale)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("n", [4096, 1536])
@pytest.mark.parametrize("tokens", [1, 6, 8])
def test_w8a8_blockscale_skinny_synthetic_transposed_scales(n: int, tokens: int):
    """Transposed scale views must give the same result as the canonical layout.

    For n == 4096 the weight_scale is square (32x32), which is exactly the case
    where shape alone cannot distinguish a canonical [N/128, K/128] tensor from a
    transposed [K/128, N/128] view. The stride-based disambiguation must resolve
    both to identical numerics.
    """
    _skip_if_not_gfx1151()
    weight, activation, a_scale, w_scale = _make_synthetic_inputs(n, tokens)

    canonical = _run_op(weight, activation, a_scale, w_scale)

    a_scale_t = a_scale.t()
    w_scale_t = w_scale.t()
    assert not w_scale_t.is_contiguous()
    transposed = _run_op(weight, activation, a_scale_t, w_scale_t)

    torch.testing.assert_close(
        transposed.float(), canonical.float(), atol=0.0, rtol=0.0
    )


def test_w8a8_blockscale_skinny_synthetic_rejects_ambiguous_square_scale():
    """A square weight_scale with no unit-stride dimension is undecidable and
    must be rejected rather than silently misread."""
    _skip_if_not_gfx1151()
    n, tokens = 4096, 6
    weight, activation, a_scale, w_scale = _make_synthetic_inputs(n, tokens)

    # Strided view of a [32, 64] buffer -> shape (32, 32) with strides (64, 2):
    # neither dimension is contiguous, so the layout is ambiguous.
    base = torch.empty(
        (w_scale.shape[0], w_scale.shape[1] * 2),
        dtype=w_scale.dtype,
        device=w_scale.device,
    )
    ambiguous = base[:, ::2]
    assert ambiguous.shape == w_scale.shape
    assert ambiguous.stride(0) != 1 and ambiguous.stride(1) != 1

    out = torch.empty((tokens, n), dtype=torch.bfloat16, device=activation.device)
    cu_count = torch.cuda.get_device_properties(activation.device).multi_processor_count
    with pytest.raises(RuntimeError, match="square weight_scale layout is ambiguous"):
        ops.wvSplitKQBlockScale(
            weight, activation, a_scale, ambiguous, out, cu_count, True
        )
