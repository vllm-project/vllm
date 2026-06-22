# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AITER helper-op tests.

This file owns the ROCm AITER helper ops that do not already have a more
specific home:
- ``rms_norm`` and ``rms_norm2d_with_add``
- ``triton_rotary_embedding``
- ``per_token_quant`` and ``per_tensor_quant``
- ``act_mul_and_fp8_group_quant``
- the fused RMSNorm + quantization helper ops

Ownership boundaries:
- detailed ``group_fp8_quant`` coverage lives in
  ``tests/kernels/quantization/test_rocm_aiter_grouped_quant.py``
- broader ROCm FP8 reference comparisons live in
  ``tests/kernels/quantization/rocm/test_rocm_fp8.py``
- generic RMSNorm + quant fusion coverage lives in
  ``tests/kernels/core/test_fused_quant_layernorm.py``
"""

import importlib
import warnings

import pytest
import torch

from tests.kernels.utils import _assert_deterministic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported

    assert is_aiter_found_and_supported(), (
        "aiter is required on supported ROCm hardware for this test"
    )


def require_fp8():
    assert current_platform.supports_fp8(), (
        "FP8 is expected on supported ROCm hardware for this test"
    )


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = int(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return torch.quantile(values, q).item()


def _assert_close_budget(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    atol: float,
    rtol: float = 0.0,
    pass_rate: float = 0.99999,
    max_violation_factor: float = 3.0,
) -> None:
    actual_f = actual.detach().float().flatten()
    expected_f = expected.detach().float().flatten()
    abs_diff = (actual_f - expected_f).abs()
    allowed = atol + rtol * expected_f.abs()

    total = abs_diff.numel()
    passed = int((abs_diff <= allowed).sum().item())
    failed = total - passed
    allowed_fail_rate = 1.0 - pass_rate
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    p99_abs = _quantile(abs_diff, 0.99)
    p999_abs = _quantile(abs_diff, 0.999)
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()
    max_atol = max_violation_factor * atol
    above_max_count = int((abs_diff > max_atol).sum().item())

    msg = (
        "[rocm_aiter_ops] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={_format_observed_rate(failed, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
        f"abs>{max_atol:g}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(0.0, total)} "
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"p99_abs={p99_abs:.6g} "
        f"p999_abs={p999_abs:.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )
    print(msg)
    if failed > 0:
        warnings.warn(msg, stacklevel=2)

    assert passed / total >= pass_rate, msg
    assert max_abs <= max_atol, msg
    assert mean_abs <= atol * 0.25, msg


def _assert_rel_error_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    mean_limit: float,
    preferred_rel: float,
    max_rel: float,
    max_fail_rate: float,
) -> None:
    rel = (
        (actual.float() - expected.float()).abs()
        / expected.float().abs().clamp_min(1e-5)
    ).flatten()
    total = rel.numel()
    within_preferred_count = int((rel <= preferred_rel).sum().item())
    fail_count = total - within_preferred_count
    above_max_count = int((rel > max_rel).sum().item())
    mean_rel = rel.mean().item()
    max_rel_err = rel.max().item()
    p99 = _quantile(rel, 0.99)
    p999 = _quantile(rel, 0.999)

    msg = (
        "[rocm_aiter_ops] "
        f"{label}: "
        f"rel<={preferred_rel:g} pass={within_preferred_count / total:.4%} "
        f"({within_preferred_count}/{total}) "
        f"fail={_format_observed_rate(fail_count, total)} "
        f"mean_limit={mean_limit:.4%} "
        f"rel>{max_rel:g}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(max_fail_rate, total)} "
        f"mean_rel={mean_rel:.6g} "
        f"max_rel={max_rel_err:.6g} "
        f"p99={p99:.6g} "
        f"p999={p999:.6g}"
    )
    print(msg)
    if fail_count > 0:
        warnings.warn(msg, stacklevel=2)

    assert mean_rel < mean_limit, msg
    assert above_max_count / total <= max_fail_rate, msg


# Env var tests ------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_name", "attr_name"),
    [
        ("VLLM_ROCM_USE_AITER_RMSNORM", "VLLM_ROCM_USE_AITER_RMSNORM"),
        ("VLLM_ROCM_USE_AITER_TRITON_ROPE", "VLLM_ROCM_USE_AITER_TRITON_ROPE"),
        ("VLLM_ROCM_USE_AITER_TRITON_GEMM", "VLLM_ROCM_USE_AITER_TRITON_GEMM"),
        ("VLLM_ROCM_USE_AITER_LINEAR", "VLLM_ROCM_USE_AITER_LINEAR"),
    ],
)
@pytest.mark.parametrize("enabled", [True, False])
def test_aiter_env_flags_follow_exact_bool_parse(
    env_name: str,
    attr_name: str,
    enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """AITER ROCm env flags should follow the exact bool parse contract."""
    with monkeypatch.context() as mp:
        mp.setenv(env_name, "1" if enabled else "0")
        envs = _reload_envs()
        assert getattr(envs, attr_name) is enabled

    _reload_envs()


# Op registration tests ----------------------------------------------------


@pytest.mark.parametrize(
    "op_name",
    [
        "rocm_aiter_rms_norm",
        "rocm_aiter_rmsnorm2d_fwd_with_add",
        "rocm_aiter_triton_rotary_embedding",
        "rocm_aiter_per_token_quant",
        "rocm_aiter_per_tensor_quant",
        "rocm_aiter_act_mul_and_fp8_group_quant",
    ],
)
def test_rocm_aiter_helper_ops_registered(op_name: str):
    """The raw AITER helper ops owned by this file should stay registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, op_name)
    assert callable(getattr(torch.ops.vllm, op_name))


# -- rocm_aiter_rms_norm correctness tests ---------------------------------


def test_rocm_aiter_rms_norm_output_shape():
    """rocm_aiter_rms_norm returns tensor of same shape as input."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    out = rocm_aiter_ops.rms_norm(x, weight, eps)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert not torch.any(torch.isnan(out))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_aiter_rms_norm_dtype(dtype):
    """rocm_aiter_rms_norm preserves input dtype."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    x = torch.randn(16, 256, dtype=dtype)
    weight = torch.ones(256, dtype=dtype)
    out = rocm_aiter_ops.rms_norm(x, weight, 1e-5)
    assert out.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_aiter_rms_norm_vs_torch(dtype):
    """rocm_aiter_rms_norm matches PyTorch manual RMSNorm for float16 and bfloat16."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    M, N = 8, 128
    x = torch.randn(M, N, dtype=dtype)
    weight = torch.ones(N, dtype=dtype)
    eps = 1e-5

    # Reference: float32 RMSNorm for precision
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref = (x.float() / rms * weight.float()).to(dtype)

    out = rocm_aiter_ops.rms_norm(x, weight, eps)
    _assert_close_budget(
        out.float(),
        ref.float(),
        label=f"rms_norm dtype={dtype}",
        atol=1e-2,
        rtol=1e-2,
    )


# -- rocm_aiter_rmsnorm2d_fwd_with_add tests -------------------------------


def test_rocm_aiter_rmsnorm_with_add_output_shapes():
    """rocm_aiter_rmsnorm2d_fwd_with_add returns (normed, residual)
    with correct shapes."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 16, 256
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    out, res_out = rocm_aiter_ops.rms_norm2d_with_add(x, residual, weight, eps)
    assert out.shape == (M, N)
    assert res_out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert not torch.any(torch.isnan(out))


# -- rocm_aiter_per_token_quant tests --------------------------------------


def test_rocm_aiter_per_token_quant_output_shapes():
    """rocm_aiter_per_token_quant returns (quantized, scale) with correct shapes."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    fp8_dtype = current_platform.fp8_dtype()

    x_quant, scale = rocm_aiter_ops.per_token_quant(x, fp8_dtype)
    assert x_quant.shape == (M, N)
    assert x_quant.dtype == fp8_dtype
    assert scale.shape[0] == M  # one scale per token
    assert not torch.any(torch.isnan(scale))


# -- rocm_aiter_per_tensor_quant tests -------------------------------------


def test_rocm_aiter_per_tensor_quant_output_shapes():
    """rocm_aiter_per_tensor_quant returns (quantized, scale) with correct shapes."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    fp8_dtype = current_platform.fp8_dtype()

    x_quant, scale = rocm_aiter_ops.per_tensor_quant(x, fp8_dtype)
    assert x_quant.shape == (M, N)
    assert x_quant.dtype == fp8_dtype
    # Scale is a scalar or single-element tensor
    assert scale.numel() == 1


# -- rocm_aiter_act_mul_and_fp8_group_quant tests --------------------------


def test_rocm_aiter_act_mul_and_fp8_group_quant_output_shapes():
    """act_mul_and_fp8_group_quant halves the last dim (gate+up) and returns FP8."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401 - ensure op is registered

    torch.set_default_device("cuda")
    M, N = 32, 512  # N must be even (gate + up halves)
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)

    x_quant, scale = torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant(
        x, group_size
    )
    N_half = N // 2
    fp8_dtype = current_platform.fp8_dtype()
    assert x_quant.shape == (M, N_half)
    assert x_quant.dtype == fp8_dtype
    expected_scale_cols = (N_half + group_size - 1) // group_size
    assert scale.shape == (M, expected_scale_cols)


def test_rocm_aiter_act_mul_and_fp8_group_quant_fake_implementation():
    """The SiLU-mul FP8 group-quant op should preserve fake-tensor support."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401

    x = torch.randn((32, 512), dtype=torch.bfloat16, device="cuda")
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant,
        (x, 128),
        test_utils=("test_faketensor",),
    )


# ``group_fp8_quant`` has a dedicated sibling file with deeper raw-op coverage:
# tests/kernels/quantization/test_rocm_aiter_grouped_quant.py


# -- Numerical accuracy tests for AITER custom ops -------------------------


def test_rocm_aiter_rmsnorm_with_add_vs_torch():
    """rocm_aiter_rmsnorm2d_fwd_with_add matches manual residual+RMSNorm reference."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    M, N = 16, 256
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    # Reference: add residual, then RMSNorm
    h = x.float() + residual.float()
    rms = h.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref_normed = (h / rms * weight.float()).to(torch.bfloat16)
    ref_residual = (x.float() + residual.float()).to(torch.bfloat16)

    out, res_out = rocm_aiter_ops.rms_norm2d_with_add(x, residual, weight, eps)

    _assert_close_budget(
        out.float(),
        ref_normed.float(),
        label="rmsnorm_with_add normed",
        atol=1e-2,
        rtol=1e-2,
    )
    _assert_close_budget(
        res_out.float(),
        ref_residual.float(),
        label="rmsnorm_with_add residual",
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "AITER Triton RoPE precision gap: max element error ~0.03125 (2^-5 BF16 ULP) "
        "requires atol~=2e-2; NVIDIA CUDA RoPE achieves atol=1e-3 for bf16 "
        "(allclose_default.py). Fix in upstream aiter rope kernel."
    ),
)
def test_rocm_aiter_triton_rotary_embedding_vs_torch():
    """rocm_aiter_triton_rotary_embedding matches manual NeoX-style RoPE reference.

    The AITER kernel calls rope_cached_thd_positions_offsets_2c_fwd_inplace with
    reuse_freqs_front_part=True. This means it reads only the first head_size//2
    entries of cos and sin from the cache and applies them to both halves of the
    head (pairwise NeoX rotation). The cos_sin_cache must be built with the second
    half mirroring the first so the reference and kernel agree.
    """
    require_aiter()
    import vllm._aiter_ops  # noqa: F401 - ensure op is registered

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    num_tokens = 8
    num_heads = 4
    head_size = 64
    half_dim = head_size // 2
    max_pos = 32

    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.long)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=torch.bfloat16)
    key = torch.randn(num_tokens, num_heads * head_size, dtype=torch.bfloat16)

    # Build cos/sin cache with second half mirroring first half.
    # The kernel uses reuse_freqs_front_part=True: it reads only cos/sin[:, :half_dim]
    # and applies those same frequencies to both the first and second
    # halves of the head.
    cos_half = torch.randn(max_pos, half_dim, dtype=torch.bfloat16)
    sin_half = torch.randn(max_pos, half_dim, dtype=torch.bfloat16)
    cos_cache = torch.cat([cos_half, cos_half], dim=-1)  # [max_pos, head_size]
    sin_cache = torch.cat([sin_half, sin_half], dim=-1)  # [max_pos, head_size]
    cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1)  # [max_pos, 2*head_size]

    # Reference: NeoX-style pairwise rotation with front-half frequencies only.
    # rotate_style=0 (NeoX): [x1*c - x2*s, x2*c + x1*s]
    cos_pos = cos_half[positions]  # [num_tokens, half_dim]
    sin_pos = sin_half[positions]  # [num_tokens, half_dim]

    def apply_rope_ref(t: torch.Tensor) -> torch.Tensor:
        t_r = t.float().view(num_tokens, num_heads, head_size)
        c = cos_pos.float().unsqueeze(1)  # [num_tokens, 1, half_dim]
        s = sin_pos.float().unsqueeze(1)
        x1, x2 = t_r[..., :half_dim], t_r[..., half_dim:]
        rotated = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
        return rotated.to(t.dtype).view(num_tokens, num_heads * head_size)

    ref_q = apply_rope_ref(query)
    ref_k = apply_rope_ref(key)

    # AITER in-place RoPE (modifies query/key in-place)
    q_aiter = query.clone()
    k_aiter = key.clone()
    torch.ops.vllm.rocm_aiter_triton_rotary_embedding(
        positions,
        q_aiter,
        k_aiter,
        head_size,
        cos_sin_cache,
        True,  # is_neox style -> rotate_style=0
    )

    _assert_close_budget(
        q_aiter.float(),
        ref_q.float(),
        label="triton_rope query",
        atol=1e-3,
        rtol=1.6e-2,
    )
    _assert_close_budget(
        k_aiter.float(),
        ref_k.float(),
        label="triton_rope key",
        atol=1e-3,
        rtol=1.6e-2,
    )


def test_rocm_aiter_per_token_quant_roundtrip():
    """rocm_aiter_per_token_quant: dequantized output is close to original."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(1)

    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    fp8_dtype = current_platform.fp8_dtype()

    x_quant, scale = rocm_aiter_ops.per_token_quant(x, fp8_dtype)

    # Dequantize: scale is [M] or [M, 1]
    scale_exp = scale.view(M, 1).float()
    x_dequant = x_quant.float() * scale_exp

    _assert_rel_error_quality(
        x_dequant,
        x.float(),
        label="per_token_quant",
        mean_limit=0.05,
        preferred_rel=0.05,
        max_rel=0.5,
        max_fail_rate=0.01,
    )


def test_rocm_aiter_per_tensor_quant_roundtrip():
    """rocm_aiter_per_tensor_quant: dequantized output is close to original."""
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    fp8_dtype = current_platform.fp8_dtype()

    x_quant, scale = rocm_aiter_ops.per_tensor_quant(x, fp8_dtype)

    # Dequantize: scale is scalar
    x_dequant = x_quant.float() * scale.float()

    _assert_rel_error_quality(
        x_dequant,
        x.float(),
        label="per_tensor_quant",
        mean_limit=0.05,
        preferred_rel=0.05,
        max_rel=0.5,
        max_fail_rate=0.01,
    )


def test_rocm_aiter_act_mul_fp8_group_quant_roundtrip():
    """act_mul_and_fp8_group_quant: dequantized output matches SiLU gate reference."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401 - ensure op is registered

    torch.set_default_device("cuda")
    torch.manual_seed(3)

    M, N = 32, 512  # N even: N//2 gate, N//2 up
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)

    x_quant, scale = torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant(
        x, group_size
    )

    N_half = N // 2
    # Reference: SiLU(gate) * up
    gate = x.float()[:, :N_half]
    up = x.float()[:, N_half:]
    ref = torch.sigmoid(gate) * gate * up  # SiGLU

    # Dequantize: scale is [M, num_groups]
    _num_groups = (N_half + group_size - 1) // group_size
    scale_exp = scale.repeat_interleave(group_size, dim=1)[:, :N_half]
    x_dequant = x_quant.float() * scale_exp

    _assert_rel_error_quality(
        x_dequant,
        ref,
        label="act_mul_fp8_group_quant",
        mean_limit=0.1,
        preferred_rel=0.1,
        max_rel=1.0,
        max_fail_rate=0.01,
    )


def test_rocm_aiter_rms_norm_determinism():
    """rocm_aiter_rms_norm produces bitwise-identical results across N runs."""
    require_aiter()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(5)

    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    _assert_deterministic(rocm_aiter_ops.rms_norm, x, weight, eps, n_runs=4)


# -- Op registration tests for fused RMSNorm+quant ops ---------------------


def test_rocm_aiter_rmsnorm_fused_dynamic_quant_registered():
    """rocm_aiter_rmsnorm_fused_dynamic_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_fused_dynamic_quant")


def test_rocm_aiter_rmsnorm_fused_add_dynamic_quant_registered():
    """rocm_aiter_rmsnorm_fused_add_dynamic_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_fused_add_dynamic_quant")


def test_rocm_aiter_rmsnorm_fp8_group_quant_registered():
    """rocm_aiter_rmsnorm_fp8_group_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_fp8_group_quant")


def test_rocm_aiter_rmsnorm_with_add_fp8_group_quant_registered():
    """rocm_aiter_rmsnorm_with_add_fp8_group_quant custom op is registered."""
    require_aiter()
    import vllm._aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_rmsnorm_with_add_fp8_group_quant")


# -- Fused RMSNorm + quantization accuracy tests ---------------------------


def test_rocm_aiter_rmsnorm_fused_dynamic_quant_vs_sequential():
    """Fused RMSNorm+per-token-FP8-quant matches sequential rms_norm->per_token_quant.

    Tests that the fused kernel produces the same result as the two-step
    sequential composition: rms_norm(x) followed by per_token_quant.
    The fused path is used in production for inference throughput.
    """
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    M, N = 32, 512
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()

    # Sequential reference
    normed = rocm_aiter_ops.rms_norm(x, weight, eps)
    ref_q, ref_scale = rocm_aiter_ops.per_token_quant(normed, fp8_dtype)
    ref_dequant = ref_q.float() * ref_scale.float()

    # Fused op
    fused_q, fused_scale = torch.ops.vllm.rocm_aiter_rmsnorm_fused_dynamic_quant(
        x, weight, eps, fp8_dtype
    )
    fused_dequant = fused_q.float() * fused_scale.float()

    assert fused_q.shape == (M, N)
    assert fused_q.dtype == fp8_dtype
    assert fused_scale.shape == (M, 1)

    # Fused vs sequential: both should recover RMSNorm output within FP8 error
    _assert_rel_error_quality(
        fused_dequant,
        ref_dequant,
        label="rmsnorm_fused_dynamic_quant",
        mean_limit=0.05,
        preferred_rel=0.05,
        max_rel=0.5,
        max_fail_rate=0.01,
    )


def test_rocm_aiter_rmsnorm_fused_add_dynamic_quant_vs_reference():
    """Fused (residual-add + RMSNorm + per-token-FP8-quant)
    matches sequential reference.

    Production path: input + residual -> RMSNorm -> FP8 quantize, returning
    both the quantized norm output and the residual sum for the next layer.
    """
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(1)

    M, N = 16, 256
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()

    # Sequential reference: add residual -> rms_norm -> per_token_quant
    h = (x.float() + residual.float()).to(torch.bfloat16)
    normed = rocm_aiter_ops.rms_norm(h, weight, eps)
    ref_q, ref_scale = rocm_aiter_ops.per_token_quant(normed, fp8_dtype)
    ref_residual_out = h

    # Fused op: returns (x_quant, residual_out, scale)
    fused_q, fused_res_out, fused_scale = (
        torch.ops.vllm.rocm_aiter_rmsnorm_fused_add_dynamic_quant(
            x, residual, weight, eps, fp8_dtype
        )
    )

    assert fused_q.shape == (M, N)
    assert fused_q.dtype == fp8_dtype
    assert fused_scale.shape == (M, 1)
    assert fused_res_out.shape == (M, N)

    # Residual output matches x + residual
    _assert_close_budget(
        fused_res_out.float(),
        ref_residual_out.float(),
        label="rmsnorm_fused_add_dynamic_quant residual",
        atol=1e-3,
        rtol=1e-3,
    )
    # Dequantized output matches sequential path
    fused_dequant = fused_q.float() * fused_scale.float()
    ref_dequant = ref_q.float() * ref_scale.float()
    _assert_rel_error_quality(
        fused_dequant,
        ref_dequant,
        label="rmsnorm_fused_add_dynamic_quant",
        mean_limit=0.05,
        preferred_rel=0.05,
        max_rel=0.5,
        max_fail_rate=0.01,
    )


def test_rocm_aiter_rmsnorm_fp8_group_quant_vs_sequential():
    """Fused RMSNorm+FP8-group-quant matches sequential rms_norm->group_fp8_quant.

    Tests both output shapes and dequantized accuracy against the two-step
    sequential composition.
    """
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    M, N = 32, 512
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()
    expected_groups = (N + group_size - 1) // group_size

    # Fused op: (x_quant, scales)
    fused_q, fused_scales = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant(
        x, weight, eps, group_size
    )
    assert fused_q.shape == (M, N)
    assert fused_q.dtype == fp8_dtype
    assert fused_scales.shape == (M, expected_groups)

    # Dequantize and compare to reference: rms_norm -> group quant -> dequant
    normed = rocm_aiter_ops.rms_norm(x, weight, eps)
    ref_q, ref_scales = rocm_aiter_ops.group_fp8_quant(normed, group_size)
    scales_exp = ref_scales.repeat_interleave(group_size, dim=1)[:, :N]
    ref_dequant = ref_q.float() * scales_exp
    fused_scales_exp = fused_scales.repeat_interleave(group_size, dim=1)[:, :N]
    fused_dequant = fused_q.float() * fused_scales_exp

    _assert_rel_error_quality(
        fused_dequant,
        ref_dequant,
        label="rmsnorm_fp8_group_quant",
        mean_limit=0.05,
        preferred_rel=0.05,
        max_rel=0.5,
        max_fail_rate=0.01,
    )


def test_rocm_aiter_rmsnorm_with_add_fp8_group_quant_shapes():
    """Fused (residual-add + RMSNorm + FP8-group-quant) returns correct shapes."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401

    torch.set_default_device("cuda")
    torch.manual_seed(3)

    M, N = 32, 512
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()
    expected_groups = (N + group_size - 1) // group_size

    # Returns (x_quant, residual_out, scales)
    fused_q, fused_res, fused_scales = (
        torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant(
            x, residual, weight, eps, group_size
        )
    )

    assert fused_q.shape == (M, N)
    assert fused_q.dtype == fp8_dtype
    assert fused_res.shape == (M, N)
    assert fused_res.dtype == torch.bfloat16
    assert fused_scales.shape == (M, expected_groups)
    assert not torch.any(torch.isnan(fused_scales))


def test_rocm_aiter_rmsnorm_with_add_fp8_group_quant_residual_accuracy():
    """Fused rmsnorm_with_add_fp8_group_quant residual output matches x + residual."""
    require_aiter()
    require_fp8()
    import vllm._aiter_ops  # noqa: F401
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(4)

    M, N = 16, 256
    group_size = 128
    x = torch.randn(M, N, dtype=torch.bfloat16)
    residual = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.ones(N, dtype=torch.bfloat16)
    eps = 1e-5

    fused_q, fused_res, fused_scales = (
        torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant(
            x, residual, weight, eps, group_size
        )
    )

    # Residual output must equal x + residual
    ref_residual = (x.float() + residual.float()).to(torch.bfloat16)
    _assert_close_budget(
        fused_res.float(),
        ref_residual.float(),
        label="rmsnorm_with_add_fp8_group_quant residual",
        atol=1e-2,
        rtol=1e-2,
    )

    # Dequantized quant output must match rms_norm(x + residual)
    h = ref_residual
    rms = h.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref_normed = (h.float() / rms * weight.float()).to(torch.bfloat16)
    ref_q, ref_scales = rocm_aiter_ops.group_fp8_quant(ref_normed, group_size)
    ref_scales_exp = ref_scales.repeat_interleave(group_size, dim=1)[:, :N]
    ref_dequant = ref_q.float() * ref_scales_exp
    fused_scales_exp = fused_scales.repeat_interleave(group_size, dim=1)[:, :N]
    fused_dequant = fused_q.float() * fused_scales_exp
    _assert_rel_error_quality(
        fused_dequant,
        ref_dequant,
        label="rmsnorm_with_add_fp8_group_quant",
        mean_limit=0.05,
        preferred_rel=0.05,
        max_rel=0.5,
        max_fail_rate=0.01,
    )


# -- End-to-end inference chain test ---------------------------------------


def test_rocm_aiter_rms_norm_then_per_token_quant_e2e():
    """End-to-end: BF16 RMSNorm -> per-token FP8 quantization -> dequantize.

    Simulates the inference path through a transformer layer norm before a
    linear projection: verifies the full chain produces accurate output
    compared to a float32 reference.
    """
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    # Llama-style hidden dim
    M, N = 32, 4096
    x = torch.randn(M, N, dtype=torch.bfloat16)
    weight = torch.randn(N, dtype=torch.bfloat16)  # learned scale
    eps = 1e-5
    fp8_dtype = current_platform.fp8_dtype()

    # Float32 reference for the full chain
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref_normed_f32 = x.float() / rms * weight.float()

    # AITER chain: RMSNorm -> per-token FP8 quant -> dequant
    normed = rocm_aiter_ops.rms_norm(x, weight, eps)
    x_q, scale = rocm_aiter_ops.per_token_quant(normed, fp8_dtype)
    x_dequant = x_q.float() * scale.float()  # scale: [M, 1]

    # Dequantized result should match the float32 reference within FP8 quant error
    _assert_rel_error_quality(
        x_dequant,
        ref_normed_f32,
        label="rms_norm_then_per_token_quant_e2e",
        mean_limit=0.05,
        preferred_rel=0.05,
        max_rel=0.5,
        max_fail_rate=0.01,
    )
    # Shape and dtype checks
    assert x_q.shape == (M, N)
    assert x_q.dtype == fp8_dtype
    assert scale.shape == (M, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_aiter_rms_norm_then_group_fp8_quant_e2e(dtype):
    """End-to-end: RMSNorm (fp16/bf16) -> FP8 group quantization -> dequantize.

    Covers both float16 and bfloat16 inputs to verify dtype-agnostic
    behavior of the RMSNorm+group-quant pipeline.
    """
    require_aiter()
    require_fp8()
    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    M, N = 16, 512
    group_size = 128
    x = torch.randn(M, N, dtype=dtype)
    weight = torch.randn(N, dtype=dtype)
    eps = 1e-5

    # Float32 reference
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    ref_normed_f32 = x.float() / rms * weight.float()

    # AITER chain: RMSNorm -> group FP8 quant -> dequant
    normed = rocm_aiter_ops.rms_norm(x, weight, eps)
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(normed.bfloat16(), group_size)
    scales_exp = scales.repeat_interleave(group_size, dim=1)[:, :N]
    x_dequant = x_fp8.float() * scales_exp

    _assert_rel_error_quality(
        x_dequant,
        ref_normed_f32,
        label=f"rms_norm_then_group_fp8_quant_e2e dtype={dtype}",
        mean_limit=0.1,
        preferred_rel=0.1,
        max_rel=1.0,
        max_fail_rate=0.01,
    )
