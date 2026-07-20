# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import warnings

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.utils.platform_utils import num_compute_units

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only tests"
)

DTYPES = [torch.bfloat16, torch.float16]
BIAS_MODES = [0, 1, 2]
DISTRIBUTIONS = ["normal", "mixed_scale", "sparse_activations"]

LLMM1_SHAPES = [
    (1, 128, 256),
    (1, 512, 1024),
    (1, 2048, 4096),
    (1, 6144, 1024),
    (1, 4096, 8192),
]

WVSPLITK_SHAPES = [
    (1, 64, 64),
    (2, 256, 256),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (1, 9216, 512),
    (4, 16384, 8192),
]

WVSPLITK_REALISTIC_SHAPES = [
    (1, 4096, 32000),
    (4, 4096, 32000),
    (1, 8192, 28672),
    (1, 4096, 151936),
]

WVSPLITKRC_SHAPES = [
    (16, 2880, 128),
    (32, 3072, 256),
    (64, 2888, 640),
    (128, 3080, 656),
]

FP8_SHAPES = [
    (1, 64, 64),
    (3, 512, 528),
    (4, 2064, 2064),
    (4, 4096, 4096),
    (1, 14336, 1024),
    (2, 24576, 2048),
]

LLMM1_ERROR_BUDGETS = {
    "default": {
        torch.bfloat16: (2e-3, 4e-3, 0.999),
        torch.float16: (5e-4, 1e-3, 0.999),
    },
    "mixed_scale": {
        torch.bfloat16: (1.5625e-2, 3.125e-2, 0.98),
        torch.float16: (4e-3, 8e-3, 0.98),
    },
}

WVSPLITK_ERROR_BUDGETS = {
    torch.bfloat16: (5e-4, 4e-3, 0.999),
    torch.float16: (5e-4, 1e-3, 0.999),
}

WVSPLITKRC_ERROR_BUDGETS = {
    torch.bfloat16: (5e-4, 4e-3, 0.999),
    torch.float16: (5e-4, 1e-3, 0.999),
}

WVSPLITKRC_FEATURE_ERROR_BUDGETS = {
    (torch.bfloat16, False): (5e-4, 4e-3, 0.999),
    (torch.float16, False): (4e-3, 3.125e-2, 0.9995),
    (torch.bfloat16, True): (1e-4, 1e-4, 1.0),
    (torch.float16, True): (5e-5, 1e-4, 1.0),
}

FP8_ERROR_BUDGETS = {
    torch.bfloat16: (1e-4, 1e-4, 1.0),
    torch.float16: (2e-4, 5e-4, 0.9999),
}


def _pad_last_dim(x: torch.Tensor) -> torch.Tensor:
    num_pad = 256 // x.element_size()
    import torch.nn.functional as F

    return F.pad(x, (0, num_pad), "constant", 0)[..., :-num_pad]


def _mixed_scale_multiplier(dtype: torch.dtype) -> float:
    return min(50.0, 0.25 / torch.finfo(dtype).eps)


def _make_inputs(
    n: int,
    k: int,
    m: int,
    dtype: torch.dtype,
    distribution: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    xavier = math.sqrt(2 / k)

    if distribution == "normal":
        a = torch.randn(n, k, dtype=dtype, device="cuda") * xavier
        b = torch.randn(m, k, dtype=dtype, device="cuda") * xavier
    elif distribution == "mixed_scale":
        a = torch.randn(n, k, dtype=dtype, device="cuda") * xavier
        b = torch.randn(m, k, dtype=dtype, device="cuda") * xavier
        hot = torch.randperm(k, device="cuda")[: max(1, k // 10)]
        b[:, hot] *= _mixed_scale_multiplier(dtype)
    elif distribution == "sparse_activations":
        a = torch.randn(n, k, dtype=dtype, device="cuda").clamp(min=0) * xavier
        b = torch.randn(m, k, dtype=dtype, device="cuda") * xavier
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return a, b


def _make_inputs_fp8(
    n: int,
    k: int,
    m: int,
    distribution: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = _make_inputs(n, k, m, torch.float32, distribution, seed)
    a_fp8, scale_a = ref_dynamic_per_tensor_fp8_quant(a)
    b_fp8, scale_b = ref_dynamic_per_tensor_fp8_quant(b)
    return a_fp8, b_fp8, scale_a, scale_b


def _make_bias(
    n: int,
    m: int,
    dtype: torch.dtype,
    bias_mode: int,
    seed: int = 42,
) -> torch.Tensor | None:
    torch.manual_seed(seed)
    if bias_mode == 0:
        return None
    if bias_mode == 1:
        return torch.rand(m, dtype=dtype, device="cuda") * 2 - 1
    if bias_mode == 2:
        return torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1
    raise ValueError(f"Unknown bias_mode: {bias_mode}")


def _fits_wvsplitkrc(n: int, k: int, m: int, cu_count: int) -> bool:
    n_p2 = 1 << (n - 1).bit_length()
    rndup_cus = ((m + 63) // 64) * ((k + 511) // 512)
    groups_sharing_b = min(n_p2 // 16, 4)
    cu_needed = rndup_cus * groups_sharing_b
    fits_scratch = (n_p2 * m * ((k + 511) // 512)) <= 128 * 1024 * 12
    return fits_scratch and cu_needed <= cu_count


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = math.floor(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _assert_deterministic(fn, runs: int = 4) -> None:
    ref = fn()
    for _ in range(runs - 1):
        out = fn()
        assert torch.equal(out, ref), (
            f"non-deterministic output, max diff="
            f"{(out.float() - ref.float()).abs().max().item():.2e}"
        )


def _assert_logprobs_close(
    out_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    *,
    max_top1_logprob_diff: float = 0.01,
    min_top1_match: float = 0.99,
    min_top5_overlap: float = 0.95,
) -> None:
    ref_logprobs = torch.nn.functional.log_softmax(ref_logits.float(), dim=-1)
    out_logprobs = torch.nn.functional.log_softmax(out_logits.float(), dim=-1)

    ref_top1 = ref_logprobs.argmax(dim=-1)
    out_top1 = out_logprobs.argmax(dim=-1)
    top1_match = (ref_top1 == out_top1).float().mean().item()
    assert top1_match >= min_top1_match, (
        f"top-1 match {top1_match:.4f} < {min_top1_match:.4f}"
    )

    k = min(5, ref_logits.shape[-1])
    ref_top5 = ref_logprobs.topk(k, dim=-1).indices
    out_top5 = out_logprobs.topk(k, dim=-1).indices
    top5_overlap = (
        sum(
            len(set(ref_row.tolist()) & set(out_row.tolist())) / k
            for ref_row, out_row in zip(ref_top5, out_top5)
        )
        / ref_top5.shape[0]
    )
    assert top5_overlap >= min_top5_overlap, (
        f"top-5 overlap {top5_overlap:.4f} < {min_top5_overlap:.4f}"
    )

    top1_logprob_diff = (
        (
            ref_logprobs.gather(1, ref_top1.unsqueeze(1))
            - out_logprobs.gather(1, ref_top1.unsqueeze(1))
        )
        .abs()
        .max()
        .item()
    )
    assert top1_logprob_diff <= max_top1_logprob_diff, (
        f"top-1 logprob diff {top1_logprob_diff:.6f} > {max_top1_logprob_diff:.6f}"
    )


def _assert_with_error_budget(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    tight_atol: float,
    max_atol: float,
    pass_rate: float,
    label: str | None = None,
) -> None:
    abs_err = (actual.detach().float() - expected.detach().float()).abs()
    total = abs_err.numel()
    tight_count = int((abs_err <= tight_atol).sum().item())
    above_tight_count = total - tight_count
    above_max_count = int((abs_err > max_atol).sum().item())
    tight_rate = tight_count / total
    above_tight_rate = above_tight_count / total
    allowed_fail_rate = 1.0 - pass_rate
    above_max_rate = above_max_count / total
    max_fail_rate = 0.0
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    p99_err = torch.quantile(abs_err.flatten(), 0.99).item()
    p999_err = torch.quantile(abs_err.flatten(), 0.999).item()
    prefix = f"[rocm_skinny_gemms] {label}: " if label else "[rocm_skinny_gemms] "
    stats = (
        f"{prefix}tight<={tight_atol:.6g} "
        f"pass={_format_observed_rate(tight_count, total)} "
        f"fail={_format_observed_rate(above_tight_count, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"max<={max_atol:.6g} "
        f"above_max={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(max_fail_rate, total)} "
        f"max_diff={max_err:.6g} mean_diff={mean_err:.6g} "
        f"p99={p99_err:.6g} p999={p999_err:.6g}"
    )
    print(stats)

    assert tight_rate >= pass_rate, (
        f"tight accuracy pass rate {tight_rate:.6f} < {pass_rate} "
        f"(tight_atol={tight_atol})\n{stats}"
    )

    assert above_max_rate <= max_fail_rate, (
        f"above-max rate {above_max_rate:.6f} > {max_fail_rate:.6f}\n{stats}"
    )

    if above_tight_rate > 0.0:
        warnings.warn(
            f"tolerance mismatch within allowed budget; {stats}",
            stacklevel=2,
        )


@pytest.mark.parametrize("n,k,m", LLMM1_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows_per_block", [4, 8])
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_llmm1_accuracy(n, k, m, dtype, rows_per_block, distribution):
    a, b = _make_inputs(n, k, m, dtype, distribution)
    out = ops.LLMM1(b, a, rows_per_block)
    ref = torch.matmul(a, b.t())
    budget_name = "mixed_scale" if distribution == "mixed_scale" else "default"
    tight_atol, max_atol, pass_rate = LLMM1_ERROR_BUDGETS[budget_name][dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=(
            f"LLMM1 shape=({n},{k},{m}) dtype={dtype} "
            f"rows_per_block={rows_per_block} distribution={distribution}"
        ),
    )


@pytest.mark.parametrize("n,k,m", [(1, 2048, 4096), (1, 4096, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_llmm1_determinism(n, k, m, dtype):
    a, b = _make_inputs(n, k, m, dtype, "normal")
    _assert_deterministic(lambda: ops.LLMM1(b, a, 4))


@pytest.mark.parametrize("n,k,m", WVSPLITK_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
def test_wvsplitk_accuracy(n, k, m, dtype, distribution, bias_mode):
    a, b = _make_inputs(n, k, m, dtype, distribution)
    bias = _make_bias(n, m, dtype, bias_mode)
    out = ops.wvSplitK(b, a.view(-1, a.size(-1)), num_compute_units(), bias)
    ref = torch.nn.functional.linear(a, b, bias)

    tight_atol, max_atol, pass_rate = WVSPLITK_ERROR_BUDGETS[dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=(
            f"wvSplitK shape=({n},{k},{m}) dtype={dtype} "
            f"distribution={distribution} bias_mode={bias_mode}"
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("padded_b", [False, True])
def test_wvsplitk_features(dtype, bias_mode, padded_a, padded_b):
    n, k, m = 4, 4096, 4096
    a, b = _make_inputs(n, k, m, dtype, "normal")
    bias = _make_bias(n, m, dtype, bias_mode)

    if padded_a:
        a = _pad_last_dim(a)
    if padded_b:
        b = _pad_last_dim(b)

    out = ops.wvSplitK(b, a.view(-1, a.size(-1)), num_compute_units(), bias)
    ref = torch.nn.functional.linear(a, b, bias)
    tight_atol, max_atol, pass_rate = WVSPLITK_ERROR_BUDGETS[dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=(
            f"wvSplitK features dtype={dtype} bias_mode={bias_mode} "
            f"padded_a={padded_a} padded_b={padded_b}"
        ),
    )


@pytest.mark.parametrize("n,k,m", WVSPLITK_REALISTIC_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", ["normal", "sparse_activations"])
def test_wvsplitk_realistic_shapes(n, k, m, dtype, distribution):
    a, b = _make_inputs(n, k, m, dtype, distribution)
    out = ops.wvSplitK(b, a.view(-1, a.size(-1)), num_compute_units())
    ref = torch.nn.functional.linear(a, b)

    tight_atol, max_atol, pass_rate = WVSPLITK_ERROR_BUDGETS[dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=(
            f"wvSplitK realistic shape=({n},{k},{m}) dtype={dtype} "
            f"distribution={distribution}"
        ),
    )


@pytest.mark.parametrize("n,k,m", [(1, 2048, 4096), (4, 4096, 4096), (1, 9216, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitk_determinism(n, k, m, dtype):
    a, b = _make_inputs(n, k, m, dtype, "normal")
    _assert_deterministic(
        lambda: ops.wvSplitK(b, a.view(-1, a.size(-1)), num_compute_units())
    )


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
@pytest.mark.parametrize("n,k,m", WVSPLITKRC_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitkrc_accuracy(n, k, m, dtype):
    cu_count = num_compute_units()
    if not _fits_wvsplitkrc(n, k, m, cu_count):
        pytest.skip("shape exceeds wvSplitKrc launch limits")

    a, b = _make_inputs(n, k, m, dtype, "normal")
    out = ops.wvSplitKrc(a, b, cu_count, None)
    ref = torch.nn.functional.linear(a, b)

    tight_atol, max_atol, pass_rate = WVSPLITKRC_ERROR_BUDGETS[dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=f"wvSplitKrc shape=({n},{k},{m}) dtype={dtype}",
    )


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("xnorm", [False, True])
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.parametrize("padded_a", [False, True])
def test_wvsplitkrc_features(dtype, xnorm, bias_mode, padded_a):
    n, k, m = 64, 3072, 640
    cu_count = num_compute_units()
    if not _fits_wvsplitkrc(n, k, m, cu_count):
        pytest.skip("shape exceeds wvSplitKrc launch limits")

    xavier = math.sqrt(2 / k) if xnorm else 1
    torch.manual_seed(0)
    a = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    b = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    bias = _make_bias(n, m, dtype, bias_mode)

    if padded_a:
        a = _pad_last_dim(a)

    out = ops.wvSplitKrc(a, b, cu_count, bias)
    ref = torch.nn.functional.linear(a, b, bias)

    tight_atol, max_atol, pass_rate = WVSPLITKRC_FEATURE_ERROR_BUDGETS[(dtype, xnorm)]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=(
            f"wvSplitKrc features dtype={dtype} xnorm={xnorm} "
            f"bias_mode={bias_mode} padded_a={padded_a}"
        ),
    )


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
@pytest.mark.parametrize("n,k,m", [(16, 2880, 128), (64, 3072, 256)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitkrc_determinism(n, k, m, dtype):
    cu_count = num_compute_units()
    if not _fits_wvsplitkrc(n, k, m, cu_count):
        pytest.skip("shape exceeds wvSplitKrc launch limits")

    a, b = _make_inputs(n, k, m, dtype, "normal")
    _assert_deterministic(lambda: ops.wvSplitKrc(a, b, cu_count, None))


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("n,k,m", [(32, 2880, 128), (128, 3072, 640)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitkrc_distributions(distribution, n, k, m, dtype):
    cu_count = num_compute_units()
    if not _fits_wvsplitkrc(n, k, m, cu_count):
        pytest.skip("shape exceeds wvSplitKrc launch limits")

    a, b = _make_inputs(n, k, m, dtype, distribution)
    out = ops.wvSplitKrc(a, b, cu_count, None)
    ref = torch.nn.functional.linear(a, b)

    tight_atol, max_atol, pass_rate = WVSPLITKRC_ERROR_BUDGETS[dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=(
            f"wvSplitKrc distribution shape=({n},{k},{m}) dtype={dtype} "
            f"distribution={distribution}"
        ),
    )


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="ROCm FP8 only")
@pytest.mark.parametrize("n,k,m", FP8_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fp8_accuracy(n, k, m, dtype):
    a, b, scale_a, scale_b = _make_inputs_fp8(n, k, m, "normal")
    out = ops.wvSplitKQ(
        b,
        a,
        dtype,
        scale_a,
        scale_b,
        num_compute_units(),
    )
    ref = torch._scaled_mm(
        a,
        b.t(),
        out_dtype=dtype,
        scale_a=scale_a,
        scale_b=scale_b,
    )

    tight_atol, max_atol, pass_rate = FP8_ERROR_BUDGETS[dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=f"FP8 shape=({n},{k},{m}) out_dtype={dtype}",
    )


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="ROCm FP8 only")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("padded_b", [False, True])
@pytest.mark.parametrize("biased", [False, True])
def test_fp8_features(dtype, padded_a, padded_b, biased):
    n, k, m = 4, 4096, 4096
    a, b, scale_a, scale_b = _make_inputs_fp8(n, k, m, "normal")

    if padded_a:
        a = _pad_last_dim(a)
    if padded_b:
        b = _pad_last_dim(b)

    bias = None if not biased else (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1)
    out = ops.wvSplitKQ(
        b,
        a,
        dtype,
        scale_a,
        scale_b,
        num_compute_units(),
        bias,
    )
    ref = torch._scaled_mm(
        a,
        b.t(),
        out_dtype=dtype,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
    )
    tight_atol, max_atol, pass_rate = FP8_ERROR_BUDGETS[dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=(
            f"FP8 features out_dtype={dtype} padded_a={padded_a} "
            f"padded_b={padded_b} biased={biased}"
        ),
    )


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="ROCm FP8 only")
@pytest.mark.parametrize("n,k,m", [(3, 512, 528), (4, 4096, 4096), (2, 24576, 2048)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", ["normal", "sparse_activations"])
def test_fp8_distributions(n, k, m, dtype, distribution):
    a, b, scale_a, scale_b = _make_inputs_fp8(n, k, m, distribution)
    out = ops.wvSplitKQ(
        b,
        a,
        dtype,
        scale_a,
        scale_b,
        num_compute_units(),
    )
    ref = torch._scaled_mm(
        a,
        b.t(),
        out_dtype=dtype,
        scale_a=scale_a,
        scale_b=scale_b,
    )
    tight_atol, max_atol, pass_rate = FP8_ERROR_BUDGETS[dtype]
    _assert_with_error_budget(
        out,
        ref,
        tight_atol=tight_atol,
        max_atol=max_atol,
        pass_rate=pass_rate,
        label=(
            f"FP8 distribution shape=({n},{k},{m}) out_dtype={dtype} "
            f"distribution={distribution}"
        ),
    )


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="ROCm FP8 only")
@pytest.mark.parametrize("n,k,m", [(4, 4096, 4096), (2, 24576, 2048)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fp8_determinism(n, k, m, dtype):
    a, b, scale_a, scale_b = _make_inputs_fp8(n, k, m, "normal")
    _assert_deterministic(
        lambda: ops.wvSplitKQ(b, a, dtype, scale_a, scale_b, num_compute_units())
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitk_nan_propagation(dtype):
    n, k, m = 2, 256, 256
    torch.manual_seed(0)
    a = torch.rand(n, k, dtype=dtype, device="cuda") - 0.5
    b = torch.rand(m, k, dtype=dtype, device="cuda") - 0.5
    a[0, 0] = float("nan")

    out = ops.wvSplitK(b, a.view(-1, a.size(-1)), num_compute_units())
    assert out[0].isnan().any()
    assert not out[1].isnan().any()


@pytest.mark.parametrize("dtype", DTYPES)
def test_llmm1_nan_propagation(dtype):
    n, k, m = 1, 256, 256
    torch.manual_seed(0)
    a = torch.rand(n, k, dtype=dtype, device="cuda")
    b = torch.rand(m, k, dtype=dtype, device="cuda")
    a[0, 0] = float("nan")

    out = ops.LLMM1(b, a, 4)
    assert out[0].isnan().any()


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitkrc_nan_propagation(dtype):
    cu_count = num_compute_units()
    n, k, m = 16, 2880, 128
    if not _fits_wvsplitkrc(n, k, m, cu_count):
        pytest.skip("shape exceeds wvSplitKrc launch limits")

    torch.manual_seed(0)
    a = torch.rand(n, k, dtype=dtype, device="cuda") - 0.5
    b = torch.rand(m, k, dtype=dtype, device="cuda") - 0.5
    a[0, 0] = float("nan")

    out = ops.wvSplitKrc(a, b, cu_count, None)
    assert out[0].isnan().any()
    assert not out[1].isnan().any()


@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitk_zero_and_bias(dtype):
    n, k, m = 2, 256, 256
    a = torch.zeros(n, k, dtype=dtype, device="cuda")
    b = torch.zeros(m, k, dtype=dtype, device="cuda")

    out = ops.wvSplitK(b, a.view(-1, a.size(-1)), num_compute_units())
    assert torch.all(out == 0)

    bias = torch.rand(m, dtype=dtype, device="cuda") - 0.5
    out = ops.wvSplitK(b, a.view(-1, a.size(-1)), num_compute_units(), bias)
    torch.testing.assert_close(out, bias.unsqueeze(0).expand(n, -1), atol=0, rtol=0)


@pytest.mark.parametrize("n,k,m", [(1, 2048, 4096), (1, 4096, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_llmm1_logprobs(n, k, m, dtype):
    a, b = _make_inputs(n, k, m, dtype, "normal")
    _assert_logprobs_close(ops.LLMM1(b, a, 4), torch.matmul(a, b.t()))


@pytest.mark.parametrize(
    "n,k,m",
    [(1, 4096, 32000), (4, 4096, 32000), (1, 4096, 151936)],
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_wvsplitk_logprobs(n, k, m, dtype, distribution):
    a, b = _make_inputs(n, k, m, dtype, distribution)
    out = ops.wvSplitK(b, a.view(-1, a.size(-1)), num_compute_units())
    ref = torch.nn.functional.linear(a, b)
    _assert_logprobs_close(out, ref)


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
@pytest.mark.parametrize("n,k,m", [(16, 2880, 128), (128, 2880, 640)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_wvsplitkrc_logprobs(n, k, m, dtype, distribution):
    cu_count = num_compute_units()
    if not _fits_wvsplitkrc(n, k, m, cu_count):
        pytest.skip("shape exceeds wvSplitKrc launch limits")

    a, b = _make_inputs(n, k, m, dtype, distribution)
    out = ops.wvSplitKrc(a, b, cu_count, None)
    ref = torch.nn.functional.linear(a, b)
    _assert_logprobs_close(out, ref)


@pytest.mark.skipif(not current_platform.supports_fp8(), reason="ROCm FP8 only")
@pytest.mark.parametrize("n,k,m", [(4, 4096, 4096), (2, 24576, 2048)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fp8_logprobs(n, k, m, dtype):
    a, b, scale_a, scale_b = _make_inputs_fp8(n, k, m, "normal")
    out = ops.wvSplitKQ(b, a, dtype, scale_a, scale_b, num_compute_units())
    ref = torch._scaled_mm(
        a,
        b.t(),
        out_dtype=dtype,
        scale_a=scale_a,
        scale_b=scale_b,
    )
    _assert_logprobs_close(out, ref, max_top1_logprob_diff=0.02)


def _assert_e2e_outputs_match(reference, current, *, atol: float) -> None:
    for prompt_idx, (ref_output, cur_output) in enumerate(zip(reference, current)):
        ref_token_ids, ref_text, ref_logprobs = ref_output
        cur_token_ids, cur_text, cur_logprobs = cur_output

        assert ref_token_ids == cur_token_ids, (
            f"token ids diverged for prompt {prompt_idx}: "
            f"{ref_token_ids[:10]} != {cur_token_ids[:10]}"
        )
        assert ref_text == cur_text, (
            f"text diverged for prompt {prompt_idx}: {ref_text!r} != {cur_text!r}"
        )
        assert len(ref_logprobs) == len(cur_logprobs), (
            f"logprob length diverged for prompt {prompt_idx}"
        )

        for step_idx, (ref_step, cur_step) in enumerate(
            zip(ref_logprobs, cur_logprobs)
        ):
            assert set(ref_step.keys()) == set(cur_step.keys()), (
                f"logprob token set diverged for prompt {prompt_idx}, step {step_idx}"
            )
            for token_id in ref_step:
                diff = abs(ref_step[token_id].logprob - cur_step[token_id].logprob)
                assert diff <= atol, (
                    f"logprob drift {diff:.6e} > {atol:.6e} for "
                    f"prompt {prompt_idx}, step {step_idx}, token {token_id}"
                )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_e2e_logprob_reproducibility(enforce_eager, monkeypatch, vllm_runner):
    monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "1")

    model = "TitanML/tiny-mixtral"
    prompts = [
        "The capital of France is",
        "In quantum computing, a qubit",
        "def fibonacci(n):\n",
    ]
    max_tokens = 32
    num_logprobs = 5
    atol = 0.0 if enforce_eager else 1e-6

    runs = []
    for _ in range(2):
        with vllm_runner(
            model,
            dtype="half",
            enforce_eager=enforce_eager,
            max_num_seqs=1,
            seed=0,
            enable_prefix_caching=False,
        ) as llm:
            runs.append(llm.generate_greedy_logprobs(prompts, max_tokens, num_logprobs))

    _assert_e2e_outputs_match(runs[0], runs[1], atol=atol)
