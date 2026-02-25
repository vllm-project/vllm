# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.utils.platform_utils import get_cu_count

DTYPES = [torch.bfloat16, torch.float16]

TOLERANCE_ULP = 1  # ULPs for fp32-accumulation kernels
LLMM1_TOLERANCE_ULP = 5  # 5x wider for dtype-precision accumulation

# LLMM1: N must be 1, M must be divisible by rows_per_block (4, 8).
# K values span small (128) to large (6144), all multiples of 8.
# M values are all divisible by lcm(4,8)=8.
NKM_LLMM1 = [
    (1, 128, 256),
    (1, 512, 1024),
    (1, 2048, 4096),
    (1, 6144, 1024),  # wide K, moderate M
    (1, 4096, 8192),  # large M
]

# wvSplitK: N in {1..4}, K % 8 == 0.
# Includes small shapes (64x64) through large (16384x8192),
# plus tiny-M edge cases (M=8) that stress YTILE/commitColumn logic.
NKM_WVSPLITK = [
    (1, 64, 64),
    (2, 256, 256),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (1, 9216, 512),  # wide K, narrow M
    (4, 16384, 8192),  # max batch x large dimensions
    (1, 64, 8),  # tiny M, single batch
    (4, 256, 8),  # tiny M, max batch
]

# wvSplitKrc: N values that round to valid N_p2 in {16,32,64,128}.
# K values include non-power-of-2 (2880) and aligned (3072),
# plus +8 offsets to test non-aligned K.
# M values include aligned and +16 offsets for boundary testing.
N_RC = [13, 16, 32, 64, 103, 128]
K_RC = [2880, 2880 + 8, 3072, 3072 + 8]
M_RC = [128, 128 + 16, 256, 256 + 16, 640, 640 + 16]

# FP8: N in {1..4}, K % 16 == 0.
# Spans from small (64) to very large (65552) K.
# +16 offsets test non-aligned-but-legal shapes.
NKM_FP8 = [
    (1, 64, 64),
    (1, 64 + 16, 64 + 16),
    (4, 64, 64 + 16),
    (3, 512, 512 + 16),
    (4, 2048 + 16, 2048 + 16),
    (4, 4096, 4096),
    (4, 16400, 2048 + 16),
    (1, 14336, 1024),
    (2, 24576, 2048),
    (4, 32768, 28672),
    (4, 32768 * 2 + 16, 28672 + 16),  # K=65552, max stress
]

# Model-realistic shapes matching real LLM architectures.
# (1, 4096, 32000) = Llama-7B hidden -> vocab single-token decode
# (4, 4096, 32000) = same with 4-token batch
# (1, 8192, 28672) = Llama-70B FFN intermediate
# (1, 4096, 14336) = Llama-7B FFN gate projection
# (1, 4096, 151936) = Qwen-7B hidden -> vocab (large vocabulary)
NKM_MODEL_REALISTIC = [
    (1, 4096, 32000),
    (4, 4096, 32000),
    (1, 8192, 28672),
    (1, 4096, 14336),
    (1, 4096, 151936),
]

DISTRIBUTIONS = ["normal", "mixed_scale", "sparse_activations"]


def pad_fp8(weight):
    """Pad FP8 tensor to 256-byte alignment, then remove padding.

    This simulates the memory layout after cudaMalloc alignment,
    testing that kernels correctly handle stride != size(1).
    """
    num_pad = 256 // weight.element_size()
    import torch.nn.functional as F

    return F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]


def _fits_rc(n, k, m, cu_count):
    """Check if wvSplitKrc shape fits within available CUs.

    The kernel assigns ceil(M/64) x ceil(K/512) workgroups, multiplied by
    GrpsShrB (number of waves sharing B-tile loads within a group).
    If this exceeds CuCount, the kernel cannot launch.
    """
    N_p2 = 1 << (n - 1).bit_length()
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    GrpsShrB = min(N_p2 // 16, 4)
    return rndup_cus * GrpsShrB <= cu_count


def _mixed_scale_multiplier(dtype):
    """Maximum safe dynamic-range multiplier for mixed_scale distribution.

    With fp32 accumulation, a partial sum S and a small addend x are
    correctly accumulated as long as |x| > |S| * epsilon_fp32, which is
    essentially always true.  But we also need the INPUTS themselves
    to be representable in dtype, and the OUTPUTS to not overflow.

    We limit the hot/cold ratio to 0.25/epsilon_dtype so that:
    1. Cold-channel contributions retain >= 2 significant bits when
       added to hot-channel partial sums in fp32 and then rounded
       to dtype output.
    2. The dynamic range stays within dtype representable range.

    Results:  bf16 -> ~32x,  fp16 -> 50x (capped for small-K stability).
    """
    eps = torch.finfo(dtype).eps
    return min(50.0, 0.25 / eps)


def _make_inputs(n, k, m, dtype, distribution, seed=0):
    """Generate test matrices with controlled distributions.

    Returns A (NxK) and B (MxK) in the specified dtype on CUDA.
    """
    torch.manual_seed(seed)
    xavier = math.sqrt(2 / k)

    if distribution == "normal":
        A = torch.randn(n, k, dtype=dtype, device="cuda") * xavier
        B = torch.randn(m, k, dtype=dtype, device="cuda") * xavier

    elif distribution == "mixed_scale":
        A = torch.randn(n, k, dtype=dtype, device="cuda") * xavier
        B = torch.randn(m, k, dtype=dtype, device="cuda") * xavier
        hot = torch.randperm(k, device="cuda")[: max(1, k // 10)]
        B[:, hot] *= _mixed_scale_multiplier(dtype)

    elif distribution == "sparse_activations":
        A = torch.randn(n, k, dtype=dtype, device="cuda").clamp(min=0) * xavier
        B = torch.randn(m, k, dtype=dtype, device="cuda") * xavier

    elif distribution == "uniform":
        A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
        B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return A, B


def _make_inputs_fp8(n, k, m, distribution, seed=0):
    """Generate FP8-quantized inputs.

    Starts from fp32 to avoid double-quantization artifacts (generating
    in bf16 then quantizing to fp8 would compound rounding errors).
    Returns (A_fp8, B_fp8, scale_a, scale_b).
    """
    A, B = _make_inputs(n, k, m, torch.float32, distribution, seed)
    A_fp8, sa = ref_dynamic_per_tensor_fp8_quant(A)
    B_fp8, sb = ref_dynamic_per_tensor_fp8_quant(B)
    return A_fp8, B_fp8, sa, sb


def _make_bias(n, m, dtype, bias_mode, seed=42):
    """Generate bias tensor.

    bias_mode=0: None (no bias)
    bias_mode=1: 1-D bias (M,) - typical layer bias
    bias_mode=2: 2-D bias (N,M) - per-batch bias (rare but supported)
    """
    torch.manual_seed(seed)
    if bias_mode == 0:
        return None
    elif bias_mode == 1:
        return torch.rand(m, dtype=dtype, device="cuda") * 2 - 1
    elif bias_mode == 2:
        return torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1
    raise ValueError(f"Unknown bias_mode: {bias_mode}")


def _gemm_tol(dtype):
    """Tolerance for fp32-accumulation kernels (wvSplitK, wvSplitKrc).

    Both kernel and reference produce:
      result = round_to_dtype(fp32_accumulated_sum)

    We use 1*epsilon (1 ULP), the theoretical minimum for fp32
    accumulation where kernel and reference follow the same
    reduction order.

    Resulting tolerances:
      bf16: rtol = atol ~= 7.81e-3  (practical: <= 0.8% relative error)
      fp16: rtol = atol ~= 9.77e-4  (practical: <= 0.1% relative error)
    """
    eps = torch.finfo(dtype).eps
    tol = TOLERANCE_ULP * eps
    return tol, tol


def _llmm1_tol(dtype):
    """Tolerance for LLMM1 (dtype-precision accumulation).

    LLMM1 accumulates 8 products per thread in native dtype via
    __hmul2/__hfma2, introducing O(4 * epsilon_dtype) rounding per group.
    This compounds with output rounding to produce errors significantly
    larger than the fp32-accumulation baseline.

    The 5x multiplier (5 ULP total) covers worst-case dtype-precision
    accumulation while remaining tight enough to catch real precision
    regressions.

    Resulting tolerances:
      bf16: rtol = atol ~= 3.91e-2  (practical: up to 3.9% relative)
      fp16: rtol = atol ~= 4.88e-3  (practical: up to 0.5% relative)
    """
    eps = torch.finfo(dtype).eps
    tol = LLMM1_TOLERANCE_ULP * eps
    return tol, tol


def _fp8_gemm_tol(dtype, k):
    """Tolerance for FP8 GEMM (wvSplitKQ).

    Two error sources combine:

    1. Quantization noise: each fp8 input has relative error <= epsilon_fp8/2.
       Over K products, the noise in the dot product grows as O(sqrt(K))
       under a random-noise model (central limit theorem on K
       independent quantization errors).

       Empirical coefficient: 0.005 * sqrt(K) * epsilon_fp8, capped at 0.05
       to prevent false passes on very large K.

    2. Output rounding: same as fp32-accumulation kernels, 1 * epsilon_dtype.

    We take the maximum of both components for atol, with rtol scaled
    down by 10x (since relative tolerance only needs to handle the
    multiplicative component of quantization noise, not the additive floor).
    """
    eps = torch.finfo(dtype).eps
    fp8_eps = 0.0625  # epsilon for fp8-e4m3
    quant_component = min(math.sqrt(k) * fp8_eps * 0.005, 0.05)
    output_component = TOLERANCE_ULP * eps
    atol = max(quant_component, output_component)
    rtol = max(quant_component / 10, TOLERANCE_ULP * eps)
    return atol, rtol


def _assert_accurate(out, ref, dtype, label="", atol_override=None, rtol_override=None):
    """Assert GEMM accuracy with principled, dtype-derived tolerances.

    Per-element check: pass iff |diff| <= max(atol, rtol * |ref|).
    The dual criterion ensures:
      - Near-zero outputs judged by absolute error (avoids division-by-zero
        in relative checks)
      - Large outputs judged by relative error (scales with magnitude)

    Three-pronged failure criteria (ALL must hold for a pass):
      1. >=99.999% of elements pass per-element tolerance
         (allows only extremely rare rounding edge cases)
      2. No element exceeds 3x its tolerance
         (catches catastrophic single-element bugs like wrong index)
      3. Mean absolute error < atol * 0.25
         (catches systematic bias early, before it reaches full tolerance)
    """
    atol, rtol = _gemm_tol(dtype)
    if atol_override is not None:
        atol = atol_override
    if rtol_override is not None:
        rtol = rtol_override

    diff = (out.float() - ref.float()).abs()
    ref_abs = ref.float().abs()

    per_elem_tol = torch.maximum(
        torch.full_like(diff, atol),
        rtol * ref_abs,
    )

    element_ok = diff <= per_elem_tol
    pass_rate = element_ok.float().mean().item()
    max_violation_ratio = (diff / per_elem_tol.clamp(min=1e-30)).max().item()
    mean_abs = diff.mean().item()

    passed = (
        pass_rate >= 0.99999 and max_violation_ratio <= 3.0 and mean_abs < atol * 0.25
    )

    if not passed:
        w = diff.argmax().item()
        nc = out.shape[-1] if out.dim() > 1 else out.shape[0]
        reasons = []
        if pass_rate < 0.99999:
            reasons.append(f"pass_rate={pass_rate:.7f}<0.99999")
        if max_violation_ratio > 3.0:
            reasons.append(f"max_violation={max_violation_ratio:.1f}x>3x")
        if mean_abs >= atol * 0.25:
            reasons.append(f"mean_abs={mean_abs:.6e}>={atol * 0.25:.6e}")

        raise AssertionError(
            f"Accuracy FAILED ({label})\n"
            f"  tolerances: atol={atol:.6e}, rtol={rtol:.6e}\n"
            f"  failures: {'; '.join(reasons)}\n"
            f"  max_abs={diff.max().item():.6e}, "
            f"mean_abs={mean_abs:.6e}, "
            f"pass_rate={pass_rate:.7f}\n"
            f"  worst [{w // nc},{w % nc}]: "
            f"got={out.flatten()[w].item():.6f} "
            f"ref={ref.flatten()[w].item():.6f} "
            f"tol={per_elem_tol.flatten()[w].item():.6f}\n"
            f"  shape={list(out.shape)}, dtype={dtype}"
        )


def _assert_deterministic(fn, num_runs=10, label=""):
    """Assert bitwise-identical output across repeated runs.

    Non-determinism in GEMM kernels propagates through softmax and
    corrupts logprobs, causing token-selection instability during
    generation.  The deterministic kernel path must be bitwise exact
    because downstream consumers (e.g., speculative decoding
    verification) rely on reproducible logits.
    """
    results = [fn() for _ in range(num_runs)]
    for i in range(1, num_runs):
        if not torch.equal(results[0], results[i]):
            d = (results[0].float() - results[i].float()).abs()
            ndiff = (d > 0).sum().item()
            raise AssertionError(
                f"Non-determinism ({label}): run 0 vs {i}, "
                f"{ndiff}/{d.numel()} elements differ, "
                f"max_diff={d.max().item():.6e}"
            )


def _set_rows(idx_tensor, n):
    return [set(idx_tensor[r].tolist()) for r in range(n)]


def _assert_logprobs(out_logits, ref_logits, label="", top1_lp_max_diff=0.01):
    """Assert logprobs derived from kernel output match reference.

    Checks three properties that directly impact generation quality:

    1. Top-1 token agreement (>= 99%):
       Determines greedy decoding output. A top-1 mismatch means the
       kernel would produce different text than the reference.
       We allow 1% disagreement because near-tied logits (where two
       tokens have nearly equal probability) are inherently sensitive
       to rounding - a 0.001 logit difference can flip the argmax.

    2. Top-5 set overlap (>= 95%):
       Determines sampling diversity. Even when top-1 agrees, if the
       top-5 set is different, sampling-based generation explores
       different token spaces.

    3. Top-1 logprob accuracy (<= 0.01 nats):
       Determines confidence calibration. A 0.01 nat error in the
       top-1 logprob translates to ~1% multiplicative probability
       error, acceptable for most applications.
    """
    n, m = ref_logits.shape
    ref_lp = torch.nn.functional.log_softmax(ref_logits.float(), dim=-1)
    out_lp = torch.nn.functional.log_softmax(out_logits.float(), dim=-1)

    ref_t1 = ref_lp.argmax(-1)
    out_t1 = out_lp.argmax(-1)
    t1_rate = (ref_t1 == out_t1).float().mean().item()

    k5 = min(5, m)
    ref_t5 = _set_rows(ref_lp.topk(k5, -1).indices, n)
    out_t5 = _set_rows(out_lp.topk(k5, -1).indices, n)
    t5_rate = sum(len(a & b) / k5 for a, b in zip(ref_t5, out_t5)) / n

    lp_diff = (
        (ref_lp.gather(1, ref_t1.unsqueeze(1)) - out_lp.gather(1, ref_t1.unsqueeze(1)))
        .abs()
        .max()
        .item()
    )

    failures = []
    if t1_rate < 0.99:
        failures.append(f"top1_match={t1_rate:.4f}<0.99")
    if t5_rate < 0.95:
        failures.append(f"top5_overlap={t5_rate:.4f}<0.95")
    if lp_diff > top1_lp_max_diff:
        failures.append(f"top1_lp_diff={lp_diff:.6e}>{top1_lp_max_diff}")
    if failures:
        raise AssertionError(f"Logprobs FAILED ({label}): {'; '.join(failures)}")


def _assert_logprobs_deterministic(fn, num_runs=10, label=""):
    """Assert logprobs are bitwise identical across runs.

    Even if raw logits have sub-ULP non-deterministic differences,
    softmax amplifies them near decision boundaries - a 1e-7 logit
    jitter between two near-tied tokens can flip the argmax and
    change generated text.
    """
    lps = [
        torch.nn.functional.log_softmax(fn().float(), dim=-1) for _ in range(num_runs)
    ]
    for i in range(1, num_runs):
        if not torch.equal(lps[0], lps[i]):
            d = (lps[0] - lps[i]).abs()
            flips = (lps[0].argmax(-1) != lps[i].argmax(-1)).sum().item()
            raise AssertionError(
                f"Logprobs non-determinism ({label}): run 0 vs {i}, "
                f"{(d > 0).sum().item()} diffs, "
                f"max={d.max().item():.6e}, "
                f"top1_flips={flips}/{lps[0].shape[0]}"
            )


# Shape constraints enforced:
#   - N=1 (all tuples have N=1)
#   - M divisible by rows_per_block (all M in NKM_LLMM1 are divisible by 8)
#   - K multiple of 8 (all K in NKM_LLMM1 satisfy this)
@pytest.mark.parametrize("n,k,m", NKM_LLMM1)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows_per_block", [4, 8])
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@torch.inference_mode()
def test_llmm1_accuracy(n, k, m, dtype, rows_per_block, distribution):
    assert n == 1, f"LLMM1 requires N=1, got {n}"
    assert m % rows_per_block == 0, (
        f"M={m} not divisible by rows_per_block={rows_per_block}"
    )

    A, B = _make_inputs(n, k, m, dtype, distribution)
    out = ops.LLMM1(B, A, rows_per_block)
    ref = torch.matmul(A, B.t())

    atol, rtol = _llmm1_tol(dtype)
    _assert_accurate(
        out,
        ref,
        dtype,
        atol_override=atol,
        rtol_override=rtol,
        label=f"LLMM1 {n}x{k}x{m} {dtype} rpb={rows_per_block} {distribution}",
    )


@pytest.mark.parametrize("n,k,m", [(1, 2048, 4096), (1, 4096, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@torch.inference_mode()
def test_llmm1_determinism(n, k, m, dtype):
    A, B = _make_inputs(n, k, m, dtype, "normal")
    _assert_deterministic(
        lambda: ops.LLMM1(B, A, 4), label=f"LLMM1 {n}x{k}x{m} {dtype}"
    )


@pytest.mark.parametrize("n,k,m", [(1, 2048, 4096), (1, 4096, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@torch.inference_mode()
def test_llmm1_logprobs(n, k, m, dtype):
    A, B = _make_inputs(n, k, m, dtype, "normal")
    _assert_logprobs(
        ops.LLMM1(B, A, 4),
        torch.matmul(A, B.t()),
        label=f"LLMM1 lp {n}x{k}x{m} {dtype}",
    )


# Shape constraints enforced:
#   - N in {1,2,3,4} (all tuples satisfy this)
#   - K % 8 == 0 (all K values satisfy this)
@pytest.mark.parametrize("n,k,m", NKM_WVSPLITK)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("bias_mode", [0, 1, 2])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
def test_wvsplitk_accuracy(n, k, m, dtype, distribution, bias_mode):
    assert 1 <= n <= 4, f"wvSplitK requires N in {{1..4}}, got {n}"
    assert k % 8 == 0, f"wvSplitK requires K%8==0, got K={k}"

    A, B = _make_inputs(n, k, m, dtype, distribution)
    BIAS = _make_bias(n, m, dtype, bias_mode)
    cu = get_cu_count()
    _assert_accurate(
        ops.wvSplitK(B, A.view(-1, A.size(-1)), cu, BIAS),
        torch.nn.functional.linear(A, B, BIAS),
        dtype,
        label=f"wvSplitK {n}x{k}x{m} {dtype} {distribution} b={bias_mode}",
    )


@pytest.mark.parametrize("n,k,m", NKM_WVSPLITK)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
def test_wvsplitk_determinism(n, k, m, dtype):
    A, B = _make_inputs(n, k, m, dtype, "normal")
    cu = get_cu_count()
    _assert_deterministic(
        lambda: ops.wvSplitK(B, A.view(-1, A.size(-1)), cu),
        label=f"wvSplitK {n}x{k}x{m} {dtype}",
    )


@pytest.mark.parametrize("n,k,m", NKM_MODEL_REALISTIC)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
def test_wvsplitk_model_realistic(n, k, m, dtype, distribution):
    """Test with shapes from real LLM architectures.

    These shapes exercise the 'big' kernel path (A >> LDS capacity)
    and stress the persistent-WG iteration over large M dimensions.
    The vocabulary-sized M (32000, 151936) ensures the commitColumn
    fragmentation logic at the tail is exercised.
    """
    assert 1 <= n <= 4
    assert k % 8 == 0

    A, B = _make_inputs(n, k, m, dtype, distribution)
    cu = get_cu_count()
    _assert_accurate(
        ops.wvSplitK(B, A.view(-1, A.size(-1)), cu),
        torch.nn.functional.linear(A, B),
        dtype,
        label=f"wvSplitK realistic {n}x{k}x{m} {dtype} {distribution}",
    )


# Shape constraints enforced:
#   - gfx950 only (skipif)
#   - N -> N_p2 in {16, 32, 64, 128} (all N_RC values round to valid N_p2)
#   - K % 8 == 0 (all K_RC values satisfy this)
#   - CU fit checked via _fits_rc at runtime
# Full dimension sweep - normal distribution, both kernel paths
@pytest.mark.parametrize("fast_skinny_gemm", [False, True])
@pytest.mark.parametrize("n", N_RC)
@pytest.mark.parametrize("k", K_RC)
@pytest.mark.parametrize("m", M_RC)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
def test_wvsplitkrc_accuracy(fast_skinny_gemm, n, k, m, dtype):
    cu = get_cu_count()
    if not _fits_rc(n, k, m, cu):
        pytest.skip("Shape exceeds CU capacity for wvSplitKrc")
    A, B = _make_inputs(n, k, m, dtype, "normal")
    _assert_accurate(
        ops.wvSplitKrc(A, B, cu, None, fast_skinny_gemm=fast_skinny_gemm),
        torch.nn.functional.linear(A, B),
        dtype,
        label=f"wvSplitKrc fast={fast_skinny_gemm} {n}x{k}x{m} {dtype}",
    )


# Feature coverage: xnorm, padding, bias
@pytest.mark.parametrize("xnorm", [False, True])
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("bias_mode", [0, 1, 2])
@pytest.mark.parametrize("n", [13, 64, 128])
@pytest.mark.parametrize("k", [2880, 3072])
@pytest.mark.parametrize("m", [128, 640])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
def test_wvsplitkrc_features(xnorm, padded_a, bias_mode, n, k, m, dtype):
    """Test feature interactions: xnorm scaling, stride padding, bias modes.

    xnorm=True uses xavier scaling sqrt(2/K), keeping values small enough
    that a tighter absolute tolerance (1e-3) is appropriate - this
    verifies that the kernel doesn't introduce artifacts at small magnitudes.
    """
    cu = get_cu_count()
    if not _fits_rc(n, k, m, cu):
        pytest.skip("Shape exceeds CU capacity")
    xavier = math.sqrt(2 / k) if xnorm else 1
    torch.manual_seed(0)
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    if padded_a:
        A = pad_fp8(A)
    BIAS = _make_bias(n, m, dtype, bias_mode)
    ref = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitKrc(A, B, cu, BIAS)
    if xnorm:
        # Xavier-scaled values are O(sqrt(2/K)) ~= 0.02-0.04 for typical K.
        # Output magnitudes are O(1/sqrt(K)), making 1e-3 a ~1-3 ULP check.
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-8)
    else:
        _assert_accurate(
            out,
            ref,
            dtype,
            label=f"wvSplitKrc xnorm={xnorm} pad={padded_a} "
            f"bias={bias_mode} {n}x{k}x{m} {dtype}",
        )


# Distribution stress on representative subset
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("fast_skinny_gemm", [False, True])
@pytest.mark.parametrize("n", [32, 128])
@pytest.mark.parametrize("k", [2880, 3072])
@pytest.mark.parametrize("m", [128, 640])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
def test_wvsplitkrc_distributions(distribution, fast_skinny_gemm, n, k, m, dtype):
    cu = get_cu_count()
    if not _fits_rc(n, k, m, cu):
        pytest.skip("Shape exceeds CU capacity")
    A, B = _make_inputs(n, k, m, dtype, distribution)
    _assert_accurate(
        ops.wvSplitKrc(A, B, cu, None, fast_skinny_gemm=fast_skinny_gemm),
        torch.nn.functional.linear(A, B),
        dtype,
        label=f"wvSplitKrc {distribution} fast={fast_skinny_gemm} {n}x{k}x{m} {dtype}",
    )


# Determinism - deterministic path only (fast_skinny_gemm=False)
# The fast path uses atomicAdd which is non-deterministic by design.
@pytest.mark.parametrize("n", N_RC)
@pytest.mark.parametrize("k", [2880, 3072])
@pytest.mark.parametrize("m", [128, 256, 640])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
def test_wvsplitkrc_determinism(n, k, m, dtype):
    cu = get_cu_count()
    if not _fits_rc(n, k, m, cu):
        pytest.skip("Shape exceeds CU capacity")
    A, B = _make_inputs(n, k, m, dtype, "normal")
    _assert_deterministic(
        lambda: ops.wvSplitKrc(A, B, cu, None, fast_skinny_gemm=False),
        label=f"wvSplitKrc {n}x{k}x{m} {dtype}",
    )


# Shape constraints enforced:
#   - MI3XX + fp8 support (skipif)
#   - N in {1..4} (all NKM_FP8 tuples satisfy this)
#   - K % 16 == 0 (all K values in NKM_FP8 satisfy this)
@pytest.mark.parametrize("n,k,m", NKM_FP8)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="rocm fp8 only",
)
def test_fp8_accuracy(n, k, m, dtype, padded_a, biased):
    assert 1 <= n <= 4, f"wvSplitKQ requires N in {{1..4}}, got {n}"
    assert k % 16 == 0, f"wvSplitKQ requires K%16==0, got K={k}"

    A, B, sa, sb = _make_inputs_fp8(n, k, m, "normal")
    if padded_a:
        A = pad_fp8(A)
    BIAS = None if not biased else (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1)
    ref = torch._scaled_mm(A, B.t(), out_dtype=dtype, scale_a=sa, scale_b=sb, bias=BIAS)
    out = ops.wvSplitKQ(B, A, dtype, sa, sb, get_cu_count(), BIAS)
    atol, rtol = _fp8_gemm_tol(dtype, k)
    _assert_accurate(
        out,
        ref,
        dtype,
        atol_override=atol,
        rtol_override=rtol,
        label=f"FP8 {n}x{k}x{m} {dtype} pad={padded_a} bias={biased}",
    )


@pytest.mark.parametrize(
    "n,k,m", [(3, 512, 512 + 16), (4, 4096, 4096), (2, 24576, 2048), (4, 32768, 28672)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", ["normal", "sparse_activations"])
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="rocm fp8 only",
)
def test_fp8_distributions(n, k, m, dtype, distribution):
    """Test FP8 with non-normal distributions.

    mixed_scale is excluded because fp8 quantization with per-tensor
    scaling cannot resolve the dynamic range - hot channels saturate
    fp8 range while cold channels quantize to zero, making accuracy
    comparison meaningless. sparse_activations is meaningful because
    the zero structure is preserved through quantization.
    """
    A, B, sa, sb = _make_inputs_fp8(n, k, m, distribution)
    ref = torch._scaled_mm(A, B.t(), out_dtype=dtype, scale_a=sa, scale_b=sb)
    out = ops.wvSplitKQ(B, A, dtype, sa, sb, get_cu_count())
    atol, rtol = _fp8_gemm_tol(dtype, k)
    _assert_accurate(
        out,
        ref,
        dtype,
        atol_override=atol,
        rtol_override=rtol,
        label=f"FP8 dist {n}x{k}x{m} {dtype} {distribution}",
    )


@pytest.mark.parametrize(
    "n,k,m", [(4, 4096, 4096), (2, 24576, 2048), (4, 32768, 28672)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="rocm fp8 only",
)
def test_fp8_determinism(n, k, m, dtype):
    A, B, sa, sb = _make_inputs_fp8(n, k, m, "normal")
    _assert_deterministic(
        lambda: ops.wvSplitKQ(B, A, dtype, sa, sb, get_cu_count()),
        label=f"FP8 {n}x{k}x{m} {dtype}",
    )


@pytest.mark.parametrize(
    "n,k,m", [(4, 4096, 4096), (4, 16400, 2048 + 16), (1, 14336, 1024)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="rocm fp8 only",
)
def test_fp8_padded_b(n, k, m, dtype):
    """Test FP8 with padded B tensor (stride(0) != size(1)).

    The wvSplitKQ kernel reads B using Kbp = in_b.stride(0), which
    differs from K when the tensor has alignment padding.  This test
    verifies the kernel uses the stride, not the logical size.
    """
    A, B, sa, sb = _make_inputs_fp8(n, k, m, "normal")
    B = pad_fp8(B)
    ref = torch._scaled_mm(A, B.t(), out_dtype=dtype, scale_a=sa, scale_b=sb)
    out = ops.wvSplitKQ(B, A, dtype, sa, sb, get_cu_count())
    atol, rtol = _fp8_gemm_tol(dtype, k)
    _assert_accurate(
        out,
        ref,
        dtype,
        atol_override=atol,
        rtol_override=rtol,
        label=f"FP8 padB {n}x{k}x{m} {dtype}",
    )


# These tests verify that GEMM accuracy is sufficient for the downstream
# task that matters most: correct token selection in autoregressive
# generation. A GEMM kernel can pass element-wise tolerance checks
# while still producing wrong tokens if errors cluster near decision
# boundaries in the softmax output.
@pytest.mark.parametrize(
    "n,k,m", [(1, 4096, 32000), (4, 4096, 32000), (1, 8192, 28672), (1, 4096, 151936)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
def test_wvsplitk_logprobs(n, k, m, dtype, distribution):
    A, B = _make_inputs(n, k, m, dtype, distribution)
    cu = get_cu_count()
    _assert_logprobs(
        ops.wvSplitK(B, A.view(-1, A.size(-1)), cu),
        torch.nn.functional.linear(A, B),
        label=f"wvSplitK lp {n}x{k}x{m} {dtype} {distribution}",
    )


@pytest.mark.parametrize(
    "n,k,m", [(1, 4096, 32000), (4, 4096, 32000), (1, 4096, 151936)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
def test_wvsplitk_logprobs_determinism(n, k, m, dtype):
    A, B = _make_inputs(n, k, m, dtype, "normal")
    cu = get_cu_count()
    _assert_logprobs_deterministic(
        lambda: ops.wvSplitK(B, A.view(-1, A.size(-1)), cu),
        label=f"wvSplitK lp det {n}x{k}x{m} {dtype}",
    )


@pytest.mark.parametrize("fast_skinny_gemm", [False, True])
@pytest.mark.parametrize("n,k,m", [(16, 2880, 128), (64, 3072, 256), (128, 2880, 640)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
def test_wvsplitkrc_logprobs(fast_skinny_gemm, n, k, m, dtype, distribution):
    cu = get_cu_count()
    if not _fits_rc(n, k, m, cu):
        pytest.skip("Shape exceeds CU capacity")
    A, B = _make_inputs(n, k, m, dtype, distribution)
    _assert_logprobs(
        ops.wvSplitKrc(A, B, cu, None, fast_skinny_gemm=fast_skinny_gemm),
        torch.nn.functional.linear(A, B),
        label=f"wvSplitKrc lp fast={fast_skinny_gemm} {n}x{k}x{m} "
        f"{dtype} {distribution}",
    )


@pytest.mark.parametrize("n,k,m", [(16, 2880, 128), (128, 2880, 640)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
def test_wvsplitkrc_logprobs_determinism(n, k, m, dtype):
    cu = get_cu_count()
    if not _fits_rc(n, k, m, cu):
        pytest.skip("Shape exceeds CU capacity")
    A, B = _make_inputs(n, k, m, dtype, "normal")
    _assert_logprobs_deterministic(
        lambda: ops.wvSplitKrc(A, B, cu, None, fast_skinny_gemm=False),
        label=f"wvSplitKrc lp det {n}x{k}x{m} {dtype}",
    )


@pytest.mark.parametrize("n,k,m", [(4, 4096, 4096), (1, 14336, 1024), (2, 24576, 2048)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="rocm fp8 only",
)
def test_fp8_logprobs(n, k, m, dtype):
    """FP8 logprob check with relaxed top-1 logprob threshold.

    FP8 quantization noise is larger than dtype rounding noise, so
    we allow 0.02 nats of logprob difference (vs 0.01 for bf16/fp16).
    This corresponds to ~2% multiplicative probability error.
    """
    A, B, sa, sb = _make_inputs_fp8(n, k, m, "normal")
    ref = torch._scaled_mm(A, B.t(), out_dtype=dtype, scale_a=sa, scale_b=sb)
    out = ops.wvSplitKQ(B, A, dtype, sa, sb, get_cu_count())
    _assert_logprobs(
        out, ref, label=f"FP8 lp {n}x{k}x{m} {dtype}", top1_lp_max_diff=0.02
    )


@pytest.mark.parametrize("n,k,m", [(4, 4096, 4096), (2, 24576, 2048)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="rocm fp8 only",
)
def test_fp8_logprobs_determinism(n, k, m, dtype):
    A, B, sa, sb = _make_inputs_fp8(n, k, m, "normal")
    _assert_logprobs_deterministic(
        lambda: ops.wvSplitKQ(B, A, dtype, sa, sb, get_cu_count()),
        label=f"FP8 lp det {n}x{k}x{m} {dtype}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@torch.inference_mode()
def test_wvsplitk_nan_propagation(dtype):
    """NaN must propagate to the affected output row, not leak to others.

    Silent NaN absorption masks upstream bugs (e.g., uninitialized
    memory, division by zero in prior layers).  NaN leaking across
    rows indicates an indexing bug in the kernel.
    """
    n, k, m = 2, 256, 256
    torch.manual_seed(0)
    A = torch.rand(n, k, dtype=dtype, device="cuda") - 0.5
    B = torch.rand(m, k, dtype=dtype, device="cuda") - 0.5
    A[0, 0] = float("nan")
    out = ops.wvSplitK(B, A.view(-1, A.size(-1)), get_cu_count())
    assert out[0].isnan().any(), "NaN lost in wvSplitK row 0"
    assert not out[1].isnan().any(), "NaN leaked in wvSplitK row 1"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@torch.inference_mode()
def test_llmm1_nan_propagation(dtype):
    """NaN must propagate to the output.

    LLMM1 requires N=1, so we cannot test row isolation -- only that
    NaN is not silently dropped by the half-precision FMA chain or
    the warp shuffle reduction.
    """
    n, k, m = 1, 256, 256
    torch.manual_seed(0)
    A = torch.rand(n, k, dtype=dtype, device="cuda")
    B = torch.rand(m, k, dtype=dtype, device="cuda")
    A[0, 0] = float("nan")
    out = ops.LLMM1(B, A, 4)
    assert out[0].isnan().any(), "NaN lost in LLMM1"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
@pytest.mark.skipif(not on_gfx950(), reason="gfx950 only")
def test_wvsplitkrc_nan_propagation(dtype):
    """NaN row isolation for wvSplitKrc.

    The K-split-and-reduce architecture (store partial sums to global,
    then sum) could mask NaN if the reduction uses non-NaN-propagating
    operations. We verify both propagation and isolation.
    """
    cu = get_cu_count()
    n, k, m = 16, 2880, 128
    if not _fits_rc(n, k, m, cu):
        pytest.skip("Shape exceeds CU capacity")
    torch.manual_seed(0)
    A = torch.rand(n, k, dtype=dtype, device="cuda") - 0.5
    B = torch.rand(m, k, dtype=dtype, device="cuda") - 0.5
    A[0, 0] = float("nan")
    out = ops.wvSplitKrc(A, B, cu, None)
    assert out[0].isnan().any(), "NaN lost in wvSplitKrc row 0"
    assert not out[1].isnan().any(), "NaN leaked in wvSplitKrc row 1"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
def test_zero_and_bias(dtype):
    """Zero inputs must produce exactly zero; with bias, exactly the bias.

    This catches:
    1. Uninitialized accumulators (would produce non-zero from garbage)
    2. Incorrect bias indexing (would produce wrong values)
    3. Spurious accumulation artifacts (e.g., from LDS residuals)

    The rtol=0, atol=0 check is exact - zero x anything is exactly
    zero in IEEE 754, with no tolerance needed.
    """
    cu = get_cu_count()
    n, k, m = 2, 256, 256
    A = torch.zeros(n, k, dtype=dtype, device="cuda")
    B = torch.zeros(m, k, dtype=dtype, device="cuda")
    out = ops.wvSplitK(B, A.view(-1, A.size(-1)), cu)
    assert torch.all(out == 0), f"0x0 produced non-zero: max={out.abs().max().item()}"
    BIAS = torch.rand(m, dtype=dtype, device="cuda") - 0.5
    out_b = ops.wvSplitK(B, A.view(-1, A.size(-1)), cu, BIAS)
    torch.testing.assert_close(out_b, BIAS.unsqueeze(0).expand(n, -1), atol=0, rtol=0)


@pytest.mark.parametrize("num_runs", [10])
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
def test_e2e_logprob_reproducibility(num_runs, enforce_eager, vllm_runner):
    """End-to-end logprob reproducibility across repeated inference runs.

    With enforce_eager=True (no CUDA graphs), results must be bitwise identical.
    With enforce_eager=False (CUDA graphs enabled), we allow a small tolerance
    (1e-6) to account for non-determinism introduced by graph capture/replay.
    """
    # When CUDA graphs are enabled, allow tolerance up to 1e-6.
    # When eager, require exact bitwise reproducibility.
    atol = 1e-6 if not enforce_eager else 0.0

    model = "TitanML/tiny-mixtral"
    prompts = [
        "The capital of France is",
        "In quantum computing, a qubit",
        "def fibonacci(n):\n",
    ]
    max_tokens = 32
    top_logprobs = 5

    all_runs = []
    for run_idx in range(num_runs):
        with vllm_runner(
            model,
            dtype="half",
            enforce_eager=enforce_eager,
            max_num_seqs=1,
            seed=0,
            enable_prefix_caching=False,
        ) as llm:
            outputs = llm.generate_greedy_logprobs(prompts, max_tokens, top_logprobs)
            all_runs.append(outputs)

    mode_label = "eager" if enforce_eager else "cuda-graph"

    # Per-run comparison tables and summary collection
    summary_rows = []
    for run_idx in range(1, num_runs):
        print(f"\n{'=' * 80}")
        print(f"  Run 0 vs Run {run_idx}  (mode={mode_label}, atol={atol:.1e})")
        print(f"{'=' * 80}")

        for prompt_idx, prompt in enumerate(prompts):
            ref_token_ids, ref_text, ref_lps = all_runs[0][prompt_idx]
            cur_token_ids, cur_text, cur_lps = all_runs[run_idx][prompt_idx]

            token_match = sum(
                r == c for r, c in zip(ref_token_ids, cur_token_ids)
            ) / len(ref_token_ids)

            # Compute logprob stats across all positions
            lp_diffs = []
            top5_mismatches = 0
            for pos in range(len(ref_lps)):
                ref_top = ref_lps[pos]
                cur_top = cur_lps[pos]
                if set(ref_top.keys()) != set(cur_top.keys()):
                    top5_mismatches += 1
                for token_id in set(ref_top.keys()) & set(cur_top.keys()):
                    d = abs(ref_top[token_id].logprob - cur_top[token_id].logprob)
                    lp_diffs.append(d)

            max_lp_diff = max(lp_diffs) if lp_diffs else 0.0
            mean_lp_diff = sum(lp_diffs) / len(lp_diffs) if lp_diffs else 0.0
            within_tol = all(d <= atol for d in lp_diffs)

            print(f'\n  Prompt {prompt_idx}: "{prompt[:40]}..."')
            print(f"  {'─' * 60}")
            print(
                f"  {'Token match:':<25} {token_match:>8.2%} "
                f"({sum(r == c for r, c in zip(ref_token_ids, cur_token_ids))}"
                f"/{len(ref_token_ids)})"
            )
            print(
                f"  {'Top-5 set mismatches:':<25} {top5_mismatches:>8d} "
                f"/ {len(ref_lps)} positions"
            )
            print(f"  {'Max logprob diff:':<25} {max_lp_diff:>12.2e}")
            print(f"  {'Mean logprob diff:':<25} {mean_lp_diff:>12.2e}")
            print(f"  {'Within tolerance:':<25} {'YES' if within_tol else 'NO':>8}")

            summary_rows.append(
                {
                    "run": f"0 vs {run_idx}",
                    "prompt": prompt_idx,
                    "token_match": token_match,
                    "top5_mismatch": top5_mismatches,
                    "max_lp_diff": max_lp_diff,
                    "mean_lp_diff": mean_lp_diff,
                    "within_tol": within_tol,
                }
            )

            # Assertions
            assert ref_token_ids == cur_token_ids, (
                f"[{mode_label}] Token mismatch run 0 vs {run_idx}, "
                f"prompt {prompt_idx}: "
                f"ref={ref_token_ids[:10]}... "
                f"cur={cur_token_ids[:10]}..."
            )

            for pos in range(len(ref_lps)):
                ref_top = ref_lps[pos]
                cur_top = cur_lps[pos]
                ref_ids = set(ref_top.keys())
                cur_ids = set(cur_top.keys())

                assert ref_ids == cur_ids, (
                    f"[{mode_label}] Top-{top_logprobs} set mismatch at pos {pos}, "
                    f"run 0 vs {run_idx}, prompt {prompt_idx}: "
                    f"ref={ref_ids} cur={cur_ids}"
                )

                for token_id in ref_ids:
                    ref_val = ref_top[token_id].logprob
                    cur_val = cur_top[token_id].logprob
                    diff = abs(ref_val - cur_val)
                    assert diff <= atol, (
                        f"[{mode_label}] Logprob mismatch at pos {pos}, "
                        f"token {token_id}, "
                        f"run 0 vs {run_idx}, prompt {prompt_idx}: "
                        f"ref={ref_val:.10f} cur={cur_val:.10f} "
                        f"diff={diff:.2e} > atol={atol:.1e}"
                    )

    # Summary table
    print(f"\n{'=' * 80}")
    print(
        f"  REPRODUCIBILITY SUMMARY "
        f"({num_runs} runs, {len(prompts)} prompts, "
        f"mode={mode_label}, atol={atol:.1e})"
    )
    print(f"{'=' * 80}")
    print(
        f"  {'Comparison':<10} {'Prompt':<7} {'Tok Match':>10} "
        f"{'Top5 Miss':>10} {'Max LP Diff':>12} "
        f"{'Mean LP Diff':>13} {'In Tol':>8}"
    )
    print(f"  {'─' * 72}")
    for row in summary_rows:
        print(
            f"  {row['run']:<10} {row['prompt']:<7} "
            f"{row['token_match']:>9.2%} "
            f"{row['top5_mismatch']:>10d} "
            f"{row['max_lp_diff']:>12.2e} "
            f"{row['mean_lp_diff']:>13.2e} "
            f"{'YES' if row['within_tol'] else 'NO':>8}"
        )

    all_within = all(r["within_tol"] for r in summary_rows)
    all_token = all(r["token_match"] == 1.0 for r in summary_rows)
    worst_lp = max(r["max_lp_diff"] for r in summary_rows)
    print(f"  {'─' * 72}")
    print(f"  All within tolerance: {'YES' if all_within else 'NO'}")
    print(f"  All tokens match:    {'YES' if all_token else 'NO'}")
    print(f"  Worst logprob diff:  {worst_lp:.2e}")
    print(f"{'=' * 80}\n")


@pytest.mark.parametrize("num_runs", [10])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="rocm only")
def test_e2e_logprob_stability(num_runs, vllm_runner):
    """Softer e2e check: logprobs within 0.001 nats where tokens agree.

    Same locked-down settings as the strict test. If the strict test
    fails but this passes, the non-determinism source is small enough
    to only affect near-tied logits. If both fail, something
    fundamental is wrong beyond argmax sensitivity.
    """
    model = "TitanML/tiny-mixtral"
    prompts = [
        "The capital of France is",
        "In quantum computing, a qubit",
        "def fibonacci(n):\n",
    ]
    max_tokens = 32
    top_logprobs = 5

    all_runs = []
    for run_idx in range(num_runs):
        with vllm_runner(
            model,
            dtype="half",
            enforce_eager=True,
            max_num_seqs=1,
            seed=0,
            enable_prefix_caching=False,
        ) as llm:
            outputs = llm.generate_greedy_logprobs(prompts, max_tokens, top_logprobs)
            all_runs.append(outputs)

    # Per-run comparison tables and summary collection
    summary_rows = []

    for run_idx in range(1, num_runs):
        print(f"\n{'=' * 80}")
        print(f"  Run 0 vs Run {run_idx}")
        print(f"{'=' * 80}")

        for prompt_idx, prompt in enumerate(prompts):
            ref_token_ids, ref_text, ref_lps = all_runs[0][prompt_idx]
            cur_token_ids, cur_text, cur_lps = all_runs[run_idx][prompt_idx]

            token_match = sum(
                r == c for r, c in zip(ref_token_ids, cur_token_ids)
            ) / len(ref_token_ids)

            # Compute logprob stats only at agreeing positions
            lp_diffs_agree = []
            positions_checked = 0
            positions_skipped = 0
            for pos in range(min(len(ref_lps), len(cur_lps))):
                if (
                    pos < len(ref_token_ids)
                    and pos < len(cur_token_ids)
                    and ref_token_ids[pos] != cur_token_ids[pos]
                ):
                    positions_skipped += 1
                    continue
                positions_checked += 1
                ref_top1_id = max(ref_lps[pos], key=lambda k: ref_lps[pos][k].logprob)
                if ref_top1_id in cur_lps[pos]:
                    d = abs(
                        ref_lps[pos][ref_top1_id].logprob
                        - cur_lps[pos][ref_top1_id].logprob
                    )
                    lp_diffs_agree.append(d)

            max_lp_diff = max(lp_diffs_agree) if lp_diffs_agree else 0.0
            mean_lp_diff = (
                sum(lp_diffs_agree) / len(lp_diffs_agree) if lp_diffs_agree else 0.0
            )

            print(f'\n  Prompt {prompt_idx}: "{prompt[:40]}..."')
            print(f"  {'─' * 60}")
            print(
                f"  {'Token match:':<25} {token_match:>8.2%} "
                f"({sum(r == c for r, c in zip(ref_token_ids, cur_token_ids))}"
                f"/{len(ref_token_ids)})"
            )
            print(f"  {'Positions checked:':<25} {positions_checked:>8d}")
            print(f"  {'Positions skipped:':<25} {positions_skipped:>8d}")
            print(f"  {'Max top-1 LP diff:':<25} {max_lp_diff:>12.2e}")
            print(f"  {'Mean top-1 LP diff:':<25} {mean_lp_diff:>12.2e}")

            summary_rows.append(
                {
                    "run": f"0 vs {run_idx}",
                    "prompt": prompt_idx,
                    "token_match": token_match,
                    "pos_checked": positions_checked,
                    "pos_skipped": positions_skipped,
                    "max_lp_diff": max_lp_diff,
                    "mean_lp_diff": mean_lp_diff,
                }
            )

            # Assertions
            assert token_match >= 0.70, (
                f"Token match rate {token_match:.2%} < 70%, "
                f"run 0 vs {run_idx}, prompt {prompt_idx}"
            )

            for pos in range(min(len(ref_lps), len(cur_lps))):
                if (
                    pos < len(ref_token_ids)
                    and pos < len(cur_token_ids)
                    and ref_token_ids[pos] != cur_token_ids[pos]
                ):
                    continue
                ref_top1_id = max(ref_lps[pos], key=lambda k: ref_lps[pos][k].logprob)
                if ref_top1_id in cur_lps[pos]:
                    diff = abs(
                        ref_lps[pos][ref_top1_id].logprob
                        - cur_lps[pos][ref_top1_id].logprob
                    )
                    assert diff < 0.001, (
                        f"Logprob drift {diff:.6f} >= 0.001 at pos {pos}, "
                        f"run 0 vs {run_idx}, prompt {prompt_idx}"
                    )

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  STABILITY SUMMARY ({num_runs} runs, {len(prompts)} prompts)")
    print(f"{'=' * 80}")
    print(
        f"  {'Comparison':<10} {'Prompt':<7} {'Tok Match':>10} "
        f"{'Checked':>8} {'Skipped':>8} "
        f"{'Max LP Diff':>12} {'Mean LP Diff':>13}"
    )
    print(f"  {'─' * 72}")
    for row in summary_rows:
        print(
            f"  {row['run']:<10} {row['prompt']:<7} "
            f"{row['token_match']:>9.2%} "
            f"{row['pos_checked']:>8d} "
            f"{row['pos_skipped']:>8d} "
            f"{row['max_lp_diff']:>12.2e} "
            f"{row['mean_lp_diff']:>13.2e}"
        )

    worst_match = min(r["token_match"] for r in summary_rows)
    worst_lp = max(r["max_lp_diff"] for r in summary_rows)
    total_skipped = sum(r["pos_skipped"] for r in summary_rows)
    total_checked = sum(r["pos_checked"] for r in summary_rows)
    print(f"  {'─' * 72}")
    print(f"  Worst token match: {worst_match:.2%}")
    print(f"  Worst logprob diff (agreeing positions): {worst_lp:.2e}")
    print(f"  Total positions checked/skipped: {total_checked}/{total_skipped}")
    print(f"{'=' * 80}\n")
