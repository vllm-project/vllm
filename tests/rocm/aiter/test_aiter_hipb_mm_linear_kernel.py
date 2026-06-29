# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import importlib
import importlib.util
import os

import pytest
import torch

from tests.utils import TestFP8Layer
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = [
    pytest.mark.skipif(
        not (
            current_platform.is_rocm()
            and current_platform.supports_fp8()
            and aiter_available
        ),
        reason="Requires ROCm + FP8 support + aiter",
    ),
    pytest.mark.usefixtures("default_vllm_config"),
]


@pytest.fixture
def enable_hipb_mm_kernel(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR_HIPBMM", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    rocm_aiter_ops.refresh_env_variables()


def _make_config(
    *,
    weight_quant_key=kFp8StaticChannelSym,
    out_dtype: torch.dtype = torch.bfloat16,
    weight_shape: tuple[int, int] = (512, 4096),
) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=out_dtype,
    )


def _find_csv_row(path: str, m: int, n: int, k: int) -> dict | None:
    if not os.path.exists(path):
        return None

    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:
                if (
                    int(row.get("m", -1)) == m
                    and int(row.get("n", -1)) == n
                    and int(row.get("k", -1)) == k
                ):
                    return dict(row)
            except (TypeError, ValueError):
                continue
    return None


def _skip_if_no_hipb_mm_solution(exc: RuntimeError) -> None:
    if "hipblasLtMatmulAlgoGetHeuristic found 0 valid solutions" in str(exc):
        pytest.skip(
            "hipb_mm bpreshuffle path has no valid hipBLASLt solution on "
            "this ROCm stack."
        )


def _check_bpreshuffle_runtime_support(weight_shape: tuple[int, int], num_tokens: int):
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    x = torch.randn(num_tokens, weight_shape[1], dtype=torch.bfloat16, device="cuda")
    w = torch.randn(weight_shape, dtype=torch.bfloat16, device="cuda")

    aiter.hipb_create_extension()
    x_q, x_scale = aiter.pertoken_quant(x, quant_dtype=current_platform.fp8_dtype())
    w_q, w_scale = aiter.pertoken_quant(w, quant_dtype=current_platform.fp8_dtype())

    try:
        aiter.hipb_mm(
            x_q,
            shuffle_weight(w_q, layout=(16, 16)).t(),
            solution_index=-1,
            out_dtype=torch.bfloat16,
            scaleA=x_scale,
            scaleB=w_scale.t().contiguous(),
            scaleOut=None,
            bpreshuffle=True,
        )
    except RuntimeError as exc:
        _skip_if_no_hipb_mm_solution(exc)
        raise


def test_hipb_mm_kernel_requires_hipbmm_flag(monkeypatch: pytest.MonkeyPatch):
    # The kernel rejects when `is_hip_fp8bmm_enabled()` is False. That helper
    # requires AITER + AITER_LINEAR + MI3xx, so dropping AITER_LINEAR exercises
    # the rejection branch.
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "0")
    monkeypatch.delenv("VLLM_ROCM_USE_AITER_LINEAR_HIPBMM", raising=False)
    rocm_aiter_ops.refresh_env_variables()

    is_supported, reason = AiterHipbMMPerTokenFp8ScaledMMLinearKernel.is_supported()

    assert not is_supported
    assert reason == (
        "requires setting `VLLM_ROCM_USE_AITER=1`, "
        "`VLLM_ROCM_USE_AITER_LINEAR=1`, "
        "and `VLLM_ROCM_USE_AITER_LINEAR_HIPBMM=1`."
    )


def test_hipb_mm_flag_enables_hip_online_tuning(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.envs as envs_mod
    import vllm.platforms.rocm as rocm_mod

    # The rocm.py gate requires all three AITER flags (and MI3xx) to auto-set
    # HIP_ONLINE_TUNING.
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR_HIPBMM", "1")

    try:
        importlib.reload(envs_mod)
        importlib.reload(rocm_mod)
        assert envs_mod.VLLM_ROCM_USE_AITER
        assert envs_mod.VLLM_ROCM_USE_AITER_LINEAR
        assert envs_mod.VLLM_ROCM_USE_AITER_LINEAR_HIPBMM
        assert os.environ.get("HIP_ONLINE_TUNING") == "1"
    finally:
        monkeypatch.undo()
        os.environ.pop("HIP_ONLINE_TUNING", None)
        importlib.reload(envs_mod)
        importlib.reload(rocm_mod)
        rocm_aiter_ops.refresh_env_variables()


def test_hipb_mm_kernel_can_implement_success(enable_hipb_mm_kernel):
    can_implement, reason = AiterHipbMMPerTokenFp8ScaledMMLinearKernel.can_implement(
        _make_config()
    )

    assert can_implement
    assert reason is None


@pytest.mark.parametrize(
    ("config", "expected_reason"),
    [
        (
            _make_config(weight_quant_key=kFp8StaticTensorSym),
            "requires per token activation scales and per channel weight scales.",
        ),
        (
            _make_config(out_dtype=torch.float16),
            "requires bfloat16 output dtype.",
        ),
        (
            _make_config(weight_shape=(8, 4090)),
            "requires N >= 16 and both N and K divisible by 16, "
            "received N=8 and K=4090.",
        ),
    ],
)
def test_hipb_mm_kernel_can_implement_rejects_unsupported_configs(
    enable_hipb_mm_kernel,
    config: FP8ScaledMMLinearLayerConfig,
    expected_reason: str,
):
    can_implement, reason = AiterHipbMMPerTokenFp8ScaledMMLinearKernel.can_implement(
        config
    )

    assert not can_implement
    assert reason == expected_reason


def test_hipb_mm_kernel_process_weights_after_loading_shuffles_weights(
    enable_hipb_mm_kernel,
):
    weight_shape = (512, 4096)
    kernel = AiterHipbMMPerTokenFp8ScaledMMLinearKernel(
        _make_config(weight_shape=weight_shape),
        layer_param_names=("weight", "weight_scale", "input_scale", "input_scale_ub"),
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.rand(weight_shape, device="cuda").to(current_platform.fp8_dtype()).t(),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.rand((weight_shape[0], 1), dtype=torch.float32, device="cuda"),
        requires_grad=False,
    )
    layer.input_scale = None
    layer.input_scale_ub = None

    original_weight = layer.weight.detach().clone()
    original_weight_scale = layer.weight_scale.detach().clone()

    kernel.process_weights_after_loading(layer)

    # process_weights_after_loading now pre-applies the transposes that used
    # to live in _rocm_aiter_hipb_mm_fp8_impl, so the stored weight is the
    # shuffled tensor with a trailing `.t()` view, and the stored weight scale
    # is its transposed-contiguous form.
    expected_weight = rocm_aiter_ops.shuffle_weight(
        original_weight.t().contiguous()
    ).t()
    torch.testing.assert_close(layer.weight, expected_weight)

    expected_weight_scale = original_weight_scale.t().contiguous()
    torch.testing.assert_close(layer.weight_scale, expected_weight_scale)


def test_hipb_mm_kernel_forward_matches_raw_aiter_hipb_mm(enable_hipb_mm_kernel):
    import aiter

    weight_shape = (512, 4096)
    _check_bpreshuffle_runtime_support(weight_shape, num_tokens=32)

    layer = TestFP8Layer(
        weight_shape=weight_shape,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_quant_key=kFp8StaticChannelSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        force_kernel=AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    )

    # hipb_mm uses a transposed-result GEMM internally, so the flattened token
    # count becomes the effective N dimension passed into hipBLASLt. Keep it
    # aligned to avoid heuristic failures for tiny N.
    x = torch.randn(2, 16, weight_shape[1], dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(weight_shape[0], dtype=torch.bfloat16, device="cuda")

    try:
        out = layer(x, bias)
    except RuntimeError as exc:
        _skip_if_no_hipb_mm_solution(exc)
        raise

    x_2d = x.view(-1, x.shape[-1])
    x_q, x_scale = layer.kernel.quant_fp8(
        x_2d,
        layer.input_scale,
        layer.input_scale_ub,
    )
    try:
        # process_weights_after_loading already applies the trailing `.t()` on
        # the shuffled weight and the `.t().contiguous()` on the weight scale,
        # so the raw aiter call uses them directly.
        expected = aiter.hipb_mm(
            x_q,
            layer.weight,
            solution_index=-1,
            bias=bias,
            out_dtype=torch.bfloat16,
            scaleA=x_scale,
            scaleB=layer.weight_scale,
            scaleOut=None,
            bpreshuffle=True,
        ).view(*out.shape)
    except RuntimeError as exc:
        _skip_if_no_hipb_mm_solution(exc)
        raise

    assert isinstance(layer.kernel, AiterHipbMMPerTokenFp8ScaledMMLinearKernel)
    assert out.shape == (2, 16, weight_shape[0])
    torch.testing.assert_close(out, expected)


def test_hipb_mm_kernel_forward_accuracy(enable_hipb_mm_kernel):
    """Kernel output should match a dequantized fp32 reference within
    fp8 per-token / per-channel quantization noise."""
    weight_shape = (512, 4096)  # (N, K)
    num_tokens = 32
    _check_bpreshuffle_runtime_support(weight_shape, num_tokens=num_tokens)

    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = get_fp8_min_max()[1]
    device = torch.device("cuda")

    # Build a bf16 weight and quantize per output channel (one scale per row).
    w_bf16 = torch.randn(weight_shape, dtype=torch.bfloat16, device=device)
    w_amax = w_bf16.abs().amax(dim=1, keepdim=True).to(torch.float32)
    w_scale = (w_amax / fp8_max).clamp(min=1e-12)
    w_fp8 = (w_bf16.to(torch.float32) / w_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    w_dequant = w_fp8.to(torch.float32) * w_scale

    bias = torch.randn(weight_shape[0], dtype=torch.bfloat16, device=device)

    layer = torch.nn.Module()
    # Pre-`process_weights_after_loading` convention: weight stored as the
    # `[K, N]` view of the fp8 tensor.
    layer.weight = torch.nn.Parameter(w_fp8.t(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_scale, requires_grad=False)
    layer.input_scale = None
    layer.input_scale_ub = None

    kernel = AiterHipbMMPerTokenFp8ScaledMMLinearKernel(
        _make_config(weight_shape=weight_shape),
        layer_param_names=("weight", "weight_scale", "input_scale", "input_scale_ub"),
    )
    kernel.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, weight_shape[1], dtype=torch.bfloat16, device=device)

    try:
        out = kernel.apply_weights(layer, x, bias)
    except RuntimeError as exc:
        _skip_if_no_hipb_mm_solution(exc)
        raise

    # Reference: quantize x per-token the same way the kernel does, then run
    # the matmul in fp32 against the dequantized weight. This isolates plumbing
    # / reduction bugs from inherent fp8 quantization noise.
    x_amax = x.abs().amax(dim=1, keepdim=True).to(torch.float32)
    x_scale_ref = (x_amax / fp8_max).clamp(min=1e-12)
    x_q = (x.to(torch.float32) / x_scale_ref).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    x_dequant = x_q.to(torch.float32) * x_scale_ref
    expected = (x_dequant @ w_dequant.t() + bias.to(torch.float32)).to(torch.bfloat16)

    assert out.shape == (num_tokens, weight_shape[0])
    # K=4096 fp8 reduction leaves room for accumulation order drift and
    # catastrophic cancellation on near-zero outputs; tolerances are loose
    # enough to absorb that but tight enough to catch wrong layouts, missing
    # bias, swapped scales, etc.
    torch.testing.assert_close(out, expected, atol=5.0, rtol=0.1)


def test_hipb_mm_kernel_online_tuning_writes_csv(
    enable_hipb_mm_kernel,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    weight_shape = (256, 4096)
    cache_file = tmp_path / "hip_online_tuning_res.csv"

    _check_bpreshuffle_runtime_support(weight_shape, num_tokens=16)

    monkeypatch.setenv("HIP_ONLINE_TUNING", "1")
    monkeypatch.chdir(tmp_path)

    layer = TestFP8Layer(
        weight_shape=weight_shape,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_quant_key=kFp8StaticChannelSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        force_kernel=AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    )

    # The effective heuristic N dimension is the flattened token count.
    x = torch.randn(16, weight_shape[1], dtype=torch.bfloat16, device="cuda")
    try:
        out = layer(x)
    except RuntimeError as exc:
        _skip_if_no_hipb_mm_solution(exc)
        raise
    torch.accelerator.synchronize()

    assert out.shape == (16, weight_shape[0])
    assert cache_file.exists()

    # hipb_mm records the internal GEMM dimensions used by hipBLASLt after its
    # transposed-result transformation.
    row = _find_csv_row(
        str(cache_file),
        m=weight_shape[0],
        n=x.shape[0],
        k=weight_shape[1],
    )
    assert row is not None
