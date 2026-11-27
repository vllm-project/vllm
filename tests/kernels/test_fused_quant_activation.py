# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    KernelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import (
    GELU,
    ReLUSquaredActivation,
    SiluAndMul,
)
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMxFp8Config,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
)
from vllm.model_executor.models.qwen3 import Qwen3MLP
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
QUANT_DTYPES = [current_platform.fp8_dtype()]
NUM_TOKENS = [1, 17, 86, 1234, 3045]  # Arbitrary values for testing
HIDDEN_SIZES = [16, 48, 128, 1562, 4096]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


def ref_impl(
    silu_and_mul: SiluAndMul, x: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    silu_and_mul_out = silu_and_mul.forward_native(x)
    out, scales = ops.scaled_fp8_quant(silu_and_mul_out, scale)
    return out


def ops_impl(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_shape = (x.shape[0], x.shape[1] // 2)
    out = torch.empty(out_shape, dtype=current_platform.fp8_dtype(), device=x.device)
    torch.ops._C.silu_and_mul_quant(out, x, scale)
    return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    layer = SiluAndMul()

    # Make inputs
    scale = torch.randn((1), device=device, dtype=torch.float32)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    ref_out = ref_impl(layer, x, scale)
    ops_out = ops_impl(x, scale)

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    assert ref_out.shape == ops_out.shape
    assert torch.allclose(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )
    opcheck(torch.ops._C.silu_and_mul_quant, (ops_out, x, scale))


# ---------------------------------------------------------------------------
# Tests for maybe_fused_act_quant interface
# ---------------------------------------------------------------------------


class MockLinearFp8Static(torch.nn.Module):
    """Mock linear layer advertising kFp8StaticTensorSym."""

    def __init__(self, input_scale: torch.Tensor):
        super().__init__()
        self._input_quant_key = kFp8StaticTensorSym
        self.input_scale = input_scale


class MockLinearFp8Dynamic128(torch.nn.Module):
    """Mock linear layer advertising kFp8Dynamic128Sym."""

    def __init__(self):
        super().__init__()
        self._input_quant_key = kFp8Dynamic128Sym


class MockLinearNoQuant(torch.nn.Module):
    """Mock linear layer with no input quantization key."""

    pass


class MockLinearNvfp4(torch.nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self._input_quant_key = kNvfp4Dynamic
        self.input_global_scale_inv = scale


@pytest.fixture
def native_vllm_config():
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=["none"]
        )
    )
    with set_current_vllm_config(config):
        yield config


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="Act + NVFP4 fusion requires SM100+",
)
@pytest.mark.parametrize(
    "make_act, dtype",
    [
        pytest.param(
            partial(ReLUSquaredActivation, compile_native=False),
            torch.bfloat16,
            id="relu2-bf16",
        ),
        pytest.param(GELU, torch.float16, id="gelu-fp16"),
        pytest.param(GELU, torch.bfloat16, id="gelu-bf16"),
    ],
)
@pytest.mark.parametrize("shape", [(1, 16), (17, 80), (129, 1344), (256, 5376)])
@pytest.mark.parametrize("scale_value", [0.0009765625, 17.125])
@torch.inference_mode()
def test_act_nvfp4_matches_unfused(
    native_vllm_config, make_act, dtype, shape, scale_value
):
    """Match unfused act + NVFP4 bitwise and zero every padded scale."""
    from tests.kernels.quantization.test_nvfp4_quant import recover_swizzled_scales

    x = torch.randn(shape, dtype=dtype, device="cuda")
    # All-negative rows: GELU is negative there, so block scales must use |x|.
    x[::2] = -x[::2].abs()
    scale = torch.tensor(scale_value, dtype=torch.float32, device=x.device)
    linear = MockLinearNvfp4(scale)
    act = make_act()
    compiled_act = torch.compile(act.forward_native, fullgraph=True)
    expected = ops.scaled_fp4_quant(compiled_act(x), scale)
    before = x.clone()
    actual = maybe_fused_act_quant(act, x, linear)
    assert isinstance(actual, QuantizedActivation)
    assert torch.equal(x, before)
    assert torch.equal(actual.data, expected[0])
    m, k = shape
    expected_scale = recover_swizzled_scales(expected[1], m, k)
    actual_scale = recover_swizzled_scales(actual.scale, m, k, include_padding=True)
    torch.testing.assert_close(
        actual_scale[:m, : k // 16], expected_scale, rtol=0, atol=0
    )
    actual_scale[:m, : k // 16] = 0
    assert not torch.count_nonzero(actual_scale)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="Act + NVFP4 fusion requires SM100+",
)
@pytest.mark.parametrize("width", [0, 15, 17, 33])
@torch.inference_mode()
def test_relu2_nvfp4_unsupported_width_preserves_fallback(native_vllm_config, width):
    x = torch.randn((3, width), dtype=torch.bfloat16, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device=x.device)
    act = ReLUSquaredActivation(compile_native=False)
    result = maybe_fused_act_quant(act, x, MockLinearNvfp4(scale))
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, act.forward_native(x), rtol=0, atol=0)


class MockLinearRequiresUnquantized(MockLinearFp8Static):
    """Mock consumer that needs the original activation for another branch."""

    requires_unquantized_input = True


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="Act + NVFP4 fusion requires SM100+",
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 3.0e20, 100.0])
@pytest.mark.parametrize("scale_value", [2.0**-30, 1.0, 64.0, 1.0e20])
@torch.inference_mode()
def test_relu2_nvfp4_nonfinite_and_saturation(native_vllm_config, value, scale_value):
    x = torch.tensor([[value] + [1.0] * 15], dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(scale_value, dtype=torch.float32, device=x.device)
    act = ReLUSquaredActivation(compile_native=False)
    compiled_relu = torch.compile(act.forward_native, fullgraph=True)
    expected = ops.scaled_fp4_quant(
        compiled_relu(x), scale, backend="flashinfer-cutedsl"
    )
    actual = maybe_fused_act_quant(act, x, MockLinearNvfp4(scale))
    assert isinstance(actual, QuantizedActivation)
    assert torch.equal(actual.data, expected[0])
    assert torch.equal(actual.scale.view(torch.uint8), expected[1].view(torch.uint8))


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="Act + NVFP4 fusion requires SM100+",
)
@pytest.mark.parametrize("mode", ["native", "compiled_native", "cuda"])
@torch.inference_mode()
def test_relu2_nvfp4_preserves_selected_activation(mode):
    """CUDA ReLU2 suppresses NaN and must not use the native-semantics producer."""
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["all" if mode == "cuda" else "none"],
        )
    )
    with set_current_vllm_config(config):
        act = ReLUSquaredActivation(compile_native=mode == "compiled_native")
        x = torch.tensor(
            [[float("nan")] + [1.0] * 15], device="cuda", dtype=torch.bfloat16
        )
        scale = torch.ones((), device=x.device, dtype=torch.float32)
        expected = act(x)
        dispatch = maybe_fused_act_quant
        if mode == "compiled_native":
            dispatch = torch.compile(dispatch, fullgraph=True)
        actual = dispatch(act, x, MockLinearNvfp4(scale))
        if mode == "cuda":
            assert isinstance(actual, torch.Tensor)
            assert expected[0, 0].item() == 0
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert isinstance(actual, QuantizedActivation)
            expected_data, expected_scale = ops.scaled_fp4_quant(expected, scale)
            assert torch.equal(actual.data, expected_data)
            assert torch.equal(
                actual.scale.view(torch.uint8), expected_scale.view(torch.uint8)
            )


@pytest.mark.parametrize("token_shape", [(1,), (16,), (128,), (1, 17), (2, 17)])
@pytest.mark.parametrize("hidden_size", [128, 512, 1024])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_maybe_fused_act_quant_fp8_static(
    default_vllm_config,
    token_shape: tuple[int, ...],
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Test maybe_fused_act_quant with FP8 static per-tensor quantization."""
    device = "cuda:0"
    torch.set_default_device(device)

    act_fn = SiluAndMul()
    scale = torch.tensor([0.5], device=device, dtype=torch.float32)
    linear = MockLinearFp8Static(scale)

    x = torch.randn(*token_shape, hidden_size * 2, dtype=dtype, device=device)
    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, QuantizedActivation)
    assert result.quant_key == kFp8StaticTensorSym
    assert result.data.dtype == current_platform.fp8_dtype()
    assert result.orig_dtype == dtype
    assert result.orig_shape == (*token_shape, hidden_size)
    assert result.data.shape == result.orig_shape

    ref_out = ref_impl(act_fn, x.reshape(-1, hidden_size * 2), scale).reshape(
        result.orig_shape
    )
    torch.testing.assert_close(result.data.to(torch.float32), ref_out.to(torch.float32))


@pytest.mark.parametrize("token_shape", [(1,), (16,), (128,), (1, 17), (2, 17)])
@pytest.mark.parametrize("hidden_size", [128, 512, 1024])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_maybe_fused_act_quant_fp8_dynamic_block(
    default_vllm_config,
    token_shape: tuple[int, ...],
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Test maybe_fused_act_quant with FP8 dynamic per-block quantization."""
    group_size = 128  # We only support 128 for now

    device = "cuda:0"
    torch.set_default_device(device)

    act_fn = SiluAndMul()
    linear = MockLinearFp8Dynamic128()

    scale = 1 / hidden_size
    x = torch.randn(*token_shape, hidden_size * 2, dtype=dtype, device=device) * scale
    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, QuantizedActivation)
    assert result.quant_key == kFp8Dynamic128Sym
    assert result.data.dtype == current_platform.fp8_dtype()
    assert result.orig_dtype == dtype
    assert result.orig_shape == (*token_shape, hidden_size)
    assert result.data.shape == result.orig_shape

    num_groups = hidden_size // group_size
    assert result.scale.shape == (*token_shape, num_groups)

    gate, up = x.split(hidden_size, dim=-1)
    silu_out = F.silu(gate) * up
    ref_out, ref_scales = per_token_group_quant_fp8(
        silu_out.reshape(-1, hidden_size), group_size=group_size, use_ue8m0=False
    )
    ref_out = ref_out.reshape(result.orig_shape)
    ref_scales = ref_scales.reshape(*token_shape, num_groups)

    torch.testing.assert_close(result.scale, ref_scales, rtol=1e-5, atol=1e-5)

    ref_deq = ref_out.to(torch.float32) * ref_scales.repeat_interleave(
        group_size, dim=-1
    )
    result_deq = result.data.to(torch.float32) * result.scale.repeat_interleave(
        group_size, dim=-1
    )
    torch.testing.assert_close(ref_deq, result_deq, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("hidden_size", [128, 512])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_maybe_fused_act_quant_fallback(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Test maybe_fused_act_quant falls back without an input quantization key."""
    device = "cuda:0"
    torch.set_default_device(device)

    act_fn = SiluAndMul()
    linear = MockLinearNoQuant()
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype, device=device)

    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, torch.Tensor)
    assert not isinstance(result, QuantizedActivation)
    assert result.dtype == dtype
    assert result.shape == (num_tokens, hidden_size)

    ref_out = act_fn(x)
    torch.testing.assert_close(result, ref_out)


@torch.inference_mode()
def test_maybe_fused_act_quant_preserves_required_unquantized_input(
    default_vllm_config,
) -> None:
    device = "cuda:0"
    act_fn = SiluAndMul()
    scale = torch.tensor([0.5], device=device, dtype=torch.float32)
    linear = MockLinearRequiresUnquantized(scale)
    x = torch.randn(17, 256, dtype=torch.bfloat16, device=device)

    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, act_fn(x))


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="Act + NVFP4 fusion requires SM100+",
)
@pytest.mark.parametrize("width", [16, 5376])
def test_relu2_nvfp4_oversized_dispatch_falls_back(
    native_vllm_config, monkeypatch, width
):
    """Reject unsafe index products with fake inputs, without huge allocations."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.model_executor.layers.fusion.act_nvfp4_quant import _act_nvfp4_max_rows
    from vllm.model_executor.layers.fusion.fused_act_quant import (
        _FUSED_ACT_QUANT,
        _relu_squared_nvfp4_quant_supported,
    )

    def fail_fused(*args, **kwargs):
        pytest.fail("Oversized input reached the int32-indexed producer")

    monkeypatch.setitem(
        _FUSED_ACT_QUANT, (ReLUSquaredActivation, kNvfp4Dynamic), fail_fused
    )
    act = ReLUSquaredActivation(compile_native=False)
    with FakeTensorMode():
        x = torch.empty(
            (_act_nvfp4_max_rows(width) + 1, width),
            dtype=torch.bfloat16,
            device="cuda",
        )
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        linear = MockLinearNvfp4(scale)
        assert _relu_squared_nvfp4_quant_supported(act, x[:-1], linear)
        actual = maybe_fused_act_quant(act, x, linear)
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == x.shape
        assert actual.dtype == x.dtype


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="Act + NVFP4 fusion requires SM100+",
)
@pytest.mark.parametrize("width", [16, 48, 80, 1344, 5376])
@torch.inference_mode()
def test_relu2_nvfp4_single_specialization_per_width(
    native_vllm_config, monkeypatch, width
):
    """One compile per K covers every M, so the profile run needs no warmup."""
    from vllm.triton_utils import triton

    act = ReLUSquaredActivation(compile_native=False)
    linear = MockLinearNvfp4(torch.full((), 17.125, device="cuda"))
    x = torch.randn((4096, width), dtype=torch.bfloat16, device="cuda")
    maybe_fused_act_quant(act, x, linear)

    def fail_compile(**kwargs):
        pytest.fail(f"Producer recompiled for a new row count: {kwargs.get('fn')}")

    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", fail_compile)
    for m in (1, 2, 7, 17, 33, 65, 127, 128, 129, 255, 256, 257, 1023, 4097):
        x = torch.randn((m, width), dtype=torch.bfloat16, device="cuda")
        actual = maybe_fused_act_quant(act, x, linear)
        assert isinstance(actual, QuantizedActivation)
        expected = ops.scaled_fp4_quant(
            act.forward_native(x), linear.input_global_scale_inv
        )
        assert torch.equal(actual.data, expected[0])
        assert torch.equal(
            actual.scale.view(torch.uint8), expected[1].view(torch.uint8)
        )
    opcheck(torch.ops.vllm.act_nvfp4_quant, (x, linear.input_global_scale_inv, "relu2"))


_BLACKWELL_MXFP8_ONLY = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="SiLU + MXFP8 fusion requires Blackwell",
)


def _assert_mxfp8_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _mxfp8_reference(act, x, rounding_mode):
    activated = act.forward_cuda(x) if rounding_mode == "cuda" else act(x)
    return mxfp8_e4m3_quantize(activated, is_sf_swizzled_layout=True)


@_BLACKWELL_MXFP8_ONLY
@pytest.mark.parametrize("rounding_mode", ["cuda", "compiled_native"])
@pytest.mark.parametrize("m", [1, 33, 128, 129])
@pytest.mark.parametrize("k", [160, 17408, 25600])
@torch.inference_mode()
def test_silu_mxfp8_bytes(native_vllm_config, m, k, rounding_mode):
    """Exercise padding and both actual activation rounding contracts."""
    generator = torch.Generator(device="cuda").manual_seed(m + k)
    x = torch.randn(
        (m, 2 * k), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    # Whole zero-scale blocks, large gates, and subnormal products.
    x[:, :32] = 0
    x[:, 32:64] *= 32
    x[:, k + 64 : k + 96] *= 2.0**-130
    x[:, k + 96 : k + 128] = -0.0
    before = x.clone()
    act = SiluAndMul()
    expected = _mxfp8_reference(act, x, rounding_mode)
    actual = torch.ops.vllm.silu_mul_mxfp8_quant(x, rounding_mode)
    repeated = torch.ops.vllm.silu_mul_mxfp8_quant(x, rounding_mode)
    for value, repeat, reference in zip(actual, repeated, expected):
        _assert_mxfp8_bytes(value, reference)
        _assert_mxfp8_bytes(repeat, reference)
        assert value.data_ptr() != repeat.data_ptr()
    _assert_mxfp8_bytes(x, before)


@_BLACKWELL_MXFP8_ONLY
@pytest.mark.parametrize("rounding_mode", ["cuda", "compiled_native"])
@torch.inference_mode()
def test_silu_mxfp8_changed_graph_replay(native_vllm_config, rounding_mode):
    x = torch.randn(33, 320, device="cuda", dtype=torch.bfloat16)
    act = SiluAndMul()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops.vllm.silu_mul_mxfp8_quant(x, rounding_mode)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, scale = torch.ops.vllm.silu_mul_mxfp8_quant(x, rounding_mode)
    pointers = output.data_ptr(), scale.data_ptr()
    for seed in [7, 31]:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x.copy_(
            torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        )
        before = x.clone()
        expected = _mxfp8_reference(act, x, rounding_mode)
        output.view(torch.uint8).fill_(255)
        scale.fill_(255)
        graph.replay()
        _assert_mxfp8_bytes(output, expected[0])
        _assert_mxfp8_bytes(scale, expected[1])
        _assert_mxfp8_bytes(x, before)
        assert pointers == (output.data_ptr(), scale.data_ptr())
    opcheck(torch.ops.vllm.silu_mul_mxfp8_quant, (x, rounding_mode))


@_BLACKWELL_MXFP8_ONLY
@pytest.mark.parametrize("rounding_mode", ["cuda", "compiled_native"])
@torch.inference_mode()
def test_silu_mxfp8_scale_thresholds(native_vllm_config, rounding_mode):
    """Compare FlashInfer at scale boundaries and adjacent reachable BF16 amax."""
    exponents = torch.arange(-129, 120, device="cuda", dtype=torch.int32)
    thresholds = torch.ldexp(
        torch.full_like(exponents, 448, dtype=torch.float32), exponents
    ).to(torch.bfloat16)
    targets = torch.stack(
        (
            torch.nextafter(thresholds, torch.zeros_like(thresholds)),
            thresholds,
            torch.nextafter(thresholds, torch.full_like(thresholds, float("inf"))),
        ),
        dim=1,
    ).flatten()
    k = 160
    x = torch.empty((targets.numel(), 2 * k), device="cuda", dtype=torch.bfloat16)
    # SiLU(32) rounds to exactly 32 in both paths; power-of-two scaling keeps
    # all three adjacent BF16 maxima exact through the activation boundary.
    x[:, :k] = 32
    x[:, k:] = targets[:, None] / 32
    act = SiluAndMul()
    activated = act.forward_cuda(x) if rounding_mode == "cuda" else act(x)
    assert torch.equal(activated, targets[:, None].expand_as(activated))
    expected = mxfp8_e4m3_quantize(activated, is_sf_swizzled_layout=True)
    actual = torch.ops.vllm.silu_mul_mxfp8_quant(x, rounding_mode)
    for value, reference in zip(actual, expected):
        _assert_mxfp8_bytes(value, reference)


@_BLACKWELL_MXFP8_ONLY
@pytest.mark.parametrize("rounding_mode", ["cuda", "compiled_native"])
@pytest.mark.parametrize("k", [160, 2**29])
@torch.inference_mode()
def test_silu_mxfp8_empty_rows(native_vllm_config, k, rounding_mode):
    """A reused custom-op graph can receive zero rows without a kernel launch."""
    x = torch.empty((0, 2 * k), device="cuda", dtype=torch.bfloat16)
    output, scale = torch.ops.vllm.silu_mul_mxfp8_quant(x, rounding_mode)
    assert output.shape == (0, k) and output.dtype == torch.float8_e4m3fn
    assert scale.shape == (0,) and scale.dtype == torch.uint8
    assert output.device == scale.device == x.device
    opcheck(torch.ops.vllm.silu_mul_mxfp8_quant, (x, rounding_mode))


@_BLACKWELL_MXFP8_ONLY
@pytest.mark.parametrize("rounding_mode", ["cuda", "compiled_native"])
@pytest.mark.parametrize("m", [129, 257])
@pytest.mark.parametrize("k", [128, 160])
@torch.inference_mode()
def test_silu_mxfp8_row_chunks(native_vllm_config, m, k, rounding_mode):
    """Exercise the real oversized-row path using small 128-row chunk bounds."""
    from vllm.model_executor.layers.fusion.silu_mul_mxfp8_quant import (
        _silu_mul_mxfp8_quant_impl,
    )

    generator = torch.Generator(device="cuda").manual_seed(m + k)
    x = torch.randn(
        (m, 2 * k), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    before = x.clone()
    expected = _mxfp8_reference(SiluAndMul(), x, rounding_mode)
    actual = _silu_mul_mxfp8_quant_impl(x, rounding_mode, max_rows=128)
    for value, reference in zip(actual, expected):
        _assert_mxfp8_bytes(value, reference)
    _assert_mxfp8_bytes(x, before)


def _mxfp8_quant_config(kind):
    if kind == "modelopt":
        return ModelOptMxFp8Config(True, None, [])
    if kind == "inc":
        return INCConfig(
            weight_bits=8,
            group_size=32,
            data_type="mx_fp",
            packing_format="auto_round:llm_compressor",
        )
    args = dict(
        num_bits=8,
        type="float",
        strategy="group",
        group_size=32,
        symmetric=True,
        scale_dtype="torch.uint8",
    )
    return compressed_tensors.CompressedTensorsConfig.from_config(
        {
            "format": "mxfp8-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": dict(args, dynamic=False),
                    "input_activations": dict(args, dynamic=True),
                }
            },
        }
    )


@_BLACKWELL_MXFP8_ONLY
@pytest.mark.parametrize("kind", ["modelopt", "inc", "compressed-tensors"])
@pytest.mark.parametrize(
    "activation", ["cuda", "native", "compiled_native", "default", "override"]
)
@torch.inference_mode()
def test_silu_mxfp8_real_consumer(dist_init, kind, activation):
    """Require actual fusion, including the default config and Dynamo wrapper."""
    config = VllmConfig(
        compilation_config=CompilationConfig()
        if activation == "default"
        else CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["all" if activation == "cuda" else "none"],
            inductor_compile_config={"max_autotune": True}
            if activation == "override"
            else {},
        ),
        kernel_config=KernelConfig(linear_backend="flashinfer_cutlass"),
    )
    with (
        torch.device("cuda"),
        set_default_torch_dtype(torch.bfloat16),
        set_current_vllm_config(config),
    ):
        linear = RowParallelLinear(
            160,
            128,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            disable_tp=True,
            quant_config=_mxfp8_quant_config(kind),
            prefix="test_silu.down_proj",
        )
        generator = torch.Generator(device="cuda").manual_seed(19)
        weight = torch.randint(-16, 17, linear.weight.shape, generator=generator)
        scale = torch.randint(120, 128, linear.weight_scale.shape, generator=generator)
        linear.weight.weight_loader(linear.weight, weight.to(linear.weight.dtype))
        linear.weight_scale.weight_loader(
            linear.weight_scale, scale.to(linear.weight_scale.dtype)
        )
        linear.quant_method.process_weights_after_loading(linear)
        assert get_input_quant_key(linear) == kMxfp8Dynamic
        compiled_native = activation in ("compiled_native", "default", "override")
        act = (
            SiluAndMul()
            if activation == "default"
            else SiluAndMul(compile_native=compiled_native)
        )
        x = torch.randn(33, 320, dtype=torch.bfloat16, generator=generator)
        activated = act(x)
        expected = linear(activated)[0]
        dispatch = maybe_fused_act_quant
        if compiled_native:
            dispatch = torch.compile(dispatch, fullgraph=True)
        # A renamed Dynamo wrapper attribute must fail this assertion in CI.
        actual = dispatch(act, x, linear)
        assert isinstance(actual, QuantizedActivation) == (
            activation in ("compiled_native", "default")
        )
        if isinstance(actual, QuantizedActivation):
            expected_q, expected_sf = _mxfp8_reference(act, x, "compiled_native")
            _assert_mxfp8_bytes(actual.data, expected_q)
            _assert_mxfp8_bytes(actual.scale, expected_sf)
        else:
            _assert_mxfp8_bytes(actual, activated)
        _assert_mxfp8_bytes(linear(actual)[0], expected)

        # Three-dimensional activations retain the existing unfused path.
        x3 = x.unsqueeze(0)
        fallback = dispatch(act, x3, linear)
        assert isinstance(fallback, torch.Tensor)
        _assert_mxfp8_bytes(fallback, act(x3))


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="Static FP8 CUTLASS consumer requires SM90 or newer",
)
@torch.inference_mode()
def test_qwen3_mxfp8_scope_preserves_static_fp8_activation(dist_init):
    """The new model call site must not enable the existing static-FP8 producer."""
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, custom_ops=["all"]
        ),
        kernel_config=KernelConfig(linear_backend="cutlass"),
    )
    with (
        torch.device("cuda"),
        set_default_torch_dtype(torch.bfloat16),
        set_current_vllm_config(config),
    ):
        mlp = Qwen3MLP(
            128,
            256,
            "silu",
            quant_config=ModelOptFp8Config("FP8", True, None, []),
            prefix="test_qwen3_mxfp8_scope",
        )
        for linear in (mlp.gate_up_proj, mlp.down_proj):
            for name, parameter in linear.named_parameters():
                values = torch.full(
                    parameter.shape,
                    0.125 if "scale" in name else 1.0,
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                if isinstance(parameter, PerTensorScaleParameter):
                    for shard_id in range(values.shape[0]):
                        parameter.load_merged_column_weight(
                            loaded_weight=values[shard_id], shard_id=shard_id
                        )
                else:
                    parameter.weight_loader(parameter, values)
            linear.quant_method.process_weights_after_loading(linear)
        assert get_input_quant_key(mlp.down_proj) == kFp8StaticTensorSym
        x = torch.arange(128, device="cuda", dtype=torch.bfloat16)[None] / 128
        gate_up, _ = mlp.gate_up_proj(x)
        expected = mlp.down_proj(mlp.act_fn(gate_up))[0]
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as control:
            generic = maybe_fused_act_quant(mlp.act_fn, gate_up, mlp.down_proj)
        assert isinstance(generic, QuantizedActivation)
        assert "_C::silu_and_mul_quant" in {e.name for e in control.events()}
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as observed:
            actual = mlp(x)
        assert "_C::silu_and_mul_quant" not in {e.name for e in observed.events()}
        _assert_mxfp8_bytes(actual, expected)
