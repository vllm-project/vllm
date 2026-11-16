# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import ReLUSquaredActivation, SiluAndMul
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

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
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize("shape", [(1, 16), (17, 80), (129, 1344), (256, 5376)])
@pytest.mark.parametrize("scale_value", [0.0009765625, 17.125])
@torch.inference_mode()
def test_relu2_nvfp4_matches_unfused(native_vllm_config, shape, scale_value):
    """Match unfused ReLU2 + NVFP4 bitwise and zero every padded scale."""
    from tests.kernels.quantization.test_nvfp4_quant import recover_swizzled_scales

    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(scale_value, dtype=torch.float32, device=x.device)
    linear = MockLinearNvfp4(scale)
    act = ReLUSquaredActivation(compile_native=False)
    compiled_relu = torch.compile(act.forward_native, fullgraph=True)
    expected = ops.scaled_fp4_quant(compiled_relu(x), scale)
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
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
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
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
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
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
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
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
)
@pytest.mark.parametrize("width", [16, 5376])
def test_relu2_nvfp4_oversized_dispatch_falls_back(
    native_vllm_config, monkeypatch, width
):
    """Reject unsafe index products with fake inputs, without huge allocations."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.model_executor.layers.fusion.fused_act_quant import (
        _FUSED_ACT_QUANT,
        _relu_squared_nvfp4_quant_supported,
    )
    from vllm.model_executor.layers.fusion.relu2_nvfp4_quant import (
        _relu_squared_nvfp4_max_rows,
    )

    def fail_fused(*args, **kwargs):
        pytest.fail("Oversized input reached the int32-indexed producer")

    monkeypatch.setitem(
        _FUSED_ACT_QUANT, (ReLUSquaredActivation, kNvfp4Dynamic), fail_fused
    )
    act = ReLUSquaredActivation(compile_native=False)
    with FakeTensorMode():
        x = torch.empty(
            (_relu_squared_nvfp4_max_rows(width) + 1, width),
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
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="ReLU2 NVFP4 fusion is validated on SM10x",
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
    opcheck(torch.ops.vllm.relu_squared_nvfp4_quant, (x, linear.input_global_scale_inv))
