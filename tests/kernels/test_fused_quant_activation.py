# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
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


GELU_FP8_SKIP = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="GELU per-token FP8 fusion requires CUDA SM89+",
)


def gelu_per_token_ref(
    x: torch.Tensor, scale_ub: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    activation = GeluAndMul(approximate="tanh").forward_cuda(x)
    return ops.scaled_fp8_quant(
        activation.reshape(-1, activation.shape[-1]),
        scale_ub=scale_ub,
        use_per_token_if_dynamic=True,
    )


@GELU_FP8_SKIP
# 781: scalar path; 6912 and 21504: cached rows (256- and 512-thread blocks for
# 300 tokens); 36864: row too wide to cache, recomputed in the second pass.
# 17 tokens use 1024-thread blocks.
@pytest.mark.parametrize("d", [781, 6912, 21504, 36864])
@pytest.mark.parametrize("token_shape", [(17,), (300,), (2, 17)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_ub_value", [None, 0.25])
@torch.inference_mode()
def test_gelu_tanh_and_mul_per_token_quant(
    default_vllm_config,
    d: int,
    token_shape: tuple[int, ...],
    dtype: torch.dtype,
    scale_ub_value: float | None,
) -> None:
    """Bit-identical to the CUDA GELU-tanh activation + per-token quantizer."""
    set_random_seed(0)
    x = torch.randn((*token_shape, 2 * d), dtype=dtype, device="cuda")
    x[0] = 0  # all-zero row: minimum scale
    x[1] *= 20  # far above the FP8 range, still finite in FP16
    scale_ub = (
        None
        if scale_ub_value is None
        else torch.tensor([scale_ub_value], dtype=torch.float32, device="cuda")
    )
    expected, expected_scale = gelu_per_token_ref(x, scale_ub)
    actual, scale = ops.gelu_tanh_and_mul_per_token_quant(x, scale_ub)

    assert actual.shape == (*token_shape, d)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.reshape_as(expected).view(torch.uint8),
        expected.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    opcheck(
        torch.ops._C.gelu_tanh_and_mul_per_token_quant,
        (actual, x, scale, scale_ub),
        cond=d == 6912,
    )


@GELU_FP8_SKIP
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_gelu_tanh_and_mul_per_token_quant_unaligned_and_empty(
    default_vllm_config, dtype: torch.dtype
) -> None:
    set_random_seed(0)
    d = 6912
    # A contiguous view that starts one element into its storage cannot use
    # 16-byte loads and takes the scalar path. The CUDA activation used as the
    # reference assumes aligned rows, so it gets an aligned copy.
    x = torch.randn(17 * 2 * d + 1, dtype=dtype, device="cuda")[1:].view(17, 2 * d)
    expected, expected_scale = gelu_per_token_ref(x.clone(), None)
    actual, scale = ops.gelu_tanh_and_mul_per_token_quant(x, None)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )

    empty = torch.empty((0, 2 * d), dtype=dtype, device="cuda")
    actual, scale = ops.gelu_tanh_and_mul_per_token_quant(empty, None)
    assert actual.shape == (0, d)
    assert scale.shape == (0, 1)


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


class MockLinearRequiresUnquantized(MockLinearFp8Static):
    """Mock consumer that needs the original activation for another branch."""

    requires_unquantized_input = True


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
