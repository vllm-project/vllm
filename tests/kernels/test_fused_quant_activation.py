# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.activation import SiluAndMul
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


# ---------------------------------------------------------------------------
# GELU (tanh approximation) x FP8 static quant fusion
# ---------------------------------------------------------------------------


def ref_impl_gelu_tanh(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference: tanh-GELU -> gated multiply -> static fp8 quant."""
    d = x.shape[-1] // 2
    act = torch.nn.functional.gelu(x[..., :d], approximate="tanh") * x[..., d:]
    return ops.scaled_fp8_quant(act, scale)[0]


def ops_impl_gelu_tanh(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_shape = (x.shape[0], x.shape[1] // 2)
    out = torch.empty(out_shape, dtype=current_platform.fp8_dtype(), device=x.device)
    torch.ops._C.gelu_tanh_and_mul_quant(out, x, scale)
    return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_gelu_tanh_and_mul_quant(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    scale = torch.randn((1), device=device, dtype=torch.float32)
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype)

    ref_out = ref_impl_gelu_tanh(x, scale)
    fused_out = ops_impl_gelu_tanh(x, scale)

    assert fused_out.dtype == current_platform.fp8_dtype()
    assert ref_out.shape == fused_out.shape
    assert torch.equal(ref_out.view(torch.uint8), fused_out.view(torch.uint8)), (
        "fused kernel must be bit-identical to tanh-GELU -> mul -> quant"
    )
    opcheck(torch.ops._C.gelu_tanh_and_mul_quant, (fused_out, x, scale))


@torch.inference_mode()
def test_gelu_erf_does_not_fuse(default_vllm_config) -> None:
    """GeluAndMul(approximate="none") must fall back to the unfused path."""
    from vllm.model_executor.layers.activation import GeluAndMul
    from vllm.model_executor.layers.fusion.fused_act_quant import (
        maybe_fused_act_quant,
    )

    layer = GeluAndMul(approximate="none").cuda()
    linear = MockLinearFp8Static(torch.ones(1, dtype=torch.float32, device="cuda"))
    x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")

    out = maybe_fused_act_quant(layer, x, linear)
    assert isinstance(out, torch.Tensor), (
        "erf variant must not use the tanh-based fused kernel"
    )


@pytest.mark.parametrize("token_shape", [(1,), (16,), (128,), (1, 17), (2, 17)])
@pytest.mark.parametrize("hidden_size", [128, 512, 1024])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_maybe_fused_act_quant_gelu_tanh(
    default_vllm_config,
    token_shape: tuple[int, ...],
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """GeluAndMul(tanh) must fuse through maybe_fused_act_quant."""
    from vllm.model_executor.layers.activation import GeluAndMul

    device = "cuda:0"
    torch.set_default_device(device)

    act_fn = GeluAndMul(approximate="tanh")
    scale = torch.tensor([0.5], device=device, dtype=torch.float32)
    linear = MockLinearFp8Static(scale)

    x = torch.randn(*token_shape, hidden_size * 2, dtype=dtype, device=device)
    result = maybe_fused_act_quant(act_fn, x, linear)

    assert isinstance(result, QuantizedActivation)
    assert result.quant_key == kFp8StaticTensorSym
    assert result.data.dtype == current_platform.fp8_dtype()
    assert result.orig_dtype == dtype
    assert result.orig_shape == (*token_shape, hidden_size)

    ref_out = ref_impl(act_fn, x.reshape(-1, hidden_size * 2), scale).reshape(
        result.orig_shape
    )
    assert torch.equal(result.data.view(torch.uint8), ref_out.view(torch.uint8)), (
        "fused registry path must be bit-identical"
    )


# ---------------------------------------------------------------------------
# GELU (tanh) x FP8 static quant: consistency with the unfused path and edges
# ---------------------------------------------------------------------------

# Odd / tiny / non-vector-width hidden sizes exercise the scalar cleanup loop.
ODD_HIDDEN_SIZES = [1, 5, 7, 9, 1001, 4095]


def _unfused_gelu_tanh_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The path this fusion replaces: gelu_tanh_and_mul -> scaled_fp8_quant."""
    d = x.shape[-1] // 2
    tmp = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_tanh_and_mul(tmp, x)
    flat = tmp.reshape(-1, d)
    out, _ = ops.scaled_fp8_quant(flat, scale)
    return out.reshape(tmp.shape)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_gelu_tanh_and_mul_quant_matches_unfused(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    """The fused op must be bit-identical to the ops it replaces.

    `test_gelu_tanh_and_mul_quant` pins numerics against PyTorch's `F.gelu`;
    this one uses the actual unfused vLLM path as the reference -- the ops a
    model runs today -- so a drift between the fused and unfused paths shows up
    here even if PyTorch's formula happens to agree.
    """
    set_random_seed(seed)
    device = "cuda:0"
    torch.set_default_device(device)

    x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype)
    scale = torch.randn((1), dtype=torch.float32)

    ref_out = _unfused_gelu_tanh_quant(x, scale)
    fused_out = ops_impl_gelu_tanh(x, scale)

    assert torch.equal(ref_out.view(torch.uint8), fused_out.view(torch.uint8)), (
        "fused kernel must match gelu_tanh_and_mul + scaled_fp8_quant"
    )


@pytest.mark.parametrize("hidden_size", ODD_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_gelu_tanh_and_mul_quant_odd_hidden_sizes(
    default_vllm_config,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Odd and sub-vector-width hidden sizes must match the unfused path."""
    device = "cuda:0"
    torch.set_default_device(device)

    x = torch.randn(17, 2 * hidden_size, dtype=dtype)
    scale = torch.randn((1), dtype=torch.float32)

    ref_out = _unfused_gelu_tanh_quant(x, scale)
    fused_out = ops_impl_gelu_tanh(x, scale)

    assert fused_out.shape == (17, hidden_size)
    assert torch.equal(ref_out.view(torch.uint8), fused_out.view(torch.uint8))


@pytest.mark.parametrize("token_shape", [(1,), (2, 8), (3, 4, 2)])
@pytest.mark.parametrize("hidden_size", [128, 512])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_gelu_tanh_and_mul_quant_ndim(
    default_vllm_config,
    token_shape: tuple[int, ...],
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """The op accepts [..., d] input, not just 2D."""
    device = "cuda:0"
    torch.set_default_device(device)

    x = torch.randn(*token_shape, 2 * hidden_size, dtype=dtype)
    scale = torch.randn((1), dtype=torch.float32)

    out = torch.empty(
        *token_shape,
        hidden_size,
        dtype=current_platform.fp8_dtype(),
        device=device,
    )
    torch.ops._C.gelu_tanh_and_mul_quant(out, x, scale)

    ref_out = _unfused_gelu_tanh_quant(x, scale)
    assert torch.equal(ref_out.view(torch.uint8), out.view(torch.uint8)), (
        "3D input must be handled like the unfused path"
    )
