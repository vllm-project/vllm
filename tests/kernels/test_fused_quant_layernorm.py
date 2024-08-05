from enum import Enum
from typing import Optional, Tuple, Union

import pytest
import torch

import tests.kernels.quant_utils as quant_utils
import vllm._custom_ops as ops
from vllm.model_executor.layers.layernorm import RMSNorm


class QuantType(Enum):
    SymmetricInt8DynamicPerTokenQuant = 0
    SymmetricFP8DynamicPerTokenQuant = 1
    ASymmetricInt8DynamicPerTokenQuant = 2


def is_int8_quant(qtype: QuantType) -> bool:
    return qtype in [
        QuantType.SymmetricInt8DynamicPerTokenQuant,
        QuantType.ASymmetricInt8DynamicPerTokenQuant
    ]


def is_fp8_quant(qtype: QuantType) -> bool:
    return qtype in [QuantType.SymmetricFP8DynamicPerTokenQuant]


def is_symmetric_quant(qtype: QuantType) -> bool:
    return qtype in [
        QuantType.SymmetricInt8DynamicPerTokenQuant,
        QuantType.SymmetricFP8DynamicPerTokenQuant
    ]


def is_asymmetric_quant(qtype: QuantType) -> bool:
    return qtype in [QuantType.ASymmetricInt8DynamicPerTokenQuant]


def quant_dtype_from_quant_type(qtype: QuantType) -> torch.dtype:
    if is_int8_quant(qtype):
        return torch.int8
    if is_fp8_quant(qtype):
        return torch.float8_e4m3fn
    raise ValueError(f"Unsupported quant type {qtype}")


DTYPES = [torch.bfloat16, torch.float]
QUANT_TYPES = [
    QuantType.SymmetricFP8DynamicPerTokenQuant,
    QuantType.SymmetricInt8DynamicPerTokenQuant,
    QuantType.ASymmetricInt8DynamicPerTokenQuant
]
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [67, 768, 2048, 5120, 5137,
                8192]  # Arbitrary values for testing
HIDDEN_SIZES += list(range(1024, 1033))  # vectorized conversion edge cases
ADD_RESIDUAL = [True, False]  # With and without fused residual add
SCALE_UBS = [True, False]  # With and without scale_ub
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

EPS = 1e-6

## Helpers


def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')

def ref_rms_norm(rms_norm_layer: RMSNorm,
                 x: torch.Tensor,
                 residual: Optional[torch.Tensor]) \
                         -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # We try to match the unfused CUDA implementation, in the fused
    # implementation wherever possible. It is important to use `forward_cuda`
    # instead of `forward_native` so we eliminate the source of discrepancy
    # and be less-prone to bogus floating-point rounding errors.
    if residual is not None:
        # With residual the cuda version overwrites the input.
        out, residual = rms_norm_layer.forward_cuda(x.clone(),
                                                    residual.clone())
    else:
        out = rms_norm_layer.forward_cuda(x)

    return out, residual

def ref_symmetric_dynamic_per_token_quant(rms_norm_layer: RMSNorm,
                            x: torch.Tensor,
                            quant_dtype: torch.dtype,
                            residual: Optional[torch.Tensor],
                            scale_ub: Optional[torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    assert scale_ub is None or quant_dtype == torch.float8_e4m3fn

    # Norm
    torch_out, residual = ref_rms_norm(rms_norm_layer, x, residual)

    # Quant
    if quant_dtype == torch.float8_e4m3fn:
        torch_out, scales = ops.scaled_fp8_quant(torch_out,
                                                 scale_ub=scale_ub,
                                                 use_per_token_if_dynamic=True)
    else:
        assert quant_dtype == torch.int8
        torch_out, scales = ops.scaled_int8_quant(torch_out)

    return torch_out, scales, residual

def ref_asymmetric_dynamic_per_token_quant(rms_norm_layer: RMSNorm,
                            x: torch.Tensor,
                            quant_dtype: torch.dtype,
                            residual: Optional[torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                        Optional[torch.Tensor]]:
    # Only support int8 for now
    assert quant_dtype == torch.int8

    # Norm
    torch_out, residual = ref_rms_norm(rms_norm_layer, x, residual)

    # Quant
    # TODO Switch to GPU reference when it becomes available.
    torch_out, scales, azps = \
            quant_utils.ref_asymmetric_dynamic_per_token_quant(
                    torch_out, quant_dtype)

    return torch_out, scales, azps, residual

def ref_impl(rms_norm_layer: RMSNorm,
             x: torch.Tensor,
             quant_type: QuantType,
             residual: Optional[torch.Tensor],
             scale_ub: Optional[torch.Tensor]) \
           -> Tuple[torch.Tensor, torch.Tensor,
                   Optional[torch.Tensor], torch.Tensor]:

    quant_dtype = quant_dtype_from_quant_type(quant_type)

    out, scales, azps = (None, None, None)

    if is_symmetric_quant(quant_type):
        out, scales, residual = \
                ref_symmetric_dynamic_per_token_quant(rms_norm_layer, x,
                        quant_dtype, residual, scale_ub)
    else:
        assert is_asymmetric_quant(quant_type)
        out, scales, azps, residual = ref_asymmetric_dynamic_per_token_quant(
            rms_norm_layer, x, quant_dtype, residual)

    return out, scales, azps, residual

def ops_dynamic_per_token_quant(weight: torch.Tensor,
                            x: torch.Tensor,
                            quant_type: QuantType,
                            residual: Optional[torch.Tensor],
                            scale_ub: Optional[torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor,
                        Optional[torch.Tensor],
                        Optional[torch.Tensor]]:
    if residual is not None:
        residual = residual.clone()

    out, scales, azps = ops.rms_norm_dynamic_per_token_quant(
        x, weight, EPS, quant_dtype_from_quant_type(quant_type), scale_ub,
        residual, is_asymmetric_quant(quant_type))
    return out, scales, azps, residual

def ops_impl(weight: torch.Tensor,
             x: torch.Tensor,
             quant_type: QuantType,
             residual: Optional[torch.Tensor],
             scale_ub: Optional[torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor,
                        Optional[torch.Tensor], Optional[torch.Tensor]]:
    return ops_dynamic_per_token_quant(weight, x, quant_type, residual,
                                       scale_ub)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: Union[bool, torch.Tensor],
    scale_ub: Union[bool, torch.Tensor],
    dtype: torch.dtype,
    quant_type: QuantType,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    quant_dtype = quant_dtype_from_quant_type(quant_type)

    if scale_ub and quant_dtype != torch.float8_e4m3fn:
        # skip
        return

    layer = RMSNorm(hidden_size, EPS).to(dtype=dtype)

    # Make weights
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Make inputs
    x, residual = (None, None)
    using_vectorized_ops_impl = hidden_size % 4 == 0
    if using_vectorized_ops_impl or add_residual:

        # When using vectorized ops or a residual, the order of float-additions
        # in computing the RMS is different from the reference implementation.
        #  - When using_vectorized_ops_impl : the ops implementation uses
        #    vectorized datatypes.
        #  - When add_residual : the reference rms norm implementation uses
        #    vectorized datatypes.
        # Solution : Make the input and residual less sensitive to order of
        # float-additions in these cases.
        scale = as_float32_tensor(0.25)
        x = as_float32_tensor(
            torch.randint(size=(num_tokens, hidden_size), low=-5,
                          high=5)) * scale
        residual = as_float32_tensor(
            torch.randint(size=(num_tokens, hidden_size), low=-5,
                          high=5)) * scale
        x = x.to(dtype=dtype)
        residual = residual.to(dtype=dtype)
    else:
        scale = 1 / hidden_size
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = None

    if scale_ub:
        rms_x, _ = ref_rms_norm(layer, x, residual)
        scale_ub = torch.mean(rms_x).to(dtype=torch.float32, device='cuda')
    else:
        scale_ub = None

    ref_out, ref_scales, ref_azp, ref_residual = \
            ref_impl(layer, x, quant_type, residual, scale_ub)

    ops_out, ops_scales, ops_azp, ops_residual = \
            ops_impl(layer.weight, x, quant_type, residual, scale_ub)

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    # Compare scales
    assert torch.allclose(ref_scales, ops_scales)
    # Compare azps
    if is_asymmetric_quant(quant_type):
        torch.allclose(ref_azp, ops_azp)
    # Compare residual
    if add_residual:
        assert torch.allclose(ref_residual, ops_residual)
    # Compare outputs
    if quant_dtype == torch.int8:
        # big atol to account for round-off errors.
        assert torch.allclose(ref_out, ops_out, atol=1)
    else:
        assert torch.allclose(ref_out.to(dtype=torch.float32),
                              ops_out.to(dtype=torch.float32))
