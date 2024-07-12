from enum import Enum
from typing import Optional, Tuple

import pytest
import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.layernorm import RMSNorm


class QuantType(Enum):
    DynamicPerTokenInt8 = 0


DTYPES = [torch.bfloat16, torch.float]
QUANT_TYPES = [QuantType.DynamicPerTokenInt8]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192,
                8199]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

EPS = 1e-6

## Helpers


def out_dtype_from_quant_type(quant_type: QuantType) -> torch.dtype:
    if quant_type == QuantType.DynamicPerTokenInt8:
        return torch.int8


def needs_quant_scale(quant_type: QuantType) -> bool:
    return False


def scaled_int8_quant(x: torch.Tensor, scales: torch.Tensor):
    int8_traits = torch.iinfo(torch.int8)
    x = (x / scales).round().clamp(int8_traits.min,
                                   int8_traits.max).to(torch.int8)
    return x


def scaled_fp8_quant(x: torch.Tensor, scales: torch.Tensor):
    fp8_traits = torch.finfo(torch.float8_e4m3fn)
    one = torch.as_tensor(1.0, dtype=torch.float32, device='cuda')
    iscales = one / scales
    return (x.to(dtype=torch.float32) * iscales).clamp(
        min=fp8_traits.min, max=fp8_traits.max).to(dtype=torch.float8_e4m3fn)

def ref_rms_norm(rms_norm_layer: RMSNorm,
                 x: torch.Tensor,
                 residual: Optional[torch.Tensor]) \
                         -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    if residual is not None:
        residual = residual.clone()
        out, residual = rms_norm_layer.forward_native(x, residual)
    else:
        out = rms_norm_layer.forward_native(x)

    return out, residual

def ref_dynamic_per_token_int8_quant(rms_norm_layer: RMSNorm,
                            x: torch.Tensor,
                            residual: Optional[torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    # Norm
    torch_out, residual = ref_rms_norm(rms_norm_layer, x, residual)

    # Compute scales
    torch_out_token_max, _ = torch_out.abs().max(dim=1)
    torch_out_token_max = torch_out_token_max.to(dtype=torch.float32)
    scales = (torch_out_token_max / float(127.0))[:,
                                                  None].to(device="cuda",
                                                           dtype=torch.float32)
    # Quant
    torch_out = scaled_int8_quant(torch_out, scales)

    return torch_out, scales, residual

def ref_impl(rms_norm_layer: RMSNorm,
             x: torch.Tensor,
             residual: Optional[torch.Tensor],
             scales: Optional[torch.Tensor],
             quant_type: QuantType) \
           -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    assert quant_type == QuantType.DynamicPerTokenInt8
    assert scales is None
    return ref_dynamic_per_token_int8_quant(rms_norm_layer, x, residual)

def ops_impl(weight: torch.Tensor,
             x: torch.Tensor,
             residual: Optional[torch.Tensor],
             scales: Optional[torch.Tensor],
             quant_type: QuantType) \
                -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    assert quant_type == QuantType.DynamicPerTokenInt8
    num_scales = int(x.numel() / x.shape[-1])

    assert scales is None
    out_dtype = torch.int8

    out = torch.empty_like(x, dtype=out_dtype, device="cuda")
    scales = torch.empty((num_scales, 1), dtype=torch.float32, device="cuda")
    tmp = torch.empty_like(x, dtype=torch.float, device="cuda")

    if residual is not None:
        residual = residual.clone()
        ops.residual_add_rms_norm_dynamic_per_token_int8_quant(
            out, tmp, x, residual, weight, scales, 1e-6)
    else:
        ops.rms_norm_dynamic_per_token_int8_quant(out, tmp, x, weight, scales,
                                                  1e-6)
    return out, scales, residual


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", [QuantType.DynamicPerTokenInt8])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    quant_type: QuantType,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    layer = RMSNorm(hidden_size, EPS).to(dtype=dtype)

    # Make weights
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Make inputs and residual
    scale = 1 / (hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    #orig_x = x.clone()
    residual = torch.randn_like(x) * scale if add_residual else None

    # Make quant scale
    quant_scale = torch.rand(1).to(device="cuda", dtype=torch.float32) \
            if needs_quant_scale(quant_type) else None

    #torch.set_printoptions(profile="full")
    #print (f"x : {x}")
    #print (f"weight : {layer.weight}")
    #if quant_scale is not None:
    #    print (f"quant scale : {quant_scale}")
    #if residual is not None:
    #    print (f"residual : {residual}")

    # Determine out_dtype for sanity checks later.
    out_dtype = out_dtype_from_quant_type(quant_type)

    ref_out, ref_scales, ref_residual = \
            ref_impl(layer, x, residual, quant_scale, quant_type)
    ops_out, ops_scales, ops_residual = \
            ops_impl(layer.weight, x, residual, quant_scale, quant_type)

    #print (f"ref residual {ref_residual[5]}")
    #print (f"ops residual {ops_residual[5]}")
    #print (f"ref scales {ref_scales[0]}")
    #print (f"ops scales {ops_scales[0]}")
    #print (f"ref out {ref_out[5]}")
    #print (f"ops out {ops_out[5]}")
    #print (f"orig_x {orig_x[5]}")
    #print (f"weight {layer.weight[5]}")

    assert ref_out.dtype == out_dtype
    assert ops_out.dtype == out_dtype
    if out_dtype == torch.int8:
        # big atol to account,
        # 1. rms-norm errors due to order of reduction.
        # 2. quant round-off errors.
        assert torch.allclose(ref_out, ops_out, atol=2)
    else:
        assert torch.allclose(ref_out.to(dtype=torch.float32),
                              ops_out.to(dtype=torch.float32))

    assert torch.allclose(ref_scales, ops_scales, atol=1e-2, rtol=1e-2)
    if add_residual:
        assert torch.allclose(ref_residual, ops_residual, atol=1e-2, rtol=1e-2)
