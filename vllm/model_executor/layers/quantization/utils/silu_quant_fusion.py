import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.quant_fusion import (
    QuantizedActivation,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)


def _unwrap_linear_for_quant_fusion(linear: nn.Module) -> nn.Module:
    while hasattr(linear, "base_layer") and getattr(linear, "base_layer") is not None:
        linear = linear.base_layer
    return linear


def _plain_silu_mul(
    gate_up: torch.Tensor, activation_clamp: float | None = None
) -> torch.Tensor:
    hidden_size = gate_up.shape[-1] // 2
    out = torch.empty(
        (*gate_up.shape[:-1], hidden_size),
        device=gate_up.device,
        dtype=gate_up.dtype,
    )
    if activation_clamp is None:
        torch.ops._C.silu_and_mul(out, gate_up)
    else:
        torch.ops._C.silu_and_mul_with_clamp(out, gate_up, float(activation_clamp))
    return out


def silu_mul_input_quant(
    gate_up: torch.Tensor,
    linear: nn.Module,
    activation_clamp: float | None = None,
) -> torch.Tensor | QuantizedActivation:
    """Apply SiLU-and-mul and optionally pre-quantize the result for ``linear``.

    Today this helper only materializes the static-per-tensor FP8 path. All
    other quantization schemes fall back to the plain activation tensor.
    """

    linear = _unwrap_linear_for_quant_fusion(linear)
    input_quant_key = getattr(linear, "input_quant_key", None)
    if input_quant_key != kFp8StaticTensorSym:
        return _plain_silu_mul(gate_up, activation_clamp)

    input_scale = getattr(linear, "input_scale", None)
    if input_scale is None:
        return _plain_silu_mul(gate_up, activation_clamp)

    hidden_size = gate_up.shape[-1] // 2
    q = torch.empty(
        (*gate_up.shape[:-1], hidden_size),
        device=gate_up.device,
        dtype=input_quant_key.dtype,
    )
    torch.ops._C.silu_and_mul_quant(q, gate_up, input_scale)
    return QuantizedActivation(q=q, scale=input_scale, quant_key=input_quant_key)
