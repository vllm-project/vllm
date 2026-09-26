# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BitNet b1.58-style ternary ({-1, 0, +1}) weight-only quantization.

This is a first-stage, pure-PyTorch, CPU-friendly implementation, added to
give vLLM a starting point for serving BitNet b1.58 / "1-bit LLM" style
checkpoints (see e.g. microsoft/bitnet-b1.58-2B-4T, tiiuae Falcon-E via the
onebitllms library, and Tencent AngelSlim's Tequila/Sherry ternary PTQ),
which several users have asked for (vllm-project/vllm#33142, #18213,
#17279) but which vLLM does not yet support.

Quantization rule (absmean ternary), following Ma et al., "The Era of
1-bit LLMs: All Large Language Models are in 1.58 Bits"
(arXiv:2402.17764):

    gamma = mean(|W|)
    W_ternary = round(clip(W / (gamma + eps), -1, 1))  in {-1, 0, +1}

Ternary values are packed 4-per-byte (2 bits each) as a plain uint8
tensor.

Current scope (deliberately limited -- see follow-up items below):
  * Weight-only quantization of vLLM's LinearBase-derived linear layers.
  * Online quantization at load time: create_weights allocates an
    ordinary full-precision weight (so the existing weight-loading and any
    tensor-parallel sharding machinery is unaffected), and
    process_weights_after_loading quantizes+packs it in place.
  * A correctness-first "unpack, then F.linear" compute path in apply().
    This intentionally is not the fastest possible implementation; it
    trades throughput for being simple, easy to verify, and runnable on
    CPU (e.g. for development on machines without a supported GPU).

Explicitly NOT yet implemented (tracked follow-up work):
  * A fused kernel (Triton, then CUDA) that computes the matmul directly
    against packed ternary weights via add/subtract instead of multiply,
    for real throughput/memory gains at serving time.
  * Reading an already-packed ternary checkpoint format (e.g. the packed
    layout used by microsoft/BitNet's own GPU kernels) directly, instead
    of quantizing a full-precision checkpoint online.
  * Tensor-parallel-aware packing/unpacking edge cases beyond what the
    default full-precision weight loader already handles before this
    method's process_weights_after_loading step runs.
  * End-to-end validation against a real checkpoint's reference outputs.

Given the above, this should be treated as a starting point for review and
iteration, not a finished, benchmarked backend.

Usage note: because this method is registered via
register_quantization_config rather than being wired into this package's
built-in method list, the module must actually be imported once (e.g.
"import vllm.model_executor.layers.quantization.bitnet") before
"--quantization bitnet" (or quantization="bitnet") will resolve.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

BITNET_VALUES_PER_BYTE = 4  # 2 bits per ternary value: {-1, 0, +1} -> {0, 1, 2}


def bitnet_quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmean ternary quantization of a 2-D weight matrix.

    Args:
        weight: a 2-D float tensor of shape (out_features, in_features).

    Returns:
        (packed, scale): packed is a uint8 tensor of shape
        (out_features, ceil(in_features / 4)) holding four 2-bit ternary
        codes per byte; scale is a scalar float32 tensor holding gamma
        (the per-tensor absmean scale).
    """
    if weight.dim() != 2:
        raise ValueError(
            f"bitnet quantization expects a 2-D weight matrix, got shape "
            f"{tuple(weight.shape)}"
        )

    weight = weight.float()
    eps = 1e-5
    gamma = weight.abs().mean()
    ternary = (weight / (gamma + eps)).round().clamp_(-1, 1).to(torch.int8)

    out_features, in_features = ternary.shape
    pad = (-in_features) % BITNET_VALUES_PER_BYTE
    if pad:
        ternary = F.pad(ternary, (0, pad), value=0)

    # Map {-1, 0, 1} -> {0, 1, 2} (fits in 2 bits) before packing.
    codes = (ternary + 1).to(torch.uint8)
    codes = codes.view(out_features, -1, BITNET_VALUES_PER_BYTE)

    packed = torch.zeros(
        codes.shape[0], codes.shape[1], dtype=torch.uint8, device=weight.device
    )
    for i in range(BITNET_VALUES_PER_BYTE):
        packed |= codes[..., i] << (2 * i)

    return packed, gamma.to(torch.float32)


def bitnet_dequantize_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Inverse of bitnet_quantize_weight."""
    num_groups = packed.shape[1]
    codes = torch.empty(
        packed.shape[0],
        num_groups * BITNET_VALUES_PER_BYTE,
        dtype=torch.uint8,
        device=packed.device,
    )
    for i in range(BITNET_VALUES_PER_BYTE):
        codes[:, i::BITNET_VALUES_PER_BYTE] = (packed >> (2 * i)) & 0b11

    ternary = codes[:, :in_features].to(torch.float32) - 1.0
    return ternary * scale


def bitnet_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference ("unpack, then matmul") BitNet linear forward pass.

    NOTE: this materializes the full-precision weight on every call. A
    fused kernel that never materializes it is the planned follow-up (see
    module docstring); this path exists to have a simple, obviously
    correct reference to test a future fused kernel against.
    """
    weight = bitnet_dequantize_weight(packed_weight, scale, out_features, in_features)
    return F.linear(x.to(weight.dtype), weight, bias)


class BitNetLinearMethod(QuantizeMethodBase):
    """Weight-only ternary ({-1, 0, +1}) quantization for linear layers."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        # Stage 1: keep a full-precision weight around for loading (so the
        # existing weight_loader / TP-sharding path is untouched), and only
        # quantize+pack it afterwards, in process_weights_after_loading.
        output_size_per_partition = sum(output_partition_sizes)
        weight = torch.nn.Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        layer.bitnet_out_features = output_size_per_partition
        layer.bitnet_in_features = input_size_per_partition
        layer.bitnet_quantized = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "bitnet_quantized", False):
            return

        packed, scale = bitnet_quantize_weight(layer.weight.data)
        del layer.weight
        layer.register_buffer("bitnet_packed_weight", packed, persistent=True)
        layer.register_buffer("bitnet_scale", scale, persistent=True)
        layer.bitnet_quantized = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return bitnet_linear(
            x,
            layer.bitnet_packed_weight,
            layer.bitnet_scale,
            layer.bitnet_out_features,
            layer.bitnet_in_features,
            bias,
        )


@register_quantization_config("bitnet")
class BitNetConfig(QuantizationConfig):
    """Config for BitNet b1.58-style ternary weight-only quantization.

    Registered as an out-of-tree quantization method via
    register_quantization_config (see
    vllm/model_executor/layers/quantization/__init__.py) rather than being
    wired into that file's built-in method list directly, so this can be
    reviewed and iterated on independently. See the module docstring for
    current scope and known limitations.
    """

    def get_name(self) -> str:
        return "bitnet"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # Pure-PyTorch reference path: no GPU-capability floor.
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitNetConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return BitNetLinearMethod()
        return None
