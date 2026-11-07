# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``w4afp8`` checkpoints: INT4 group-128 routed experts with FP8 activations,
FP8 block-128 everywhere else.

Routed experts are remapped at load time onto the compressed-tensors W4A8 FP8
MoE method; dense layers already use the native FP8 serialization.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from compressed_tensors.quantization import QuantizationArgs

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8_fp8 import (  # noqa: E501
    CompressedTensorsW4A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.models.utils import WeightsMapper

_EXPERT = r"(\.experts\.\d+\.(?:gate_proj|up_proj|down_proj|w1|w2|w3))"
_EXPERT_WEIGHT = re.compile(_EXPERT + r"\.weight$")


def _repack_signed_int4_to_uint4b8(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Signed INT4 packed 2 per int8 -> ``uint4b8`` packed 8 per int32.

    XOR with ``0x88`` adds 8 (mod 16) to both nibbles of each byte.
    """
    for name, tensor in weights:
        if tensor.dtype is torch.int8 and _EXPERT_WEIGHT.search(name):
            tensor = tensor.view(torch.uint8).bitwise_xor(0x88).view(torch.int32)
        yield name, tensor


class W4AFP8Config(Fp8Config):
    def __init__(self, group_size: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.weight_quant = QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=group_size,
        )
        self.input_quant = QuantizationArgs(
            num_bits=8, type="float", symmetric=True, strategy="token", dynamic=True
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "w4afp8"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "W4AFP8Config":
        return cls(
            group_size=config.get("group_size", 128),
            activation_scheme="dynamic",
            ignored_layers=config.get("ignored_layers")
            or config.get("modules_to_not_convert"),
            weight_block_size=config.get("weight_block_size", [128, 128]),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, RoutedExperts) and not is_layer_skipped(
            prefix, self.ignored_layers, self.packed_modules_mapping
        ):
            return CompressedTensorsW4A8Fp8MoEMethod(
                self.weight_quant, self.input_quant, layer.moe_config, prefix
            )
        return super().get_quant_method(layer, prefix)

    def get_checkpoint_weight_mapper(self) -> "WeightsMapper":  # type: ignore[override]
        """Rename routed-expert tensors to the compressed-tensors layout and drop
        their static input scales (the W4A8 kernels quantize per token)."""
        from vllm.model_executor.models.utils import WeightsMapper

        return super().get_checkpoint_weight_mapper() | WeightsMapper(
            orig_to_new_regex={
                re.compile(_EXPERT + r"\.input_scale$"): None,
                _EXPERT_WEIGHT: r"\1.weight_packed",
                re.compile(_EXPERT + r"\.weight_scale_inv$"): r"\1.weight_scale",
            },
            weights_transforms=[_repack_signed_int4_to_uint4b8],
        )
