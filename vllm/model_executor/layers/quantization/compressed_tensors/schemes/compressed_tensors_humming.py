# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CompressedTensors scheme that dispatches pack-quantized INT weights to the
humming Linear kernel. Built for bit widths that Marlin / WNA16 does not
support (e.g. 2-bit GSQ checkpoints).
"""

from typing import Any

import torch
from compressed_tensors.quantization import QuantizationArgs

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (  # noqa: E501
    CompressedTensorsScheme,
)

__all__ = ["CompressedTensorsHumming", "build_humming_layer_quant_config"]


def _ct_weight_schema_dict(
    weight_quant: QuantizationArgs, format_: str
) -> dict[str, Any]:
    """
    Build the dict that humming's ``CompressedTensorsWeightSchema`` expects from a
    compressed-tensors ``QuantizationArgs``. This is the exact shape that
    ``HummingConfig.compressed_tensors_get_config`` would produce when humming is
    invoked via ``--quantization humming``; doing it here lets the regular
    ``compressed-tensors`` loader reach the same kernels.
    """
    cfg: dict[str, Any] = {
        "quant_method": "compressed-tensors",
        "format": format_,
        "type": weight_quant.type,
        "num_bits": weight_quant.num_bits,
        "strategy": weight_quant.strategy,
        "symmetric": weight_quant.symmetric,
    }
    if weight_quant.group_size is not None:
        cfg["group_size"] = weight_quant.group_size
    if weight_quant.block_structure is not None:
        cfg["block_structure"] = weight_quant.block_structure
    if weight_quant.actorder is not None:
        cfg["actorder"] = weight_quant.actorder
    return cfg


def build_humming_layer_quant_config(
    weight_quant: QuantizationArgs, format_: str
):
    """
    Construct a ``HummingLayerQuantizationConfig`` from a CT weight spec, with a
    default (no-op) input schema for weight-only quantization.
    """
    from humming.schema import BaseWeightSchema, HummingInputSchema

    from vllm.model_executor.layers.quantization.humming import (
        HummingLayerQuantizationConfig,
    )

    weight_schema = BaseWeightSchema.from_config(
        _ct_weight_schema_dict(weight_quant, format_)
    )
    return HummingLayerQuantizationConfig(
        weight_schema=weight_schema,
        input_schema=HummingInputSchema(),
    )


class CompressedTensorsHumming(CompressedTensorsScheme):
    """
    Pack-quantized weight-only scheme backed by the humming Linear kernel.

    Wraps ``HummingLinearMethod`` and forwards ``create_weights``,
    ``process_weights_after_loading``, and ``apply_weights`` to it. The humming
    schema is constructed from the CT ``QuantizationArgs`` so the model's HF
    config stays ``quant_method: compressed-tensors`` and no user override is
    required.
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        format_: str,
        layer_name: str | None = None,
    ):
        from vllm.model_executor.layers.quantization.humming import (
            HummingLinearMethod,
        )

        self.layer_name = layer_name
        self._method = HummingLinearMethod(
            build_humming_layer_quant_config(weight_quant, format_)
        )

    @classmethod
    def get_min_capability(cls) -> int:
        # Mirrors HummingConfig.get_min_capability (Turing and up).
        return 75

    def create_weights(self, *args, **kwargs) -> None:
        return self._method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return self._method.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._method.apply(layer, x, bias=bias)
