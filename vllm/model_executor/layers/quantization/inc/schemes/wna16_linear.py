# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
)
from vllm.scalar_type import scalar_types

from .base import INCLinearScheme

if TYPE_CHECKING:
    import torch

    from ..resolver import INCLayerConfig


class INCWNA16LinearScheme(INCLinearScheme):
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.layer_config = layer_config
        self.inner_method = self._build_inner_method()

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    def _build_inner_method(self):
        if self.layer_config.is_gptq:
            return self._build_gptq_method()
        if self.layer_config.is_awq:
            return self._build_awq_method()
        raise NotImplementedError(
            f"WNA16 linear scheme does not support {self.layer_config}"
        )

    def _build_gptq_method(self):
        gptq_type_map = {
            (4, True): scalar_types.uint4b8,
            (8, True): scalar_types.uint8b128,
        }
        use_marlin = (
            self.layer_config.backend == "auto" or "marlin" in self.layer_config.backend
        ) and (self.layer_config.bits, self.layer_config.sym) in gptq_type_map
        if use_marlin:
            use_marlin = check_marlin_supported(
                gptq_type_map[(self.layer_config.bits, self.layer_config.sym)],
                self.layer_config.group_size,
                has_zp=not self.layer_config.sym,
            )

        if use_marlin:
            from vllm.model_executor.layers.quantization.gptq_marlin import (
                GPTQMarlinLinearMethod,
            )

            return GPTQMarlinLinearMethod(
                GPTQMarlinConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=self.layer_config.group_size,
                    desc_act=False,
                    is_sym=self.layer_config.sym,
                    lm_head_quantized=False,
                    dynamic={},
                    full_config={},
                )
            )

        from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod

        return GPTQLinearMethod(
            GPTQConfig(
                weight_bits=self.layer_config.bits,
                group_size=self.layer_config.group_size,
                desc_act=False,
                lm_head_quantized=False,
                dynamic={},
            )
        )

    def _build_awq_method(self):
        awq_type_map = {
            4: scalar_types.uint4,
            8: scalar_types.uint8,
        }
        use_marlin = (
            self.layer_config.backend == "auto" or "marlin" in self.layer_config.backend
        ) and self.layer_config.bits in awq_type_map
        if use_marlin:
            use_marlin = check_marlin_supported(
                awq_type_map[self.layer_config.bits],
                self.layer_config.group_size,
                not self.layer_config.sym,
            )

        if use_marlin:
            from vllm.model_executor.layers.quantization.awq_marlin import (
                AWQMarlinLinearMethod,
            )

            return AWQMarlinLinearMethod(
                AWQMarlinConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=self.layer_config.group_size,
                    zero_point=not self.layer_config.sym,
                    lm_head_quantized=False,
                    modules_to_not_convert=[],
                    full_config={},
                )
            )

        from vllm.model_executor.layers.quantization.awq import AWQLinearMethod

        return AWQLinearMethod(
            AWQConfig(
                weight_bits=self.layer_config.bits,
                group_size=self.layer_config.group_size,
                zero_point=not self.layer_config.sym,
            )
        )

    def create_weights(
        self,
        layer: "torch.nn.Module",
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: "torch.dtype",
        **extra_weight_attrs,
    ) -> None:
        return self.inner_method.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: "torch.nn.Module") -> None:
        return self.inner_method.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: "torch.nn.Module",
        x: "torch.Tensor",
        bias: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        return self.inner_method.apply(layer, x, bias)
