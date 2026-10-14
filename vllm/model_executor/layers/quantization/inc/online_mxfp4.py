# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter

_REQUANTIZE_CHUNK_ELEMS = 16 * 1024 * 1024


class INCOnlineMxFp4LinearMethod(LinearMethodBase):
    """Load ModelOpt NVFP4 weights and requantize them to MXFP4 once."""

    uses_meta_device: bool = True

    def __init__(self, quant_config: ModelOptNvFp4Config) -> None:
        self.quant_config = quant_config
        self.kernel = init_mxfp4_linear_kernel(activation_quant_key=kMxfp4Dynamic)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        if input_size_per_partition % 32 != 0:
            raise ValueError(
                "MXFP4 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by 32."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
                device="meta",
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        input_scale = PerTensorScaleParameter(
            data=torch.empty(
                len(output_partition_sizes),
                dtype=torch.float32,
                device="meta",
            ),
            weight_loader=weight_loader,
        )
        input_scale.needs_scalar_to_array = True
        layer.register_parameter("input_scale", input_scale)

        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(
                len(output_partition_sizes),
                dtype=torch.float32,
                device="meta",
            ),
            weight_loader=weight_loader,
        )
        weight_scale_2.needs_scalar_to_array = True
        layer.register_parameter("weight_scale_2", weight_scale_2)

        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=torch.float8_e4m3fn,
                device="meta",
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        partition_widths = torch.tensor(
            layer.logical_widths,
            device=layer.weight_scale_2.device,
        )
        row_global_scale = torch.repeat_interleave(
            layer.weight_scale_2, partition_widths
        )
        chunk_rows = max(1, _REQUANTIZE_CHUNK_ELEMS // layer.input_size_per_partition)
        weight_mxfp4 = None
        weight_scale_mxfp4 = None

        for start in range(0, layer.output_size_per_partition, chunk_rows):
            rows = slice(
                start,
                min(start + chunk_rows, layer.output_size_per_partition),
            )
            weight = dequantize_to_dtype(
                layer.weight[rows],
                layer.weight_scale[rows],
                row_global_scale[rows, None],
                torch.float32,
                block_size=self.quant_config.group_size,
                swizzle=False,
            )
            quantized, scale = mxfp4_quantize(weight.contiguous())
            if weight_mxfp4 is None:
                weight_mxfp4 = torch.empty(
                    (layer.output_size_per_partition, quantized.shape[1]),
                    dtype=quantized.dtype,
                    device=quantized.device,
                )
                weight_scale_mxfp4 = torch.empty(
                    (layer.output_size_per_partition, scale.shape[1]),
                    dtype=scale.dtype,
                    device=scale.device,
                )
            assert weight_scale_mxfp4 is not None
            weight_mxfp4[rows] = quantized
            weight_scale_mxfp4[rows] = scale

        assert weight_mxfp4 is not None
        assert weight_scale_mxfp4 is not None
        del layer.input_scale
        del layer.weight_scale_2
        replace_parameter(layer, "weight", weight_mxfp4.data)
        replace_parameter(layer, "weight_scale", weight_scale_mxfp4.data)
        layer.input_scale = None
        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class INCOnlineMxFp4Config(ModelOptMixedPrecisionConfig):
    """ModelOpt mixed config with INC online NVFP4-to-MXFP4 conversion."""

    def get_name(self) -> QuantizationMethods:
        return "inc_online_mxfp4"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if user_quant == "inc_online_mxfp4":
            return "inc_online_mxfp4"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if (
            isinstance(layer, (LinearBase, ParallelLMHead))
            and not self.is_layer_excluded(prefix)
            and self._resolve_quant_algo(prefix) == "NVFP4"
        ):
            return INCOnlineMxFp4LinearMethod(self.nvfp4_config)
        return super().get_quant_method(layer, prefix)
