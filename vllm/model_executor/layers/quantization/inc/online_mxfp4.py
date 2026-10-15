# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineMoEMethod,
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


def _requantize_nvfp4_moe_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    row_global_scale: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts, output_size, packed_input_size = weight.shape
    chunk_rows = max(1, _REQUANTIZE_CHUNK_ELEMS // (packed_input_size * 2))
    weight_mxfp4 = None
    weight_scale_mxfp4 = None

    for expert_id in range(num_experts):
        for start in range(0, output_size, chunk_rows):
            rows = slice(start, min(start + chunk_rows, output_size))
            dequantized = dequantize_to_dtype(
                weight[expert_id, rows],
                weight_scale[expert_id, rows],
                row_global_scale[expert_id, rows, None],
                torch.float32,
                block_size=group_size,
                swizzle=False,
            )
            quantized, scale = mxfp4_quantize(dequantized.contiguous())
            if weight_mxfp4 is None:
                weight_mxfp4 = torch.empty(
                    (num_experts, output_size, quantized.shape[1]),
                    dtype=quantized.dtype,
                    device=quantized.device,
                )
                weight_scale_mxfp4 = torch.empty(
                    (num_experts, output_size, scale.shape[1]),
                    dtype=scale.dtype,
                    device=scale.device,
                )
            assert weight_scale_mxfp4 is not None
            weight_mxfp4[expert_id, rows] = quantized
            weight_scale_mxfp4[expert_id, rows] = scale

    assert weight_mxfp4 is not None
    assert weight_scale_mxfp4 is not None
    return weight_mxfp4, weight_scale_mxfp4


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


class INCNvFp4Bf16LMHeadMethod(INCOnlineMxFp4LinearMethod):
    """Load NVFP4 lm_head weights and use BF16 dense logits."""

    def __init__(self, quant_config: ModelOptNvFp4Config) -> None:
        self.quant_config = quant_config
        self.dense_method = UnquantizedLinearMethod()

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
        weight_bf16 = None

        for start in range(0, layer.output_size_per_partition, chunk_rows):
            rows = slice(
                start, min(start + chunk_rows, layer.output_size_per_partition)
            )
            weight = dequantize_to_dtype(
                layer.weight[rows],
                layer.weight_scale[rows],
                row_global_scale[rows, None],
                torch.bfloat16,
                block_size=self.quant_config.group_size,
                swizzle=False,
            )
            if weight_bf16 is None:
                weight_bf16 = torch.empty(
                    (layer.output_size_per_partition, weight.shape[1]),
                    dtype=torch.bfloat16,
                    device=weight.device,
                )
            weight_bf16[rows] = weight

        assert weight_bf16 is not None
        del layer.input_scale
        del layer.weight_scale_2
        del layer.weight_scale
        replace_parameter(layer, "weight", weight_bf16.data)
        layer.input_scale = None
        self.dense_method.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.dense_method.apply(layer, x, bias)


class INCMxFp4MoEMethod(Mxfp4OnlineMoEMethod):
    """Load ModelOpt NVFP4 experts and convert them to MXFP4 at load time."""

    def __init__(
        self, quant_config: ModelOptNvFp4Config, moe_config: FusedMoEConfig
    ) -> None:
        self.quant_config = quant_config
        super().__init__(moe=moe_config)

    @staticmethod
    def _weight_loader(layer: RoutedExperts, default_loader):
        def load_weight(
            param,
            loaded_weight,
            weight_name,
            shard_id,
            expert_id,
            return_success=False,
        ):
            local_expert_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if local_expert_id == -1:
                return False if return_success else None
            if "input_scale" in weight_name and shard_id in ("w1", "w3"):
                shard_index = 0 if shard_id == "w1" else 1
                param.data[local_expert_id, shard_index] = layer._to_scalar(
                    loaded_weight.to(param.device)
                )
            elif "weight_scale_2" in weight_name:
                layer._load_per_tensor_weight_scale(
                    shard_id, param, loaded_weight.to(param.device), local_expert_id
                )
            else:
                return default_loader(
                    param,
                    loaded_weight,
                    weight_name,
                    shard_id,
                    expert_id,
                    return_success,
                )
            return True if return_success else None

        return load_weight

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size % 32 or intermediate_size_per_partition % 32:
            raise ValueError("MXFP4 MoE requires dimensions divisible by 32.")

        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.quant_config = self.quant_config
        layer.weight_block_size = [1, 32]
        weight_loader = self._weight_loader(layer, extra_weight_attrs["weight_loader"])
        w13_num_shards = self.moe.w13_num_shards

        for name, rows, columns in (
            ("w13", w13_num_shards * intermediate_size_per_partition, hidden_size),
            ("w2", hidden_size, intermediate_size_per_partition),
        ):
            layer.register_parameter(
                f"{name}_weight",
                ModelWeightParameter(
                    data=torch.empty(
                        num_experts,
                        rows,
                        columns // 2,
                        dtype=torch.uint8,
                        device="meta",
                    ),
                    input_dim=1,
                    output_dim=2,
                    weight_loader=weight_loader,
                ),
            )
            weight_scale = ModelWeightParameter(
                data=torch.empty(
                    num_experts,
                    rows,
                    columns // self.quant_config.group_size,
                    dtype=torch.float8_e4m3fn,
                    device="meta",
                ),
                input_dim=1,
                output_dim=2,
                weight_loader=weight_loader,
            )
            weight_scale.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
            layer.register_parameter(f"{name}_weight_scale", weight_scale)

        for name, shape in (
            ("w13", (num_experts, w13_num_shards)),
            ("w2", (num_experts,)),
        ):
            layer.register_parameter(
                f"{name}_weight_scale_2",
                PerTensorScaleParameter(
                    data=torch.empty(*shape, dtype=torch.float32, device="meta"),
                    weight_loader=weight_loader,
                ),
            )

        for name, shape in (
            ("w13", (num_experts, w13_num_shards)),
            ("w2", (num_experts,)),
        ):
            layer.register_parameter(
                f"{name}_input_scale",
                PerTensorScaleParameter(
                    data=torch.empty(*shape, dtype=torch.float32, device="meta"),
                    weight_loader=weight_loader,
                ),
            )
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        shard_width = layer.w13_weight.shape[1] // self.moe.w13_num_shards
        w13_global_scale = torch.repeat_interleave(
            layer.w13_weight_scale_2, shard_width, dim=1
        )
        w2_global_scale = layer.w2_weight_scale_2[:, None].expand(
            -1, layer.w2_weight.shape[1]
        )
        w13, w13_scale = _requantize_nvfp4_moe_weight(
            layer.w13_weight,
            layer.w13_weight_scale,
            w13_global_scale,
            self.quant_config.group_size,
        )
        w2, w2_scale = _requantize_nvfp4_moe_weight(
            layer.w2_weight,
            layer.w2_weight_scale,
            w2_global_scale,
            self.quant_config.group_size,
        )

        del layer.w13_weight_scale_2
        del layer.w2_weight_scale_2
        del layer.w13_input_scale
        del layer.w2_input_scale
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            getattr(layer, "w13_bias", None),
            getattr(layer, "w2_bias", None),
        )
        layer._already_called_process_weights_after_loading = True


class INCMxFp4Config(ModelOptMixedPrecisionConfig):
    """ModelOpt mixed config with INC online NVFP4-to-MXFP4 conversion."""

    def get_name(self) -> QuantizationMethods:
        return "inc_mxfp4"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if user_quant == "inc_mxfp4":
            return "inc_mxfp4"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if (
            isinstance(layer, ParallelLMHead)
            and self.is_layer_excluded(prefix)
            and self._resolve_quant_algo(prefix) in ("NVFP4", "W4A16_NVFP4")
        ):
            quant_config = (
                self.nvfp4_config
                if self._resolve_quant_algo(prefix) == "NVFP4"
                else self.w4a16_nvfp4_config
            )
            return INCNvFp4Bf16LMHeadMethod(quant_config)
        if (
            isinstance(layer, (LinearBase, ParallelLMHead))
            and not self.is_layer_excluded(prefix)
            and self._resolve_quant_algo(prefix) in ("NVFP4", "W4A16_NVFP4")
        ):
            quant_config = (
                self.nvfp4_config
                if self._resolve_quant_algo(prefix) == "NVFP4"
                else self.w4a16_nvfp4_config
            )
            return INCOnlineMxFp4LinearMethod(quant_config)
        if isinstance(layer, RoutedExperts) and not self.is_layer_excluded(prefix):
            quant_algo = self._resolve_quant_algo(prefix)
            if quant_algo in ("NVFP4", "W4A16_NVFP4"):
                quant_config = (
                    self.nvfp4_config
                    if quant_algo == "NVFP4"
                    else self.w4a16_nvfp4_config
                )
                return INCMxFp4MoEMethod(quant_config, layer.moe_config)
        return super().get_quant_method(layer, prefix)
