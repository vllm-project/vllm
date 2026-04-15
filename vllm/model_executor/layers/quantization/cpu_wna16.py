# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    pack_cols,
    unpack_cols,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_safetensors_params_metadata

logger = init_logger(__name__)


class CPUAWQConfig(QuantizationConfig):
    """Config class for CPU AWQ"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: list[str] | None,
        full_config: dict[str, Any],
    ) -> None:
        super().__init__()
        assert weight_bits == 4
        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.zero_point = zero_point
        self.lm_head_quantized = lm_head_quantized
        self.weight_bits = weight_bits
        self.modules_to_not_convert = modules_to_not_convert or []
        self.full_config = full_config

    def __repr__(self) -> str:
        return (
            f"AWQMarlinConfig("
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    @classmethod
    def get_name(cls) -> "QuantizationMethods":
        return "cpu_awq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CPUAWQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        quant_method = hf_quant_cfg.get("quant_method", "").lower()
        if current_platform.is_cpu() and (quant_method == "awq"):
            return cls.get_name()
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            return CPUAWQLinearMethod(self)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        if self.modules_to_not_convert:
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        layers = {param_name.rsplit(".", 1)[0] for param_name in metadata}
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_to_not_convert = list(layers - quant_layers)


class CPUAWQLinearMethod(LinearMethodBase):
    """Linear method for CPU AWQ.

    Args:
        quant_config: The CPU AWQ quantization config.
    """

    def __init__(self, quant_config: CPUAWQConfig) -> None:
        self.quant_config = quant_config
        assert self.quant_config.zero_point

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
        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.use_w4a8 = envs.VLLM_CPU_INT4_W4A8 and torch.cpu._is_amx_tile_supported()
        if layer.use_w4a8:
            self._process_weights_sglang_int4(layer)
        else:
            self._process_weights_woq(layer)

    def _process_weights_woq(self, layer: torch.nn.Module) -> None:
        """Original WOQ int4 repack path."""
        packed_weight = layer.qweight.data
        packed_zeros = layer.qzeros.data
        group_num = packed_zeros.size(0)
        bits = self.quant_config.weight_bits
        pack_factor = int(self.quant_config.pack_factor)
        input_size, packed_output_size = packed_weight.size()
        output_size = packed_output_size * pack_factor
        isa_hint = _get_isa_hint(layer.scales.dtype)
        layer.isa_hint = isa_hint

        interleave_map = (0, 4, 1, 5, 2, 6, 3, 7)
        weight = unpack_cols(
            packed_weight,
            bits,
            input_size,
            output_size,
        )
        zeros = unpack_cols(
            packed_zeros,
            bits,
            group_num,
            output_size,
        )
        weight = (
            weight.view(input_size, -1, pack_factor)[:, :, interleave_map]
            .reshape(input_size, output_size)
            .contiguous()
        )
        zeros = (
            zeros.view(group_num, -1, pack_factor)[:, :, interleave_map]
            .reshape(group_num, output_size)
            .contiguous()
        )

        zeros = pack_cols(zeros, bits, group_num, output_size).contiguous()
        weight = pack_cols(weight, bits, input_size, output_size)
        weight = (
            weight.view(input_size, -1, 16 // pack_factor)
            .permute(1, 0, 2)
            .reshape(-1, input_size * 16 // pack_factor)
            .contiguous()
        )
        layer.qweight.data = weight
        layer.qzeros.data = zeros

    def _process_weights_sglang_int4(self, layer: torch.nn.Module) -> None:
        """SGLang INT4 W4A8 path: pack int4 weights with VNNI reordering."""
        packed_weight = layer.qweight.data
        packed_zeros = layer.qzeros.data
        scales = layer.scales.data
        blocked_w, blocked_zp, blocked_s = torch.ops._C.convert_weight_packed_scale_zp(
            packed_weight, packed_zeros, scales
        )

        layer.packed_weight = blocked_w
        layer.packed_qzeros = blocked_zp
        layer.packed_scales = blocked_s
        layer.qweight = None
        layer.qzeros = None
        layer.scales = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.use_w4a8:
            return self._apply_sglang_int4(layer, x, bias)
        return self._apply_woq(layer, x, bias)

    def _apply_woq(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Original WOQ int4 GEMM path."""
        x = ops.cpu_gemm_wna16(
            input=x,
            q_weight=layer.qweight,
            scales=layer.scales,
            zeros=layer.qzeros,
            g_idx=None,
            bias=bias,
            pack_factor=8,
            isa_hint=layer.isa_hint,
        )
        return x

    def _apply_sglang_int4(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SGLang INT4 W4A8 GEMM path."""
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]) if len(x_shape) > 2 else x

        out = torch.ops._C.int4_scaled_mm_cpu(
            x_2d,
            layer.packed_weight,
            layer.packed_qzeros,
            layer.packed_scales,
            bias,
        )
        out = out.reshape(x_shape[:-1] + (out.size(-1),)) if len(x_shape) > 2 else out
        return out


def _get_isa_hint(dtype: torch.dtype) -> str:
    supports_amx = torch.cpu._is_amx_tile_supported()
    if supports_amx and dtype in (torch.bfloat16,):
        return "amx"
    else:
        return "vec"
