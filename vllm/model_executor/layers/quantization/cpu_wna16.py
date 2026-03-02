# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

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
        cls, hf_quant_cfg, user_quant
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

    def maybe_update_config(self, model_name: str, revision: str | None = None):
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
        # lyt_debug_A1 no more re-pack to int4; do dequant -> re-quantize int8 instead
        packed_weight = layer.qweight.data
        packed_zeros = layer.qzeros.data
        group_num = packed_zeros.size(0)
        bits = self.quant_config.weight_bits
        pack_factor = int(self.quant_config.pack_factor)
        input_size, packed_output_size = packed_weight.size()
        output_size = packed_output_size * pack_factor
        group_size = input_size // group_num

        print(f'lyt_debug_A1 process_weights_after_loading ENTER: '
            f'input_size(K)={input_size}, output_size(N)={output_size}, '
            f'group_num={group_num}, group_size={group_size}, '
            f'bits={bits}, pack_factor={pack_factor}')
        print(f'lyt_debug_A1 packed_weight shape: {packed_weight.shape}, '
            f'packed_zeros shape: {packed_zeros.shape}, scales shape: {layer.scales.shape}')

        # lyt_debug: Unpack int4 values from packed int32 
        interleave_map = (0, 4, 1, 5, 2, 6, 3, 7)
        weight_int4 = unpack_cols(
            packed_weight,
            bits,
            input_size,
            output_size,
        )
        zeros_int4 = unpack_cols(
            packed_zeros,
            bits,
            group_num,
            output_size,
        )

        # lyt_debug: Reverse AWQ interleave orderring
        weight_int4 = (
            weight_int4.view(input_size, -1, pack_factor)[:, :, interleave_map]
            .reshape(input_size, output_size)
            .contiguous()
        )
        zeros_int4 = (
            zeros_int4.view(group_num, -1, pack_factor)[:, :, interleave_map]
            .reshape(group_num, output_size)
            .contiguous()
        )

        print(f'lyt_debug_A1 after unpack+de-interleave: '
              f'weight_int4 shape={weight_int4.shape}, '
              f'zeros_int4 shape={zeros_int4.shape}')

        # lyt_debug Dequant AWQ int4 -> float32 (A2)
        float_weight = _dequant_awq_to_float(
            weight_int4, zeros_int4, layer.scales.data, group_size
        )

        # lyt_debug Re-quantize float32 -> int8 per-channel (A3)
        weight_int8, channel_scale = _requantize_to_int8(float_weight)

        print(f'lyt_debug_A1 final weight_int8 shape: {weight_int8.shape}, '
              f'dtype: {weight_int8.dtype}')
        print(f'lyt_debug_A1 final channel_scale shape: {channel_scale.shape}, '
              f'dtype: {channel_scale.dtype}')

        # lyt_debug_A4 create oneDNN handler for int8 GEMM, replaces old int4 path
        channel_scale_2d = channel_scale.unsqueeze(0)  # [1, N] for oneDNN

        # AZP adjustment: needed for dynamic quantization compensation
        # Same formula as scaled_mm/cpu.py:118-119
        azp_adj = weight_int8.sum(dim=0, keepdim=True, dtype=torch.float32)
        azp_adj = azp_adj * channel_scale_2d

        # lyt_debug_A4 oneDNN requires column-major weight: stride(0)==1
        # Convert [K, N] row-major → [K, N] column-major via t().contiguous().t()
        weight_int8 = weight_int8.t().contiguous().t()

        print(f'lyt_debug_A4 creating oneDNN handler: '
            f'weight_int8 shape={weight_int8.shape}, stride={weight_int8.stride()}, '
            f'channel_scale_2d shape={channel_scale_2d.shape}')
        print(f'lyt_debug_A4 azp_adj shape={azp_adj.shape}, '
              f'range=[{azp_adj.min().item():.4f}, {azp_adj.max().item():.4f}]')

        layer.dnnl_handler = ops.create_onednn_scaled_mm(
            weight_int8,                # [K, N] int8, column-major
            channel_scale_2d,           # [1, N] float32
            torch.get_default_dtype(),  # output type (typically bf16)
            True,                       # dynamic_act_quant
            False,                      # use_azp (symmetric input)
            32,                         # primitive_cache_size
        )
        layer.azp_adj = torch.nn.Parameter(azp_adj, requires_grad=False)

        print(f'lyt_debug_A4 oneDNN handler created: '
            f'handler.k={layer.dnnl_handler.k}, handler.n={layer.dnnl_handler.n}')

        # lyt_debug_A4 clean up: weight is prepacked in d'nnnn'l_handler,release old int4 params to save memory
        del weight_int8, float_weight
        layer.qweight = None
        layer.qzeros = None
        layer.scales = None

        print(f'lyt_debug_A4 process_weights_after_loading DONE. '
            f'int8 oneDNN path ready, old int4 params cleaned up.')

    _apply_debug_logged = False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # lyt_debug_A5 use oneDNN int8 GEMM instead of cpu_gemm_wna16 int4
        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]) if len(x_shape) > 2 else x

        if not CPUAWQLinearMethod._apply_debug_logged:
            print(f'lyt_debug_A5 apply ENTER (first call): x shape={x.shape}, dtype={x.dtype}')

        # Dynamic per-token symmetric quantization: bf16 → int8
        x_q, x_s, _ = ops.onednn_scaled_int8_quant(x_2d, None, None, True)

        m = x_2d.size(0)
        n = layer.dnnl_handler.n
        out = torch.empty((m, n), dtype=x.dtype)

        ops.onednn_scaled_mm(
            layer.dnnl_handler,
            x_q,
            out,
            x_s,
            None,           # input_zp (symmetric → no zero point)
            layer.azp_adj,  # AZP adjustment
            bias,
        )

        out = out.reshape(x_shape[:-1] + (n,)) if len(x_shape) > 2 else out

        if not CPUAWQLinearMethod._apply_debug_logged:
            print(f'lyt_debug_A5 apply DONE (first call): '
                  f'out shape={out.shape}, dtype={out.dtype}')
            CPUAWQLinearMethod._apply_debug_logged = True

        return out


def _get_isa_hint(dtype: torch.dtype) -> str:
    supports_amx = torch._C._cpu._is_amx_tile_supported()
    if supports_amx and dtype in (torch.bfloat16,):
        return "amx"
    else:
        return "vec"


# lyt_dbug_A2 helper: dequantize AWQ int4 weights to float32
def _dequant_awq_to_float(
    weight_int4: torch.Tensor,
    zeros_int4: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize AWQ int4 weights to float32.
    Args:
        weight_int4: [K, N] int32, each element is a single int4 value (0-15)
        zeros_int4:  [num_groups, N] int32, each element is a single int4 zp (0-15)
        scales:      [num_groups, N] bf16/fp16, per-group per-channel scale
        group_size:  # of rows per group

    Returns:
        float_weight: [K, N] float32
    """
    K, N = weight_int4.shape
    num_groups = zeros_int4.shape[0]

    print(f'lyt_debug_A2 _dequant_awq_to_float called: '
        f'K={K}, N={N}, numgroups={num_groups}, group_size={group_size}')
    print(f'lyt_debug_A2 weight_int4 range: '
        f'min={weight_int4.min().item()}, max={weight_int4.max().item()}')
    print(f'lyt_debug_A2 zeros_int4 range: '
        f'min={zeros_int4.min().item()}, max={zeros_int4.max().item()}')
    print(f'lyt_debug_A2 scales range: '
        f'min={scales.min().item():.6f}, max={scales.max().item():.6f}')

    # Expand zeros &scales: [num_groups, N] -> [K, N] by repeating each group row `group_size` times along dim 0
    zeros_expanded = zeros_int4.repeat_interleave(group_size, dim=0)  # [K, N]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [K, N]
    # AWQ dequant: float_w = (int4_val - zero_point) * scale
    float_weight = ((weight_int4.float() - zeros_expanded.float()) * scales_expanded.float())

    print(f'lyt_debug_A2 float_weight range: min={float_weight.min().item():.6f}, max={float_weight.max().item():.6f}')
    print(f'lyt_debug_A2 float_weight shape: {float_weight.shape}, dtype: {float_weight.dtype}')

    return float_weight


# lyt_debug_A3 helper: re-quantize float32 weights to int8 per-channel symmetric
def _requantize_to_int8(
    float_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize float32 weights to int8 with per-channel symmetric quantization.
    Args:
        float_weight: [K, N] flot32

    Returns:
        weight_int8:    [K, N] int8 (actually stored as [N, K] for oneDNN)
        channel_scale:  [N] float32, per-output-channel scale
    """
    K, N = float_weight.shape

    # Per-channel (per-column) symmetric quantization
    channel_max = float_weight.abs().amax(dim=0)  # [N]
    channel_scale = (channel_max / 127.0).clamp(min=1e-10)  # [N]

    weight_int8 = (float_weight / channel_scale.unsqueeze(0)).round().clamp(
        -128, 127
    ).to(torch.int8)  # [K, N]

    # lyt_could_comment_cout： Verify the quantization roundtrip error
    reconstructed = weight_int8.float() * channel_scale.unsqueeze(0)
    abs_error = (float_weight - reconstructed).abs()
    rel_error_mask = float_weight.abs() > 1e-6
    rel_error = torch.zeros_like(abs_error)
    rel_error[rel_error_mask] = (
        abs_error[rel_error_mask] / float_weight[rel_error_mask].abs()
    )

    print(f'lyt_debug_A3 _requantize_to_int8 called: K={K}, N={N}')
    print(f'lyt_debug_A3 channel_scale range: '
        f'min={channel_scale.min().item():.8f}, max={channel_scale.max().item():.8f}')
    print(f'lyt_debug_A3 weight_int8 range: '
          f'min={weight_int8.min().item()}, max={weight_int8.max().item()}')
    print(f'lyt_debug_A3 requant abs_error: '
        f'mean={abs_error.mean().item():.8f}, max={abs_error.max().item():.8f}')
    print(f'lyt_debug_A3 requant rel_error (where |w|>1e-6): '
        f'mean={rel_error[rel_error_mask].mean().item():.6f}, '
        f'max={rel_error[rel_error_mask].max().item():.6f}')

    return weight_int8, channel_scale
