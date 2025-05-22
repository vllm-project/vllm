from typing import Any, Dict, List, Optional, NamedTuple

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.awq import is_layer_skipped_awq
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


class AutoQuantConfig(QuantizationConfig):
    """Config class for AutoQuant.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        from_float: bool,
        quant_mode: str,  # weight_only
        lm_head_quantized: bool = False,
        modules_to_not_convert: Optional[List[str]] = None
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.from_float = from_float
        self.quant_mode = quant_mode
        self.lm_head_quantized = lm_head_quantized
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AutoQuant, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AutoQuantConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point}, "
                f"from_float={self.from_float}, "
                f"quant_mode={self.quant_mode}, "
                f"lm_head_quantized={self.lm_head_quantized}, "
                f"modules_to_not_convert={self.modules_to_not_convert})")

    def get_name(self) -> str:
        return "auto_quant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AutoQuant kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AutoQuantConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])

        from_float = cls.get_from_keys_or(config, ["from_float"], default=False)
        quant_mode = cls.get_from_keys_or(config, ["quant_mode"], default="weight_only")
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None)
        return cls(weight_bits, group_size, zero_point,
                   from_float, quant_mode,
                   lm_head_quantized, modules_to_not_convert)

    def get_quant_method(
            self, layer: torch.nn.Module, prefix: str) -> Optional["AutoQuantLinearMethod"]:
        from vllm.attention.layer import Attention
        if (isinstance(layer, LinearBase) or (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AutoQuantLinearMethod(self)
        elif isinstance(layer, Attention):
            return AutoQuantKVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AutoQuantLinearMethod(LinearMethodBase):
    """Linear method for AutoQuant.

    Args:
        quant_config: The AutoQuant quantization config.
    """

    def __init__(self, quant_config: AutoQuantConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if self.quant_config.quant_mode == "weight_only" and \
                not self.quant_config.from_float:
            layer.process_after_load = True
            qweight = Parameter(
                torch.empty(
                    input_size_per_partition,
                    output_size_per_partition // self.quant_config.pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            set_weight_attrs(
                qweight, {
                    "input_dim": 0,
                    "output_dim": 1,
                    "packed_dim": 1,
                    "pack_factor": self.quant_config.pack_factor,
                })
            qzeros = Parameter(
                torch.empty(
                    input_size_per_partition // self.quant_config.group_size,
                    output_size_per_partition // self.quant_config.pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            set_weight_attrs(
                qzeros, {
                    "input_dim": 0,
                    "output_dim": 1,
                    "packed_dim": 1,
                    "pack_factor": self.quant_config.pack_factor,
                })
            scales = Parameter(
                torch.empty(
                    input_size_per_partition // self.quant_config.group_size,
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            set_weight_attrs(scales, {
                "input_dim": 0,
                "output_dim": 1,
            })
    
            layer.register_parameter("qweight", qweight)
            set_weight_attrs(qweight, extra_weight_attrs)
            layer.register_parameter("qzeros", qzeros)
            set_weight_attrs(qzeros, extra_weight_attrs)
            layer.register_parameter("scales", scales)
            set_weight_attrs(scales, extra_weight_attrs)
        else:
            layer.process_after_load = True
            weight = Parameter(torch.empty(output_size_per_partition,
                                           input_size_per_partition,
                                           dtype=params_dtype,
                                           device="cpu"),
                               requires_grad=False)
            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return
        if self.quant_config.from_float:
            _qweight, _scales, _qzeros = quantize_tensor(
                        layer.weight.cuda(), n_bits=4, group_size=128)
            qweight, scales_zeros = convert_s4(_qweight, _qzeros,
                                                       _scales)
            layer.qweight = Parameter(qweight, requires_grad=False)
            layer.scales_zeros = Parameter(scales_zeros, requires_grad=False)
            del layer.weight
        else:
            qweight, scales_zeros = convert_s4(layer.qweight, layer.qzeros,
                                                layer.scales)
            layer.qweight = Parameter(qweight, requires_grad=False)
            layer.scales_zeros = Parameter(scales_zeros, requires_grad=False)
            del layer.qzeros
            del layer.scales

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.quant_config.quant_mode == "weight_only":
            qweight = layer.qweight.data
            scales_zeros = layer.scales_zeros.data
            pack_factor = self.quant_config.pack_factor
            out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
            reshaped_x = x.reshape(-1, x.shape[-1])
            out = ops.autoquant_s4_f16_gemm(reshaped_x, qweight, scales_zeros)
            if bias is not None:
                out = out + bias
            return out.reshape(out_shape)


class QParams(NamedTuple):
    """A class to hold the quantization parameters."""

    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]


@torch.no_grad()
def cal_qparams_per_group_minmax(w: torch.Tensor,
                                 n_bits: int = 4,
                                 group_size: int = 128):
    """Calculate quantization parameters for each group using min and max
    values."""

    outc, inc = w.shape
    assert inc >= group_size, \
        'Input channels should be greater than or equal to group_size.'
    assert inc % group_size == 0, \
        'Input channels should be divisible by group_size.'
    w_group_wise = w.reshape(outc, -1, group_size)
    w_min = w_group_wise.min(dim=-1, keepdim=True)[0]
    w_max = w_group_wise.max(dim=-1, keepdim=True)[0]

    q_max = 2**n_bits - 1
    q_min = 0
    scales = (w_max - w_min)
    scales = scales.clamp_(min=1e-5).div_(q_max)
    # zero_points = (-w_min / scales).round().clamp(q_min, q_max)
    zero_points = (-torch.round(w_min / scales)).clamp_(q_min, q_max)
    return QParams(scales=scales, zero_points=zero_points)


def convert_s4(qw: torch.Tensor,
               qz: torch.Tensor,
               s: torch.Tensor,
               group_size: int = 128):
    assert qw.is_contiguous()
    assert qz.is_contiguous()
    assert s.is_contiguous()
    _qw = torch.zeros_like(qw)
    _sz = torch.zeros_like(s, dtype=torch.int32)  # half2
    _ws = torch.zeros_like(s)
    ops.autoquant_convert_s4_k_m8(_qw, _sz, _ws, qw, s, qz,
                                  qw.size(-1) * 8, qw.size(0), group_size)
    return _qw, _sz


def tp_m_s4(x: torch.Tensor, tp: int = 1):
    return x.view(x.size(0) // 32, tp, -1, 128).permute(0, 2, 3,
                                                        1).contiguous()


def quant(weight: torch.Tensor,
          qparams: Optional[QParams] = None) -> torch.Tensor:
    """Perform fake quantization on the given weight tensor.
    Args:
        weight (torch.Tensor): The weight tensor with shape
            (out_features, in_features).
        qparams (Optional[QParams]): A namedtuple containing 'scales'
            and 'zero_points'.
    Returns:
        torch.Tensor: The fake quantized weight tensor.
    """
    if qparams is None:
        qparams = cal_qparams_per_group_minmax(weight)
    scales = qparams.scales
    zero_points = qparams.zero_points
    out_c, in_c = weight.shape
    # Reshape the weights if using per_group quantization
    # per tensor scales shape: [1]
    # per channel scales shape: [out_c, 1]
    # per group scales shape: [out_c, in_c//group_size, 1]
    if len(scales.shape) > 2:
        # scales shape: [out_c, in_c//group_size, 1]
        weight = weight.reshape(out_c, scales.shape[1], -1)
    if zero_points is None:
        real_qweight = (weight / scales).round()
    else:
        real_qweight = ((weight + (scales * zero_points)) / scales).round()
    # add for codeqwen1.5-7b copilot from_float low accuracy
    q_min = 0
    q_max = 15
    real_qweight = real_qweight.clamp_(q_min, q_max)
    if len(scales.shape) > 2:
        real_qweight = real_qweight.reshape(out_c, in_c)
    return real_qweight.to(torch.int32)


# core quantization method
def quantize_tensor(
    weight,
    n_bits=4,
    group_size=128,
):
    pack_num = 32 // n_bits
    pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
    org_weight_shape = weight.shape
    out_features = org_weight_shape[0]
    in_features = org_weight_shape[1]
    qparams = cal_qparams_per_group_minmax(weight, n_bits)
    i32_w = quant(weight, qparams)
    i32_w = i32_w.t().contiguous()
    w_pack_oc = out_features // (32 // n_bits)
    w_inc = in_features
    pack_int_w = torch.zeros((w_inc, w_pack_oc),
                             dtype=torch.int32,
                             device=weight.device)
    for col in range(pack_int_w.shape[1]):
        for i in range(pack_num):
            pack_int_w_col = i32_w[:, col * pack_num + pack_order[i]]
            pack_int_w[:, col] |= pack_int_w_col << (i * n_bits)
    qweight = pack_int_w
    scales = qparams.scales.squeeze(-1).t().contiguous()
    if qparams.zero_points is not None:
        zeros = qparams.zero_points.to(torch.int32)
        zeros = zeros.squeeze(-1).t().contiguous()
        z_inc = in_features // group_size
        z_oc = out_features // (32 // n_bits)
        pack_int_zeros = torch.zeros((z_inc, z_oc),
                                     dtype=torch.int32,
                                     device=weight.device)
        for col in range(pack_int_zeros.shape[1]):
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + pack_order[i]]
                pack_int_zeros[:, col] |= qzero_col << (i * n_bits)
        qzeros = pack_int_zeros
    return qweight, scales, qzeros


class AutoQuantKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from checkpoints.
    """

    def __init__(self, quant_config: AutoQuantConfig):
        super().__init__(quant_config)
