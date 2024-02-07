from typing import Any, Dict, List, NamedTuple, TypeVar, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)


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
            quant_mode: str,  # llm_int8, smoothquant, weight_only
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.from_float = from_float
        self.quant_mode = quant_mode

        if quant_mode == "weight_only" and self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AutoQuant weight_only, but got {self.weight_bits} bits.")
        if quant_mode in ["llm_int8", "smoothquant"] and self.weight_bits != 8:
            raise ValueError(
                "Currently, only 8-bit weight quantization is supported for "
                "AutoQuant llm_int8 or smoothquant, "
                f"but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AutoQuantConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point}, "
                f"from_float={self.from_float}, "
                f"quant_mode={self.quant_mode})")

    def get_name(self) -> str:
        return "auto_quant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_min_capability(self) -> int:
        # The AutoQuant kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            "quantize_config.json",  # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AutoQuantConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        try:
            from_float = cls.get_from_keys(config, ["from_float"])
        except Exception:
            from_float = False
        try:
            quant_mode = cls.get_from_keys(config, ["quant_mode"])
        except Exception:
            quant_mode = "weight_only"
        return cls(weight_bits, group_size, zero_point, from_float, quant_mode)

    def get_linear_method(self) -> "AutoQuantLinearMethod":
        return AutoQuantLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AutoQuantLinearMethod(LinearMethodBase):
    """Linear method for AutoQuant.

    Args:
        quant_config: The AutoQuant quantization config.
    """

    def __init__(self, quant_config: AutoQuantConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        if self.quant_config.quant_mode == "weight_only" and \
                input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if self.quant_config.quant_mode == "weight_only" and \
                output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if self.quant_config.quant_mode == "weight_only" and \
                not self.quant_config.from_float:
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
            return {
                "qweight": qweight,
                "qzeros": qzeros,
                "scales": scales,
            }
        else:
            weight = Parameter(torch.empty(output_size_per_partition,
                                           input_size_per_partition,
                                           dtype=params_dtype),
                               requires_grad=False)
            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
            return {"weight": weight}

    def apply_weights(self,
                      weights: Dict[str, torch.Tensor],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.quant_config.quant_mode == "weight_only":
            qweight = weights["qweight"].data
            scales_zeros = weights["scales_zeros"].data
            pack_factor = self.quant_config.pack_factor
            out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
            reshaped_x = x.reshape(-1, x.shape[-1])
            out = ops.autoquant_s4_f16_gemm(reshaped_x, qweight, scales_zeros)
            if bias is not None:
                out = out + bias
            return out.reshape(out_shape)
        else:
            weight = weights["weight"]
            state = weights["state"]
            if weight.CB is not None:
                state.CB = weight.CB
                state.SCB = weight.SCB
                weight.CB = None
                weight.SCB = None
            import bitsandbytes as bnb
            out = bnb.matmul(x, weight, bias=bias, state=state)
            if not state.has_fp16_weights and \
                    state.CB is not None and state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format
                # in the first inference pass
                # we no longer need the row-major weight
                del state.CB
                weight.data = state.CxB
            return out


T = TypeVar("T", bound="torch.nn.Module")


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
    if len(scales.shape) > 2:
        real_qweight = real_qweight.reshape(out_c, in_c)
    return real_qweight.to(torch.int32)


# core quantization method (simulated quantization)
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


def replace_quant_params(model,
                         quant_config,
                         modules_to_not_convert="lm_head"):
    """
    modules_to_not_convert (`str`, *optional*, defaults to `lm_head`):
            Name of the module to not convert in `Linear8bitLt`.
            In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
    """
    if not isinstance(modules_to_not_convert, list):
        modules_to_not_convert = [modules_to_not_convert]
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_quant_params(module, quant_config, modules_to_not_convert)
        if isinstance(
            module,
                (ColumnParallelLinear, QKVParallelLinear, RowParallelLinear)) \
                and name not in modules_to_not_convert:
            if quant_config.from_float:
                module.linear_weights.pop("weight")
                param = module._parameters["weight"]
                if quant_config.quant_mode in ("llm_int8", "smoothquant"):
                    import bitsandbytes as bnb
                    new_value = bnb.nn.Int8Params(param.data,
                                                  requires_grad=False,
                                                  has_fp16_weights=False)
                    state = bnb.MatmulLtState()
                    if quant_config.quant_mode == "smoothquant":
                        state.threshold = 0.0
                    else:
                        state.threshold = 6.0
                    state.has_fp16_weights = False
                    state.memory_efficient_backward = False
                    state.use_pool = True
                    module._parameters["weight"] = new_value
                    module.linear_weights["weight"] = new_value
                    module.linear_weights["state"] = state
                    set_weight_attrs(
                        new_value, {
                            "input_dim": 0,
                            "output_dim": 1,
                            "packed_dim": 1,
                            "pack_factor": quant_config.pack_factor,
                        })
                    del param
                    torch.cuda.empty_cache()

                elif quant_config.quant_mode == "weight_only":
                    data_fp = param.cuda()
                    _qweight, _scales, _qzeros = quantize_tensor(
                        data_fp, n_bits=4, group_size=128)
                    qweight, scales_zeros = convert_s4(_qweight, _qzeros,
                                                       _scales)
                    torch.cuda.synchronize()
                    param_qweight = Parameter(qweight, requires_grad=False)
                    param_scales_zeros = Parameter(scales_zeros,
                                                   requires_grad=False)
                    module.register_parameter("qweight", param_qweight)
                    module.register_parameter("scales_zeros",
                                              param_scales_zeros)
                    set_weight_attrs(
                        param_qweight, {
                            "input_dim": 0,
                            "output_dim": 1,
                            "packed_dim": 1,
                            "pack_factor": quant_config.pack_factor,
                        })
                    set_weight_attrs(param_scales_zeros, {
                        "input_dim": 0,
                        "output_dim": 1,
                    })
                    module.linear_weights["qweight"] = param_qweight
                    module.linear_weights["scales_zeros"] = param_scales_zeros
                    del _qzeros
                    del _scales
                    del param
                    delattr(module, "weight")
                    torch.cuda.empty_cache()

            else:  # load packed int4 weight
                module.linear_weights.pop("qweight")
                module.linear_weights.pop("qzeros")
                module.linear_weights.pop("scales")
                _qweight = module._parameters["qweight"]
                _qzeros = module._parameters["qzeros"]
                _scales = module._parameters["scales"]
                qweight, scales_zeros = convert_s4(_qweight.data, _qzeros.data,
                                                   _scales.data)
                param_qweight = Parameter(qweight, requires_grad=False)
                param_scales_zeros = Parameter(scales_zeros,
                                               requires_grad=False)
                del _qweight
                del _qzeros
                del _scales
                delattr(module, "qweight")
                delattr(module, "qzeros")
                delattr(module, "scales")
                module.register_parameter("qweight", param_qweight)
                module.register_parameter("scales_zeros", param_scales_zeros)
                set_weight_attrs(
                    param_qweight, {
                        "input_dim": 0,
                        "output_dim": 1,
                        "packed_dim": 1,
                        "pack_factor": quant_config.pack_factor,
                    })
                set_weight_attrs(param_scales_zeros, {
                    "input_dim": 0,
                    "output_dim": 1,
                })
                module.linear_weights["qweight"] = param_qweight
                module.linear_weights["scales_zeros"] = param_scales_zeros
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
