from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter
import intel_extension_for_pytorch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point})")

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["AWQLinearMethod"]:
        if isinstance(layer, LinearBase):
            return AWQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
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

    def awq_reverse_reorder_int_tensor(self,int_tensor, bits: int):
        assert bits == 4

        int_tensor = int_tensor.T.contiguous()
        compress_ratio = (32 // bits)
        assert int_tensor.shape[-1] % compress_ratio == 0

        order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        order_tensor = torch.tensor(
            order_map, dtype=torch.int32, device=int_tensor.device).reshape(1, -1)
        order_tensor = order_tensor.repeat(
            int_tensor.shape[1]//compress_ratio, 1)
        order_tensor = order_tensor + torch.arange(0, int_tensor.shape[1],
                                                    compress_ratio, dtype=torch.int32, device=int_tensor.device).reshape(-1, 1)
        order_tensor = order_tensor.reshape(-1)

        reverse_order_tensor = torch.arange(order_tensor.shape[0])[order_tensor]
        reverse_order_tensor = reverse_order_tensor[order_tensor]
        int_tensor = int_tensor[:, reverse_order_tensor]
        return int_tensor
    def unpack_awq(self, awq_qweight: torch.Tensor, awq_qzeros: torch.Tensor, awq_scales: torch.Tensor, bits: int, group_size: int):
        """
        Args:
            awq_qweight (`torch.LongTensor`):
                Expected shape: (in_features, out_features // (32 // bits))
            awq_qzeros (`torch.LongTensor`):
                Expected shape: (in_features // group_size, out_features // (32 // bits))
            awq_scales (`torch.LongTensor`):
                Expected shape: (in_features // group_size, out_features)

        Returns:
            fp16_weight (`torch.LongTensor`):
                With shape (in_features, out_features).
            zeros (`torch.LongTensor`):
                With shape (in_features // group_size, out_features).
        """
        assert bits == 4

        qzeros = awq_qzeros
        qweight = awq_qweight
        qweight = qweight.T.contiguous()

        scales = awq_scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        infeatures = awq_qweight.shape[0]

        wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device).unsqueeze(0)
        zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
            torch.int16 if bits == 8 else torch.int8)

        #zeros = zeros + 1

        torch.bitwise_and(zeros, (2 ** bits) - 1, out=zeros)

        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        weight = torch.bitwise_right_shift(torch.unsqueeze(
            qweight, 1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
        weight = weight.reshape(-1, group_size, weight.shape[2])

        weight = weight.view(-1, weight.shape[-1])
        zeros = zeros.view(-1, zeros.shape[-1])

        zeros = zeros.T.contiguous()
        zeros = self.awq_reverse_reorder_int_tensor(zeros, bits)
        weight = self.awq_reverse_reorder_int_tensor(weight, bits)

        return weight.contiguous(), zeros.contiguous()


    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])

        if not hasattr(self,"_op_context") :
            t, zp_x = self.unpack_awq(qweight, qzeros, scales, 4, 128)
            # # transpose -> [N, K]
            t = t.T.contiguous()
            qweight_ = t[:, 1::2].bitwise_left_shift(4).bitwise_or_(t[:, ::2]).to(torch.uint8)
            scales_ = scales.t().contiguous()
            self._op_context = torch.ops.ipex_prepack.weight_only_qlinear_prepack_int4(
                qweight_,
                scales_,
                zp_x.t_().contiguous(),
                bias,
                None,
                None,
                128,
                2, # 2 for bf16 compute, 3 for int8 compute
                1,
            )

        out = torch.ops.torch_ipex.ipex_woq_linear(reshaped_x, self._op_context.get_data_handle())

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
