import enum
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig
from vllm.model_executor.layers.quantization.triton_utils.kernels import \
    QuantLinearInferenceOnlyFunction

try:
    import autogptq_cuda_64
    import autogptq_cuda_256
    _autogptq_cuda_available = True
except ImportError:
    logger.warning('CUDA extension not installed.')
    autogptq_cuda_256 = None
    autogptq_cuda_64 = None
    _autogptq_cuda_available = False


class GPTQLinearKernel(Enum):

    TRITON = enum.auto()
    EXLLAMA = enum.auto()
    CUDA = enum.auto()


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        use_triton: bool,
        disable_exllama: bool
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.pack_factor = 32 // self.weight_bits
        self.use_triton = use_triton
        self.disable_exllama = disable_exllama
        # Exllama kernel only supports 4 bitS, Exllama will be used if disable_exllama is False;
        # the Triton or CUDA kernel will be used for quantization precision other than 4 bit.
        if self.weight_bits in [2, 4, 8]:
            self.kernel_type = GPTQLinearKernel.TRITON if self.use_triton else GPTQLinearKernel.CUDA
            if self.weight_bits == 4:
                self.kernel_type = GPTQLinearKernel.EXLLAMA if not disable_exllama else self.kernel_type
        elif self.weight_bits == 3:
            self.kernel_type = GPTQLinearKernel.CUDA
        else:
            raise ValueError(
                "Currently, only 2, 3, 4, and 8-bit weight quantization is supported for"
                f"GPTQ, but got {self.weight_bits} bits.")
        self.maxq = 2 ** self.weight_bits - 1

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"use_triton={self.use_triton}, "
                f"disable_exllama={self.disable_exllama}")

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        use_triton = cls.get_from_keys(config, ["use_triton"])
        disable_exllama = cls.get_from_keys(config, ["disable_exllama"])
        return cls(weight_bits, group_size, desc_act, use_triton, disable_exllama)

    def get_linear_method(self) -> "GPTQLinearMethod":
        return GPTQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class ExllamaState(Enum):

    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        del output_size  # Unused.
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.kernel_type == GPTQLinearKernel.CUDA:
            if self.quant_config.weight_bits in [2, 4, 8]:
                self.wf = torch.tensor(list(range(0, 32, self.quant_config.weight_bits)), dtype=torch.int32).unsqueeze(0)
            elif self.quant_config.weight_bits == 3:
                self.wf = torch.tensor(
                    [
                        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                        [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                        [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                    ],
                    dtype=torch.int32
                ).reshape(1, 3, 12)
            self.autogptq_cuda_available = _autogptq_cuda_available

            self.autogptq_cuda = autogptq_cuda_256
            if input_size_per_partition % 256 != 0 or output_size_per_partition % 256 != 0:
                self.autogptq_cuda = autogptq_cuda_64
            if input_size_per_partition % 64 != 0 or output_size_per_partition % 64 != 0:
                self.autogptq_cuda_available = False
            
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if input_size != input_size_per_partition and self.quant_config.group_size != -1:
            # For act-order models, we cannot use Exllama for row parallel layer
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                # we need to partition qzeros and scales for exllama kernel
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            })
        g_idx = Parameter(
            torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(input_size_per_partition)
                ],
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(g_idx, {"input_dim": 0, "ignore_warning": True})
        qzeros = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": scale_and_zero_input_dim,
            "output_dim": 1,
        })
        return {
            "qweight": qweight,
            "g_idx": g_idx,
            "qzeros": qzeros,
            "scales": scales,
            "exllama_state": exllama_state, # when use_triton is true or quantization precision is not equal to 4-bit, exllama state will be ignored
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (weights["qweight"].shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        if self.quant_config.kernel_type == GPTQLinearKernel.EXLLAMA:
            # exllama needs to shuffle the weight after the weight is loaded
            # here we do the shuffle on first forward pass
            if weights["exllama_state"] == ExllamaState.UNINITIALIZED:
                if self.quant_config.desc_act:
                    weights["g_idx"] = torch.argsort(weights["g_idx"]).to(
                        torch.int)
                else:
                    weights["g_idx"] = torch.empty((1, 1), device="meta")
                weights["exllama_state"] = ExllamaState.READY
                ops.gptq_shuffle(weights["qweight"], weights["g_idx"])

            output = ops.gptq_gemm(reshaped_x, weights["qweight"],
                                weights["qzeros"], weights["scales"],
                                weights["g_idx"],
                                weights["exllama_state"] == ExllamaState.READY)
        elif self.quant_config.kernel_type == GPTQLinearKernel.TRITON:
            quant_linear_fn = QuantLinearInferenceOnlyFunction
            output = quant_linear_fn.apply(
                reshaped_x,
                weights["qweight"],
                weights["scales"],
                weights["qzeros"],
                weights["g_idx"],
                self.quant_config.weight_bits,
                self.quant_config.maxq
            )
            output = output.half().reshape(out_shape)
        else:
            self.kernel_switch_threshold = 128
            if reshaped_x.device.type == "cuda" and self.autogptq_cuda_available and (
                self.kernel_switch_threshold == 0 or reshaped_x.shape[0] < self.kernel_switch_threshold
            ):
                output = torch.zeros(out_shape, device=reshaped_x.device, dtype=torch.float32)
                if self.quant_config.weight_bits == 2:
                    self.autogptq_cuda.vecquant2matmul(reshaped_x.float(), weights["qweight"], output, weights["scales"].float(), weights["qzeros"], weights["g_idx"])
                elif self.quant_config.weight_bits == 3:
                    self.autogptq_cuda.vecquant3matmul(reshaped_x.float(), weights["qweight"], output, weights["scales"].float(), weights["qzeros"], weights["g_idx"])
                elif self.quant_config.weight_bits == 4:
                    self.autogptq_cuda.vecquant4matmul(reshaped_x.float(), weights["qweight"], output, weights["scales"].float(), weights["qzeros"], weights["g_idx"])
                elif self.quant_config.weight_bits == 8:
                    self.autogptq_cuda.vecquant8matmul(reshaped_x.float(), weights["qweight"], output, weights["scales"].float(), weights["qzeros"], weights["g_idx"])
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")
            else:
                if self.wf.device != weights["qzeros"].device:
                    self.wf = self.wf.to(weights["qzeros"].device)

                if self.quant_config.weight_bits in [2, 4, 8]:
                    zeros = torch.bitwise_right_shift(
                        torch.unsqueeze(weights["qzeros"], 2).expand(-1, -1, 32 // self.quant_config.weight_bits),
                        self.wf.unsqueeze(0)
                    ).to(torch.int16 if self.quant_config.weight_bits == 8 else torch.int8)
                    zeros = torch.bitwise_and(zeros, (2 ** self.quant_config.weight_bits) - 1)

                    zeros = zeros + 1
                    zeros = zeros.reshape(weights["scales"].shape)

                    weight = torch.bitwise_right_shift(
                        torch.unsqueeze(weights["qweight"], 1).expand(-1, 32 // self.quant_config.weight_bits, -1),
                        self.wf.unsqueeze(-1)
                    ).to(torch.int16 if self.quant_config.weight_bits == 8 else torch.int8)
                    weight = torch.bitwise_and(weight, (2 ** self.quant_config.weight_bits) - 1)
                elif self.quant_config.weight_bits == 3:
                    zeros = weights["qzeros"].reshape(
                        weights["qzeros"].shape[0], weights["qzeros"].shape[1] // 3, 3, 1
                    ).expand(-1, -1, -1, 12)
                    zeros = (zeros >> self.wf.unsqueeze(0))
                    zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
                    zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
                    zeros = zeros & 0x7
                    zeros = torch.cat([zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]], dim=2)

                    zeros = zeros + 1
                    zeros = zeros.reshape(weights["scales"].shape)

                    weight = weights["qweight"].reshape(
                        weights["qweight"].shape[0] // 3, 3, 1, weights["qweight"].shape[1]
                    ).expand(-1, -1, 12, -1)
                    weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
                    weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
                    weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
                    weight = weight & 0x7
                    weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")

                weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
                num_itr = weights["g_idx"].shape[0]//x.shape[-1]
                if num_itr == 1:
                    weights = (weights["scales"][weights["g_idx"].long()] * (weight - zeros[weights["g_idx"].long()]))
                else:
                    num_dim = weights["g_idx"].shape[0]//num_itr
                    weights = []
                    for i in range(num_itr):
                        scale_i = weights["scales"][:,i*num_dim:(i+1)*num_dim]
                        weight_i = weight[:,i*num_dim:(i+1)*num_dim]
                        zeros_i = zeros[:,i*num_dim:(i+1)*num_dim]
                        g_idx_i = weights["g_idx"][i*num_dim:(i+1)*num_dim]
                        weights.append(scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()]))
                    weights = torch.cat(weights,dim=1)
                output = torch.matmul(x, weights)
        output = output.to(x.dtype)
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)
