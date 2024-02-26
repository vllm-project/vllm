import enum
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class AQLMConfig(QuantizationConfig):
    """Config class for AQLM.

    Reference: https://github.com/Vahe1994/AQLM
    """

    def __init__(
        self,
        in_group_size: int,
        nbits_per_codebook: int,
        num_codebooks: int,
        out_group_size: int,
    ) -> None:
        self.in_group_size = in_group_size
        self.nbits_per_codebook = nbits_per_codebook
        self.num_codebooks = num_codebooks
        self.out_group_size = out_group_size
        # self.pack_factor = 32 // self.weight_bits
        # exllama kernel v1 only supports 4 bit
        # if self.weight_bits != 4:
        # raise ValueError(
        # "Currently, only 4-bit weight quantization is supported for "
        # f"GPTQ, but got {self.weight_bits} bits."
        # )

    def __repr__(self) -> str:
        return (
            f"AQLMConfig(in_group_size={self.in_group_size}, "
            f"nbits_per_codebook={self.nbits_per_codebook}, "
            f"num_codebooks={self.num_codebooks}, "
            f"out_group_size={self.out_group_size})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "aqlm"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    # such as.  (This one looks correct)
    # https://huggingface.co/BlackSamorez/Llama-2-7b-AQLM-2Bit-1x16-hf/blob/main/config.json
    #
    # "quantization_config": {
    #   "in_group_size": 8,
    #   "nbits_per_codebook": 16,
    #   "num_codebooks": 1,
    #   "out_group_size": 1,

    #   "linear_weights_not_to_quantize": [ <--- hmmm ????
    #       "model.embed_tokens.weight",
    #       "lm_head.weight"

    # "quant_method": "aqlm" duh <- shows it's aqlm.  Do we auto-detect?  How?
    # },

    #https://huggingface.co/meta-llama/Llama-2-7b-hf 

    # this one looks non-standard, has no quantization_config, just an AQLM block.
    # https://huggingface.co/BlackSamorez/Llama-2-70b-AQLM-4Bit-2x16-hf/blob/main/config.json
    # "aqlm": {
    #    "in_group_size": 8,
    #    "nbits_per_codebook": 16,
    #    "num_codebooks": 2,
    # "   "out_group_size": 1

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AQLMConfig":
        in_group_size = cls.get_from_keys(config, ["in_group_size"])
        nbits_per_codebook = cls.get_from_keys(config, ["nbits_per_codebook"])
        num_code_books = cls.get_from_keys(config, ["num_codebooks"])
        out_group_size = cls.get_from_keys(config, ["out_group_size"])
        # TODO linear_weights_not_to_quantize ?
        return cls(in_group_size, nbits_per_codebook, num_code_books, out_group_size)

    def get_linear_method(self) -> "AQLMLinearMethod":
        return AQLMLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class AQLMLinearMethod(LinearMethodBase):
    """Linear method for AQLM.

    Args:
        quant_config: The AQLM quantization config.
    """

    def __init__(self, quant_config: AQLMConfig):
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
                "tensor parallel size."
            )
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None

        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )
        g_idx = Parameter(
            torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(input_size_per_partition)
                ],
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
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )
        scales = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
            },
        )
        return {
            "qweight": qweight,
            "g_idx": g_idx,
            "qzeros": qzeros,
            "scales": scales
        }

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = weights["qweight"]
        out_shape = x.shape[:-1] + (qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1])
        output = ops.aqlm_gemm(
            reshaped_x,
            weights["qweight"],
            weights["qzeros"],
            weights["scales"],
            weights["g_idx"],
        )
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)
