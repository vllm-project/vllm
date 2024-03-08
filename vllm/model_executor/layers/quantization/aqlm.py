# Supports AQLM compression, see https://github.com/Vahe1994/AQLM and https://arxiv.org/pdf/2401.06118.pdf

from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.int8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    if nbits <= 64:
        return torch.int64
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")


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

        # out_group_size > 1 is untested, and probably won't work as-is.
        assert (self.out_group_size == 1)
        self.pack_factor = (self.in_group_size * self.out_group_size)

    def __repr__(self) -> str:
        return (f"AQLMConfig(in_group_size={self.in_group_size}, "
                f"nbits_per_codebook={self.nbits_per_codebook}, "
                f"num_codebooks={self.num_codebooks}, "
                f"out_group_size={self.out_group_size})")

    @classmethod
    def get_name(cls) -> str:
        return "aqlm"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AQLMConfig":
        in_group_size = cls.get_from_keys(config, ["in_group_size"])
        nbits_per_codebook = cls.get_from_keys(config, ["nbits_per_codebook"])
        num_code_books = cls.get_from_keys(config, ["num_codebooks"])
        out_group_size = cls.get_from_keys(config, ["out_group_size"])
        return cls(in_group_size, nbits_per_codebook, num_code_books,
                   out_group_size)

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
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        del output_size  # Unused.
        del input_size  # Unused.

        if params_dtype != torch.half:
            raise ValueError("Only half is currently supported by aqlm")
        if input_size_per_partition % self.quant_config.in_group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.out_group_size != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        codes = Parameter(
            torch.empty(
                # There could actually be two pack factors, one along input and one along output,
                # but we don't currently support out_group_size,
                # and only the one along output needs to be marked with "packed_dim".
                # in order for QKVLinear to work.
                output_size_per_partition,
                input_size_per_partition // self.quant_config.pack_factor,
                self.quant_config.num_codebooks,
                dtype=get_int_dtype(self.quant_config.nbits_per_codebook),
            ),
            requires_grad=False,
        )

        set_weight_attrs(
            codes,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        codebooks = Parameter(
            torch.empty(
                self.quant_config.num_codebooks * len(output_partition_sizes),
                2**self.quant_config.nbits_per_codebook,
                self.quant_config.out_group_size,
                self.quant_config.in_group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            codebooks,
            {
                # metadata indicates fixed size concatenated along dim 0
                "is_metadata":
                True,
                "output_partition_sizes":
                torch.tensor(output_partition_sizes, device='cpu'),
            },
        )

        scales = Parameter(
            torch.empty(
                (
                    output_size_per_partition //
                    self.quant_config.out_group_size,
                    1,
                    1,
                    1,
                ),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "output_dim": 0,
                "packed_dim": 0,
                "pack_factor": self.quant_config.out_group_size
            },
        )

        return {
            "codes": codes,
            "codebooks": codebooks,
            "scales": scales,
        }

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        codebooks = weights["codebooks"]
        codes = weights["codes"]
        scales = weights["scales"]
        output_partition_sizes = getattr(codebooks, "output_partition_sizes",
                                         None)

        output = ops.aqlm_gemm(
            x,
            codes,
            codebooks,
            scales,
            output_partition_sizes,
            bias,
        )

        return output
