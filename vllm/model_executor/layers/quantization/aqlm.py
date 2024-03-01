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

        # I think pack factor is *probably* how many elements fit into one quantized tensor element.
        # though out group size makes it interesting, because really we are doing 2D blocks, potentially.
        # maybe this is vllms first 2D packing?  Arg.
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
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

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

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       output_sizes: List[int]) -> Dict[str, Any]:
        assert (output_size == output_size_per_partition)
        assert (input_size == input_size_per_partition)
        del output_size  # Unused.
        del input_size  # Unused.

        if params_dtype != torch.half:
            raise ValueError("Only half is currently supported by aqlm")
        if input_size_per_partition % self.quant_config.in_group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size_per_partition % self.quant_config.out_group_size != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        codes = Parameter(
            torch.empty(
                output_size_per_partition,  # not entirely sure what to do with num_out_groups, if we need this pack factor.
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
                "output_sizes": output_sizes
            },
        )

        codebooks = Parameter(
            torch.empty(
                self.quant_config.num_codebooks * len(output_sizes),
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
                "shard_dim": 0,
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

        shard_dim = getattr(codebooks, "shard_dim", None)
        if shard_dim is not None:
            output_shape = x.shape[:-1] + (scales.shape[0], )
            output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            output_sizes = getattr(codes, "output_sizes", None)
            outputs = len(output_sizes)

            # break the shards apart and combine them.
            assert (shard_dim == 0)
            num_codebooks = codebooks.shape[shard_dim] // outputs

            assert (scales.shape[0] == codes.shape[0])
            assert (scales.shape[0] == sum(output_sizes))

            output_offset = 0
            codebooks_offset = 0
            for output_size in output_sizes:
                shard_output = ops.aqlm_gemm(
                    x, codes.narrow(0, output_offset, output_size),
                    codebooks.narrow(shard_dim, codebooks_offset,
                                     num_codebooks),
                    scales.narrow(0, output_offset, output_size),
                    None if bias is None else bias.narrow(
                        0, output_offset, output_size))

                output_slice = output.narrow(-1, output_offset, output_size)
                assert (output_slice.shape == shard_output.shape)
                output_slice.copy_(shard_output)
                output_offset += output_size
                codebooks_offset += num_codebooks

            return output

        output = ops.aqlm_gemm(
            x,
            codes,
            codebooks,
            scales,
            bias,
        )

        return output
