from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class MarlinConfig(QuantizationConfig):
    """Config class for Marlin.

    Reference: https://github.com/IST-DASLab/marlin/tree/master
    """

    def __init__(
        self,
        group_size: int,
    ) -> None:
        # Group size for the quantization.
        self.group_size = group_size
        if self.group_size != 128 and self.group_size != -1:
            raise ValueError(
                 "Currently, only group size 128 and -1 (channelwise) is supported for "
                f"Marlin, but got group_size of {self.group_size}")

        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = 32 // 4
        # Tile size used by marlin kernels.
        self.tile_size = 16
        # Data for workspace.
        self.max_parallel = 16
        self.min_n_threads = 128
        # Permutation length use by the marlin kernels.
        self.perm_len = 1024

    def __repr__(self) -> str:
        return f"MarlinConfig(group_size={self.group_size}"

    @classmethod
    def get_name(cls) -> str:
        return "marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MarlinConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(group_size)

    def get_linear_method(self) -> "MarlinLinearMethod":
        return MarlinLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

class MarlinLinearMethod(LinearMethodBase):
    """Linear method for Marlin.

    Args:
        quant_config: The Marlin quantization config.
    """

    def __init__(self, quant_config: MarlinConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        # del output_size  # Unused.
        if params_dtype != torch.float16:
            raise ValueError(
                f"The params dtype must be float16, but got {params_dtype}")
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
        if input_size_per_partition % 128 != 0:
            raise ValueError(
                "The input_size_per_partition must be divisible by 128, "
                f"but got {input_size_per_partition}")
        if output_size_per_partition % 256 != 0:
            raise ValueError(
                "The output_size_per_partition must be divisible by 256, "
                f"but got {output_size_per_partition}")
        if output_size_per_partition % self.quant_config.min_n_threads != 0:
            raise ValueError(
                "The output_size per partition must be divisible by the minimum "
                f"number of threads {self.quant_config.min_n_threads}, but got {output_size_per_partition}"
            )

        # check that we have at least 4 tiles horizontally in the shard
        num_tiles_per_perm = self.quant_config.perm_len // (self.quant_config.tile_size**2)
        if output_size_per_partition % num_tiles_per_perm != 0:
            raise ValueError(
                "Each permutation group must reside on the same gpu")

        # Quantized 4Bit weights packed into Int32.
        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.tile_size,
                output_size_per_partition * self.quant_config.tile_size //
                self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
                "marlin_tile_size": self.quant_config.tile_size,
            },
        )

        # Scales in Float16.
        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = input_size
        scales = Parameter(
            torch.empty(
                input_size_per_partition // group_size,
                output_size_per_partition,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim":
                None if input_size == input_size_per_partition else 0,
                "output_dim": 1,
            },
        )

        max_workspace_size = (output_size_per_partition // self.quant_config.min_n_threads) * self.quant_config.max_parallel
        workspace = Parameter(
            torch.zeros(
                max_workspace_size,
                device="cuda",
                dtype=torch.int
            ),
            requires_grad=False
        )

        return {
            "B": qweight,
            "s": scales,
            "workspace": workspace,
        }

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = weights["B"]
        scales = weights["s"]
        workspace = weights["workspace"]

        output = torch.empty(x.shape[:-1] + (scales.shape[1], ),
                             dtype=x.dtype,
                             device=x.device)

        ops.marlin_gemm(
            x.view(-1, x.shape[-1]),
            qweight,
            output.view(-1, output.shape[-1]),
            scales,
            workspace,
        )

        if bias is not None:
            output.add_(bias)

        return output
