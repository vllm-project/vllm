from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

GPTQ_MARLIN_24_TILE = 16
GPTQ_MARLIN_24_MIN_THREAD_N = 128
GPTQ_MARLIN_24_MIN_THREAD_K = 128
GPTQ_MARLIN_24_MAX_PARALLEL = 64

GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8, scalar_types.uint8b128
]
GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES = [-1, 128]


class GPTQMarlin24Config(QuantizationConfig):
    """Config class for Marlin24.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
    ) -> None:
        quant_type = {
            4: scalar_types.uint4b8,
            8: scalar_types.uint8b128,
        }.get(weight_bits)

        self.group_size = group_size

        # Verify
        if quant_type is None or \
            quant_type not in GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES:
            raise ValueError(
                f"Marlin_24 does not support quant_type = {quant_type}. "
                f"Only weight_bits = {GPTQ_MARLIN_24_SUPPORTED_QUANT_TYPES} "
                "are supported.")
        if self.group_size not in GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Marlin_24 does not support group_size = {self.group_size}. "
                f"Only group_sizes = {GPTQ_MARLIN_24_SUPPORTED_GROUP_SIZES} "
                "are supported.")

        self.quant_type = quant_type

        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = 32 // self.quant_type.size_bits

        # Tile size used by marlin kernels.
        self.tile_size = 16

        # Min out_features dim
        self.min_n_threads = GPTQ_MARLIN_24_MIN_THREAD_N

        # Min in_features dim
        self.min_k_threads = GPTQ_MARLIN_24_MIN_THREAD_K

        # Max parallel problems to solve at once (improves large
        # batch performance)
        self.max_parallel = GPTQ_MARLIN_24_MAX_PARALLEL

        # Permutation length used by the marlin kernels.
        self.perm_len = 1024

    def __repr__(self) -> str:
        return "Marlin24Config(quant_type={}, group_size={})".format(
            self.quant_type, self.group_size)

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin_24"

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
    def from_config(cls, config: Dict[str, Any]) -> "GPTQMarlin24Config":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        is_marlin_24_format = (
            hf_quant_cfg.get("checkpoint_format") == "marlin_24")

        is_valid_user_quant = (user_quant is None or user_quant == "gptq"
                               or user_quant == "gptq_marlin_24")

        if is_marlin_24_format and is_valid_user_quant:
            msg = ("The model is serialized in {} format. "
                   "Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["GPTQMarlin24LinearMethod"]:
        if isinstance(layer, LinearBase):
            return GPTQMarlin24LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class GPTQMarlin24LinearMethod(LinearMethodBase):
    """Linear method for Marlin24.

    Args:
        quant_config: The Marlin24 quantization config.
    """

    def __init__(self, quant_config: GPTQMarlin24Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.

        if params_dtype != torch.float16:
            raise ValueError(
                f"The params dtype must be float16, but got {params_dtype}")

        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.min_n_threads != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"min_n_threads = {self.quant_config.min_n_threads}.")
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"pack_factor = {self.quant_config.pack_factor}.")

        # Validate input_size_per_partition
        if input_size_per_partition % self.quant_config.min_k_threads != 0:
            raise ValueError(
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible by "
                f"min_k_threads = {self.quant_config.min_k_threads}.")
        if (self.quant_config.group_size != -1 and
                input_size_per_partition % self.quant_config.group_size != 0):
            raise ValueError(f"Weight input_size_per_partition = "
                             f"{input_size_per_partition} is not divisible by "
                             f"group_size = {self.quant_config.group_size}.")

        # Check that we have at least 4 tiles horizontally in the shard
        num_tiles_per_perm = self.quant_config.perm_len // (
            self.quant_config.tile_size**2)
        if output_size_per_partition % num_tiles_per_perm != 0:
            raise ValueError(
                "Each permutation group must reside on the same gpu")

        # Quantized 4Bit weights packed into Int32.
        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.tile_size // 2,
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

        # Meta
        meta = Parameter(
            torch.empty(
                input_size_per_partition // 8 // 2 // 2,
                output_size_per_partition * 2,
                device="cuda",
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            meta,
            {
                "input_dim": 0,
                "packed_dim": 1,
                "pack_factor": 1,
                "output_dim": 1,
                "marlin_tile_size": 2,
            },
        )

        # Determine if channelwise or not
        input_groups = (1 if self.quant_config.group_size == -1 else
                        input_size_per_partition //
                        self.quant_config.group_size)

        scales = Parameter(
            torch.empty(
                input_groups,
                output_size_per_partition,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": None if input_groups == 1 else 0,
                "output_dim": 1,
            },
        )

        # Allocate workspace (Used for internal locking mechanism)
        max_workspace_size = (
            output_size_per_partition //
            self.quant_config.min_n_threads) * self.quant_config.max_parallel
        workspace = Parameter(torch.zeros(max_workspace_size,
                                          device="cuda",
                                          dtype=torch.int),
                              requires_grad=False)

        layer.register_parameter("B_24", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("B_meta", meta)
        set_weight_attrs(meta, extra_weight_attrs)
        layer.register_parameter("s", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("workspace", workspace)
        set_weight_attrs(workspace, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.B_24
        meta = layer.B_meta
        scales = layer.s
        workspace = layer.workspace

        x_2d = x.view(-1, x.shape[-1])

        size_m = x_2d.shape[0]
        size_k = x_2d.shape[1]
        size_n = scales.shape[1]

        output_2d = ops.gptq_marlin_24_gemm(x_2d, qweight, meta, scales,
                                            workspace,
                                            self.quant_config.quant_type,
                                            size_m, size_n, size_k)

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1], ))

        if bias is not None:
            output.add_(bias)  # In-place add

        return output
