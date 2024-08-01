from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)

MARLIN_QQQ_TILE = 16
MARLIN_QQQ_MIN_THREAD_N = 64
MARLIN_QQQ_MIN_THREAD_K = 128
MARLIN_QQQ_MAX_PARALLEL = 16

MARLIN_QQQ_SUPPORTED_NUM_BITS = [4]
MARLIN_QQQ_SUPPORTED_GROUP_SIZES = [-1, 128]
MARLIN_QQQ_SUPPORTED_SYM = [True]


class QQQConfig(QuantizationConfig):
    """Config class for QQQ
    
    Reference: https://arxiv.org/pdf/2406.09904
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        is_sym: bool = True,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.is_sym = is_sym

        # Verify
        if self.weight_bits not in MARLIN_QQQ_SUPPORTED_NUM_BITS:
            raise ValueError(
                f"QQQ does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {MARLIN_QQQ_SUPPORTED_NUM_BITS} "
                "are supported.")
        if self.group_size not in MARLIN_QQQ_SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"QQQ does not support group_size = {self.group_size}. "
                f"Only group_sizes = {MARLIN_QQQ_SUPPORTED_GROUP_SIZES} "
                "are supported.")
        if self.is_sym not in MARLIN_QQQ_SUPPORTED_SYM:
            raise ValueError(
                f"QQQ does not support is_sym = {self.is_sym}. "
                f"Only sym = {MARLIN_QQQ_SUPPORTED_SYM} are supported.")

        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = 32 // self.weight_bits

        # Tile size used by QQQ kernels.
        self.tile_size = MARLIN_QQQ_TILE

        # Min out_features dim
        self.min_n_threads = MARLIN_QQQ_MIN_THREAD_N

        # Min in_features dim
        self.min_k_threads = MARLIN_QQQ_MIN_THREAD_K

        # Max parallel problems to solve at once (improves large
        # batch performance)
        self.max_parallel = MARLIN_QQQ_MAX_PARALLEL

        # Permutation length used by the QQQ kernels.
        self.perm_len = 1024

    def __repr__(self) -> str:
        return "QQQConfig(weight_bits={}, group_size={})".format(
            self.weight_bits, self.group_size)

    @classmethod
    def get_name(cls) -> str:
        return "qqq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QQQConfig":
        weight_bits = cls.get_from_keys(config, ["wbits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QQQLinearMethod"]:
        if isinstance(layer, LinearBase):
            return QQQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class QQQLinearMethod(LinearMethodBase):
    """Linear method for QQQ.

    Args:
        quant_config: The QQQ quantization config.
    """

    def __init__(self, quant_config: QQQConfig):
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

        s_channel = Parameter(
            torch.empty(
                1,
                output_size_per_partition,
                device="cuda",
                dtype=torch.float,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            s_channel,
            {
                "input_dim": None,
                "output_dim": 1,
            },
        )

        if self.quant_config.group_size == -1:
            s_group = Parameter(
                torch.tensor(
                    [],
                    device="cuda",
                    dtype=torch.half,
                ),
                requires_grad=False,
            )
        else:
            s_group = Parameter(
                torch.empty(
                    input_size_per_partition // self.quant_config.group_size,
                    output_size_per_partition,
                    device="cuda",
                    dtype=torch.half,
                ),
                requires_grad=False,
            )

        set_weight_attrs(
            s_group,
            {
                "input_dim": None if self.quant_config.group_size == -1 else 0,
                "output_dim":
                None if self.quant_config.group_size == -1 else 1,
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

        layer.register_parameter("B", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("s_channel", s_channel)
        set_weight_attrs(s_channel, extra_weight_attrs)
        layer.register_parameter("s_group", s_group)
        set_weight_attrs(s_group, extra_weight_attrs)
        layer.register_parameter("workspace", workspace)
        set_weight_attrs(workspace, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.B
        s_ch = layer.s_channel
        s_group = layer.s_group
        workspace = layer.workspace

        x_2d = x.view(-1, x.shape[-1])

        size_m = x_2d.shape[0]
        size_k = x_2d.shape[1]
        size_n = s_ch.shape[1]

        x_int8, s_tok = ops.scaled_int8_quant(x_2d)

        output_2d = ops.marlin_qqq_gemm(x_int8, qweight, s_tok, s_ch, s_group,
                                        workspace, size_m, size_n, size_k)

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1], ))

        if bias is not None:
            output.add_(bias)  # In-place add

        return output
