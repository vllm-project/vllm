from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from magic_wand import (
    MARLIN_SUPPORTED_NUM_BITS,
    MARLIN_SUPPORTED_GROUP_SIZES,
    MARLIN_TILE,
    MARLIN_MIN_THREAD_N,
    MARLIN_MIN_THREAD_K,
    MARLIN_MAX_PARALLEL,
    get_pack_factor,
)


class MarlinConfig(QuantizationConfig):
    """Config class for Marlin"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act

        # Verify
        if self.weight_bits not in MARLIN_SUPPORTED_NUM_BITS:
            raise ValueError(
                "Got weight_bits = {weight_bits}, but Marlin only supports {MARLIN_SUPPORTED_NUM_BITS}"
            )

        if self.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            raise ValueError(
                "Got group_size = {group_size}, but Marlin only supports {MARLIN_SUPPORTED_GROUP_SIZES}"
            )

        # Init
        self.pack_factor = get_pack_factor(weight_bits)
        self.tile_size = MARLIN_TILE
        self.min_thread_n = MARLIN_MIN_THREAD_N
        self.min_thread_k = MARLIN_MIN_THREAD_K
        self.max_parallel = MARLIN_MAX_PARALLEL

        # Permutation length used by the marlin kernels.
        self.perm_len = 1024

    def __repr__(self) -> str:
        return (f"MarlinConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act})")

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
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        return cls(weight_bits, group_size, desc_act)

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

        # Validate dtype
        if params_dtype != torch.float16:
            raise ValueError(
                f"The params dtype must be float16, but got {params_dtype}")

        # Validate output_size_per_partition
        if output_size_per_partition % self.quant_config.min_thread_n != 0:
            raise ValueError(
                f"Weight output_size_per_partition = {output_size_per_partition}"
                f" is not divisible by min_thread_n = {self.quant_config.min_thread_n}."
            )

        # Validate input_size_per_partition
        if input_size_per_partition % self.quant_config.min_thread_k != 0:
            raise ValueError(
                f"Weight input_size_per_partition = {input_size_per_partition}"
                f" is not divisible by min_thread_k = {self.quant_config.min_thread_k}."
            )

        if self.quant_config.group_size != -1:
            if input_size_per_partition % self.quant_config.group_size != 0:
                raise ValueError(
                    f"Weight input_size_per_partition = {input_size_per_partition}"
                    f" is not divisible by group_size = {self.quant_config.group_size}."
                )

        # Quantized weight
        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
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

        # Activation order
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

        # Detect sharding of scales/zp
        CONT CONT...
        # scale_and_zero_size = input_size // group_size
        # scale_and_zero_input_dim = None
        # if (input_size != input_size_per_partition
        #         and self.quant_config.group_size != -1):
        #     # For act-order models, we cannot use Exllama for row parallel layer
        #     if self.quant_config.desc_act:
        #         exllama_state = ExllamaState.UNUSED
        #     else:
        #         # we need to partition qzeros and scales for exllama kernel
        #         scale_and_zero_size = input_size_per_partition // group_size
        #         scale_and_zero_input_dim = 0

        # Quantized zeros
        qzeros = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
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
        # qweight = Parameter(
        #     torch.empty(
        #         input_size_per_partition // self.quant_config.tile_size,
        #         output_size_per_partition * self.quant_config.tile_size //
        #         self.quant_config.pack_factor,
        #         device="cuda",
        #         dtype=torch.int32,
        #     ),
        #     requires_grad=False,
        # )
        # set_weight_attrs(
        #     qweight,
        #     {
        #         "input_dim": 0,
        #         "output_dim": 1,
        #         "packed_dim": 1,
        #         "pack_factor": self.quant_config.pack_factor,
        #         "marlin_tile_size": self.quant_config.tile_size,
        #     },
        # )

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
        workspace = Parameter(
            torch.zeros(max_workspace_size, device="cuda", dtype=torch.int),
            requires_grad=False,
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

        x_2d = x.view(-1, x.shape[-1])

        size_m = x_2d.shape[0]
        size_k = x_2d.shape[1]
        size_n = scales.shape[1]

        output_2d = ops.marlin_gemm(x_2d, qweight, scales, workspace, size_m,
                                    size_n, size_k)

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1], ))

        if bias is not None:
            output.add_(bias)  # In-place add

        return output
