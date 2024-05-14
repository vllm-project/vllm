import enum
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

GPTQ_MARLIN_SUPPORTED_NUM_BITS = [4, 8]
GPTQ_MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
GPTQ_MARLIN_SUPPORTED_SYM = [True]


# Permutations for Marlin scale shuffling
def get_scale_perms(num_bits):
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def get_pack_factor(num_bits):
    assert (num_bits in GPTQ_MARLIN_SUPPORTED_NUM_BITS
            ), f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def marlin_permute_scales(s, size_k, size_n, group_size, num_bits):
    scale_perm, scale_perm_single = get_scale_perms(num_bits)
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    def __init__(self, weight_bits: int, group_size: int, desc_act: bool,
                 is_sym: bool) -> None:
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.is_sym = is_sym

        # Verify
        if self.weight_bits not in GPTQ_MARLIN_SUPPORTED_NUM_BITS:
            raise ValueError(
                f"Marlin does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {GPTQ_MARLIN_SUPPORTED_NUM_BITS} "
                "are supported.")
        if self.group_size not in GPTQ_MARLIN_SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Marlin does not support group_size = {self.group_size}. "
                f"Only group_sizes = {GPTQ_MARLIN_SUPPORTED_GROUP_SIZES} "
                "are supported.")
        if self.is_sym not in GPTQ_MARLIN_SUPPORTED_SYM:
            raise ValueError(
                f"Marlin does not support is_sym = {self.is_sym}. "
                f"Only sym = {GPTQ_MARLIN_SUPPORTED_SYM} are supported.")

        # Init
        self.pack_factor = get_pack_factor(weight_bits)
        self.tile_size = GPTQ_MARLIN_TILE
        self.min_thread_n = GPTQ_MARLIN_MIN_THREAD_N
        self.min_thread_k = GPTQ_MARLIN_MIN_THREAD_K
        self.max_parallel = GPTQ_MARLIN_MAX_PARALLEL

    def __repr__(self) -> str:
        return (f"GPTQMarlinConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQMarlinConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        return cls(weight_bits, group_size, desc_act, is_sym)

    def get_quant_method(
            self,
            layer: torch.nn.Module) -> Optional["GPTQMarlinLinearMethod"]:
        if isinstance(layer, LinearBase):
            return GPTQMarlinLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_marlin_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        num_bits = quant_config.get("bits", None)
        group_size = quant_config.get("group_size", None)
        sym = quant_config.get("sym", None)
        desc_act = quant_config.get("desc_act", None)

        # If we cannot find the info needed in the config, cannot convert.
        if (num_bits is None or group_size is None or sym is None
                or desc_act is None):
            return False

        # If the capability of the device is too low, cannot convert.
        major, minor = torch.cuda.get_device_capability()
        device_capability = major * 10 + minor
        if device_capability < cls.get_min_capability():
            return False

        # Otherwise, can convert if model satisfies marlin constraints.
        return (num_bits in GPTQ_MARLIN_SUPPORTED_NUM_BITS
                and group_size in GPTQ_MARLIN_SUPPORTED_GROUP_SIZES
                and sym in GPTQ_MARLIN_SUPPORTED_SYM)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class GPTQMarlinLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Marlin.

    Args:
        quant_config: The GPTQ Marlin quantization config.
    """

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
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
    ) -> None:
        del output_size

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # Validate dtype
        if params_dtype != torch.float16:
            raise ValueError(
                f"The params dtype must be float16, but got {params_dtype}")

        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.min_thread_n != 0:
            raise ValueError(
                f"Weight output_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f" min_thread_n = {self.quant_config.min_thread_n}.")

        # Validate input_size_per_partition
        if input_size_per_partition % self.quant_config.min_thread_k != 0:
            raise ValueError(
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible "
                f"by min_thread_k = {self.quant_config.min_thread_k}.")

        if (group_size < input_size
                and input_size_per_partition % group_size != 0):
            raise ValueError(
                f"Weight input_size_per_partition = {input_size_per_partition}"
                f" is not divisible by group_size = {group_size}.")

        # Detect sharding of scales/zp

        # By default, no sharding over "input dim"
        scales_and_zp_size = input_size // group_size
        scales_and_zp_input_dim = None

        if self.quant_config.desc_act:
            # Act-order case
            assert self.quant_config.group_size != -1

            is_k_full = input_size_per_partition == input_size

        else:
            # No act-order case

            # K is always full due to full alignment with
            # group-size and shard of scales/zp
            is_k_full = True

            # If this is a row-parallel case, then shard scales/zp
            if (input_size != input_size_per_partition
                    and self.quant_config.group_size != -1):
                scales_and_zp_size = input_size_per_partition // group_size
                scales_and_zp_input_dim = 0

        # Init buffers

        # Quantized weights
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
                **extra_weight_attrs,
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        # Activation order
        g_idx = Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(
            g_idx,
            {
                **extra_weight_attrs, "input_dim": 0,
                "ignore_warning": True
            },
        )

        g_idx_sort_indices = Parameter(
            torch.empty(
                g_idx.shape,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(g_idx_sort_indices, extra_weight_attrs)

        # Scales
        scales = Parameter(
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                **extra_weight_attrs,
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
            },
        )

        # Quantized zero-points
        qzeros = Parameter(
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
                device="meta",
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                **extra_weight_attrs,
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        # Allocate marlin workspace
        max_workspace_size = (
            output_size_per_partition //
            self.quant_config.min_thread_n) * self.quant_config.max_parallel
        workspace = torch.zeros(max_workspace_size,
                                dtype=torch.int,
                                requires_grad=False)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("g_idx_sort_indices", g_idx_sort_indices)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
        layer.workspace = workspace
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.is_k_full = is_k_full
        layer.marlin_state = GPTQMarlinState.REPACK

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])

        size_m = reshaped_x.shape[0]
        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition
        full_size_k = layer.input_size

        out_shape = x.shape[:-1] + (part_size_n, )

        if layer.marlin_state == GPTQMarlinState.REPACK:
            layer.marlin_state = GPTQMarlinState.READY

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def replace_tensor(name, new_t):
                # It is important to use resize_() here since it ensures
                # the same buffer is reused
                getattr(layer, name).resize_(new_t.shape)
                getattr(layer, name).copy_(new_t)
                del new_t

            cur_device = layer.qweight.device

            # Process act_order
            if self.quant_config.desc_act:
                # Get sorting based on g_idx
                g_idx_sort_indices = torch.argsort(layer.g_idx).to(torch.int)

                sorted_g_idx = layer.g_idx[g_idx_sort_indices]

                replace_tensor("g_idx", sorted_g_idx)
                replace_tensor("g_idx_sort_indices", g_idx_sort_indices)

            else:
                # Reset g_idx related tensors
                layer.g_idx = Parameter(
                    torch.empty(0, dtype=torch.int, device=cur_device),
                    requires_grad=False,
                )
                layer.g_idx_sort_indices = Parameter(
                    torch.empty(0, dtype=torch.int, device=cur_device),
                    requires_grad=False,
                )

            # Repack weights
            marlin_qweight = ops.gptq_marlin_repack(
                layer.qweight,
                layer.g_idx_sort_indices,
                part_size_k,
                part_size_n,
                self.quant_config.weight_bits,
            )
            replace_tensor("qweight", marlin_qweight)

            # Permute scales
            scales_size_k = part_size_k
            scales_size_n = part_size_n
            if self.quant_config.desc_act:
                scales_size_k = full_size_k

            marlin_scales = marlin_permute_scales(
                layer.scales,
                scales_size_k,
                scales_size_n,
                self.quant_config.group_size,
                self.quant_config.weight_bits,
            )
            replace_tensor("scales", marlin_scales)

        output = ops.gptq_marlin_gemm(
            reshaped_x,
            layer.qweight,
            layer.scales,
            layer.g_idx,
            layer.g_idx_sort_indices,
            layer.workspace,
            self.quant_config.weight_bits,
            size_m,
            part_size_n,
            part_size_k,
            layer.is_k_full,
        )

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
