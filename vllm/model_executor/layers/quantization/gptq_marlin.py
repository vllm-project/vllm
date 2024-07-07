from typing import Any, Dict, List, Optional

import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
<<<<<<< HEAD
from vllm.model_executor.layers.quantization.utils.marlin import (
    check_marlin_supported, get_max_workspace_size, marlin_permute_scales,
    replace_tensor, sort_g_idx, verify_marlin_supported,
    verify_marlin_supports_shape)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)

=======
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_K,
    GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_SUPPORTED_GROUP_SIZES,
    GPTQ_MARLIN_SUPPORTED_NUM_BITS, GPTQ_MARLIN_SUPPORTED_SYM,
    GPTQ_MARLIN_TILE)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

logger = init_logger(__name__)


# Permutations for Marlin scale shuffling
def get_scale_perms(num_bits: int):
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def get_pack_factor(num_bits: int):
    assert (num_bits in GPTQ_MARLIN_SUPPORTED_NUM_BITS
            ), f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int, num_bits: int):
    scale_perm, scale_perm_single = get_scale_perms(num_bits)
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s

>>>>>>> compressed-tensors-fp8

class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    def __init__(self, weight_bits: int, group_size: int, desc_act: bool,
                 is_sym: bool, lm_head_quantized: bool) -> None:
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        self.weight_bits = weight_bits
        self.pack_factor = 32 // self.weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.is_sym = is_sym
        self.lm_head_quantized = lm_head_quantized

        verify_marlin_supported(num_bits=self.weight_bits,
                                group_size=self.group_size,
                                is_sym=self.is_sym)

    def __repr__(self) -> str:
        return (f"GPTQMarlinConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

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
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, is_sym,
                   lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_marlin_compatible(hf_quant_cfg)

        is_valid_user_quant = (user_quant is None or user_quant == "marlin")

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "gptq":
            logger.info("Detected that the model can run with gptq_marlin"
                        ", however you specified quantization=gptq explicitly,"
                        " so forcing gptq. Use quantization=gptq_marlin for"
                        " faster inference")
        return None

    def get_quant_method(
            self,
            layer: torch.nn.Module) -> Optional["GPTQMarlinLinearMethod"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
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

<<<<<<< HEAD
        return check_marlin_supported(num_bits=num_bits,
                                      group_size=group_size,
                                      is_sym=sym,
                                      min_capability=cls.get_min_capability())
=======
        # If the capability of the device is too low, cannot convert.
        major, minor = current_platform.get_device_capability()
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
>>>>>>> compressed-tensors-fp8


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

        output_size_per_partition = sum(output_partition_sizes)

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        verify_marlin_supports_shape(
            output_size_per_partition=sum(output_partition_sizes),
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size)

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

        g_idx_sort_indices = torch.empty(
            g_idx.shape,
            dtype=torch.int32,
        )

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
        workspace = torch.zeros(
            get_max_workspace_size(output_size_per_partition),
            dtype=torch.int,
            requires_grad=False)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
        layer.g_idx_sort_indices = g_idx_sort_indices
        layer.workspace = workspace
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.is_k_full = is_k_full

    def process_weights_after_loading(self, layer: Module) -> None:

        # To be used as part of repacking
        part_size_n = layer.output_size_per_partition
        part_size_k = layer.input_size_per_partition
        full_size_k = layer.input_size

        # Sort for act_order
        if self.quant_config.desc_act:
            # Get sorting based on g_idx
            sort_g_idx(layer, "g_idx", "g_idx_sort_indices")
        else:
            # Reset g_idx to empty
            device = layer.qweight.device
            layer.g_idx = Parameter(torch.empty(0, dtype=torch.int,
                                                device=device),
                                    requires_grad=False)
            layer.g_idx_sort_indices = Parameter(torch.empty(0,
                                                             dtype=torch.int,
                                                             device=device),
                                                 requires_grad=False)

        # Repack weights into marlin format
        marlin_qweight = ops.gptq_marlin_repack(
            layer.qweight,
            layer.g_idx_sort_indices,
            part_size_k,
            part_size_n,
            self.quant_config.weight_bits,
        )
        replace_tensor(layer, "qweight", marlin_qweight)

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
        replace_tensor(layer, "scales", marlin_scales)

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

        out_shape = x.shape[:-1] + (part_size_n, )

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
