import gc
from typing import Any, Dict, List, Optional

import enum
from enum import Enum

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from magic_wand import (MARLIN_SUPPORTED_NUM_BITS,
                        MARLIN_SUPPORTED_GROUP_SIZES, MARLIN_TILE,
                        MARLIN_MIN_THREAD_N, MARLIN_MIN_THREAD_K,
                        MARLIN_MAX_PARALLEL, get_pack_factor,
                        marlin_permute_scales, marlin_repack_from_gptq,
                        marlin_gemm)


class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    def __init__(self, weight_bits: int, group_size: int, desc_act: bool,
                 is_sym: bool) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.is_sym = is_sym

        # Verify
        if self.weight_bits not in MARLIN_SUPPORTED_NUM_BITS:
            raise ValueError(
                f"Got weight_bits = {weight_bits}, but Marlin only supports {MARLIN_SUPPORTED_NUM_BITS}"
            )

        if self.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Got group_size = {group_size}, but Marlin only supports {MARLIN_SUPPORTED_GROUP_SIZES}"
            )

        if not self.is_sym:
            raise ValueError(
                "Marlin only supports symmetric quantized weights")

        # Init
        self.pack_factor = get_pack_factor(weight_bits)
        self.tile_size = MARLIN_TILE
        self.min_thread_n = MARLIN_MIN_THREAD_N
        self.min_thread_k = MARLIN_MIN_THREAD_K
        self.max_parallel = MARLIN_MAX_PARALLEL

        # Permutation length used by the marlin kernels.
        self.perm_len = 1024

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
    # Need to figure it out
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

    def get_linear_method(self) -> "GPTQMarlinLinearMethod":
        return GPTQMarlinLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class GPTQMarlinLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Marlin.

    Args:
        quant_config: The GPTQ Marlin quantization config.
    """

    def __init__(self, quant_config: GPTQMarlinConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
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

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                f"Weight input_size_per_partition = {input_size_per_partition}"
                f" is not divisible by group_size = {group_size}.")

        # Detect sharding of scales/zp

        # By default, no sharding over "input dim"
        is_k_full = True
        scales_and_zp_size = input_size // group_size
        scales_and_zp_input_dim = None

        if self.quant_config.desc_act:
            assert self.quant_config.group_size != -1

            is_k_full = input_size_per_partition == input_size

        else:
            is_k_full = True  # Because of full alignment with group-size and shard of scales/zp

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
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            })

        # Activation order
        g_idx = Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(g_idx, {"input_dim": 0, "ignore_warning": True})

        g_idx_sort_indices = Parameter(
            torch.empty(
                g_idx.shape,
                dtype=torch.int32,
            ),
            requires_grad=False,
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
        set_weight_attrs(scales, {
            "input_dim": scales_and_zp_input_dim,
            "output_dim": 1,
        })

        # Quantized zero-points
        qzeros = Parameter(
            torch.empty(scales_and_zp_size,
                        output_size_per_partition //
                        self.quant_config.pack_factor,
                        dtype=torch.int32,
                        device="meta"),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })

        # Allocate marlin workspace
        max_workspace_size = (
            output_size_per_partition //
            self.quant_config.min_thread_n) * self.quant_config.max_parallel
        workspace = Parameter(
            torch.zeros(max_workspace_size, device="cuda", dtype=torch.int),
            requires_grad=False,
        )

        return {
            "qweight": qweight,
            "g_idx": g_idx,
            "g_idx_sort_indices": g_idx_sort_indices,
            "scales": scales,
            "qzeros": qzeros,
            "workspace": workspace,
            "input_size_per_partition": input_size_per_partition,
            "output_size_per_partition": output_size_per_partition,
            "input_size": input_size,
            "output_size": output_size,
            "is_k_full": is_k_full,
            "marlin_state": GPTQMarlinState.REPACK,
        }

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])

        size_m = reshaped_x.shape[0]
        part_size_n = weights["output_size_per_partition"]
        part_size_k = weights["input_size_per_partition"]
        full_size_n = weights["output_size"]
        full_size_k = weights["input_size"]

        out_shape = x.shape[:-1] + (part_size_n, )

        if weights["marlin_state"] == GPTQMarlinState.REPACK:
            weights["marlin_state"] = GPTQMarlinState.READY

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def replace_tensor(name, new_t):
                weights[name] = weights[name].reshape(new_t.shape)
                weights[name].copy_(new_t)
                del new_t

            cur_device = weights["qweight"].device

            # Process act_order
            if self.quant_config.desc_act:
                # Get sorting based on g_idx
                g_idx_sort_indices = torch.argsort(weights["g_idx"]).to(
                    torch.int)

                sorted_g_idx = weights["g_idx"][g_idx_sort_indices]

                replace_tensor("g_idx", sorted_g_idx)
                replace_tensor("g_idx_sort_indices", g_idx_sort_indices)

            else:
                # Reset g_idx related tensors
                empty_g_idx = torch.empty(0,
                                          dtype=torch.int,
                                          device=cur_device)
                empty_g_idx_sort_indices = torch.empty(0,
                                                       dtype=torch.int,
                                                       device=cur_device)
                weights["g_idx"] = empty_g_idx
                weights["g_idx_sort_indices"] = empty_g_idx_sort_indices

            # Repack weights
            marlin_qweight = marlin_repack_from_gptq(
                weights["qweight"],
                weights["g_idx_sort_indices"],
                part_size_k,
                part_size_n,
            )
            replace_tensor("qweight", marlin_qweight)

            # Permute scales
            scales_size_k = part_size_k
            scales_size_n = part_size_n
            if self.quant_config.desc_act:
                scales_size_k = full_size_k

            marlin_scales = marlin_permute_scales(weights["scales"],
                                                  scales_size_k, scales_size_n,
                                                  self.quant_config.group_size)
            replace_tensor("scales", marlin_scales)

        output = marlin_gemm(reshaped_x, weights["qweight"], weights["scales"],
                             weights["g_idx"], weights["g_idx_sort_indices"],
                             weights["workspace"], size_m, part_size_n,
                             part_size_k, weights["is_k_full"])

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
