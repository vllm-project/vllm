# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum
from typing import Callable, Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  FusedMoEMethodBase)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig, mxfp4_w4a4_moe_quant_config)
from vllm.model_executor.layers.fused_moe.trtllm_moe import TrtLlmGenExperts
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    _can_support_mxfp4, _swizzle_mxfp4)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils import (has_triton_kernels, is_torch_equal_or_newer,
                        next_power_of_2, round_up)
from vllm.utils.flashinfer import has_flashinfer

logger = init_logger(__name__)


# enum for mxfp4 backend
class Mxfp4Backend(Enum):
    NONE = 0

    # FlashInfer Backend
    SM100_FI_MXFP4_MXFP8_TRTLLM = 1
    SM100_FI_MXFP4_MXFP8_CUTLASS = 2
    SM100_FI_MXFP4_BF16 = 3
    SM90_FI_MXFP4_BF16 = 4

    # Marlin Backend
    MARLIN = 5

    # Triton Backend
    TRITON = 6


def get_mxfp4_backend():
    # Backend Selection
    if current_platform.is_cuda():
        if (current_platform.is_device_capability(90) and has_flashinfer()
                and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16):
            logger.info_once("Using FlashInfer MXFP4 BF16 backend for SM90")
            return Mxfp4Backend.SM90_FI_MXFP4_BF16
        elif (current_platform.is_device_capability(100) and has_flashinfer()
              and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS):
            logger.info_once(
                "Using FlashInfer MXFP4 MXFP8 CUTLASS backend for SM100")
            return Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
        elif (current_platform.is_device_capability(100) and has_flashinfer()
              and envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8):
            logger.info_once(
                "Using FlashInfer MXFP4 MXFP8 TRTLLM backend for SM100, "
                "for high concurrency throughput workloads consider setting "
                "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 for better "
                "performance")
            return Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
        elif current_platform.is_device_capability(100) and has_flashinfer():
            logger.info_once(
                "Using FlashInfer MXFP4 BF16 backend for SM100, "
                "For faster performance on SM100, consider setting "
                "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1, though this may impact "
                "accuracy.")
            return Mxfp4Backend.SM100_FI_MXFP4_BF16
        elif ((current_platform.is_device_capability(100)
               or current_platform.is_device_capability(90))
              and not has_flashinfer()):
            logger.warning_once(
                "MXFP4 MoE is enabled on Hopper/Blackwell but FlashInfer "
                "is not available. This may result in degraded performance. "
                "Please `pip install vllm[flashinfer]` for best results.")

        # If FlashInfer is not available, try either Marlin or Triton
        if current_platform.get_device_capability(
        )[0] < 9 or not has_triton_kernels() or not is_torch_equal_or_newer(
                "2.8.0"):
            logger.info_once("Using Marlin backend")
            return Mxfp4Backend.MARLIN
        else:
            logger.info_once("Using Triton backend")
            return Mxfp4Backend.TRITON
    elif current_platform.is_rocm() and has_triton_kernels():
        logger.info_once("Using Triton backend")
        return Mxfp4Backend.TRITON

    return Mxfp4Backend.NONE


class Mxfp4Config(QuantizationConfig):

    def __init__(self, ignored_layers: Optional[list[str]] = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            raise NotImplementedError("Mxfp4 linear layer is not implemented")
        elif isinstance(layer, FusedMoE):
            return Mxfp4MoEMethod(layer.moe_config)
        elif isinstance(layer, Attention):
            raise NotImplementedError(
                "Mxfp4 attention layer is not implemented")
        return None


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.topk_indices_dtype = None
        self.moe = moe
        self.mxfp4_backend = get_mxfp4_backend()
        self.max_capture_size = get_current_vllm_config(
        ).compilation_config.max_capture_size

        assert self.mxfp4_backend != Mxfp4Backend.NONE, (
            "No MXFP4 MoE backend (FlashInfer/Marlin/Triton) available."
            "Please check your environment and try again.")
        self._cache_permute_indices: dict[torch.Size, torch.Tensor] = {}

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        # FIXME (zyongye): ship after torch and safetensors support mxfp4
        # is_torch_mxfp4_available = (
        #     hasattr(torch, "float4_e2m1fn_x2") and
        #     hasattr(torch, "float8_e8m0fnu"))
        # if is_torch_mxfp4_available:
        #     weight_dtype = torch.float4_e2m1fn_x2
        #     scale_dtype = torch.float8_e8m0fnu

        mxfp4_block = 32

        intermediate_size_per_partition_after_pad = \
            intermediate_size_per_partition
        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            # The moe marlin kernel requires that for each linear
            # n % 256 == 0 and k % 128 == 0.
            # In gate_up_proj:
            #    n = 2 * intermediate_size_per_partition_after_pad
            #    k = hidden_size
            # In down_proj
            #    n = hidden_size
            #    k = intermediate_size_per_partition_after_pad
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 128)
            hidden_size = round_up(hidden_size, 256)

            layer.params_dtype = params_dtype
            layer.num_experts = num_experts
            layer.hidden_size = hidden_size
            layer.intermediate_size_per_partition = \
                intermediate_size_per_partition_after_pad
        elif (self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
              or self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16):
            # pad the intermediate size to be a multiple of 2 * mxfp4_block
            # for to hold non-uniform sharded tensor as well as swizzling
            # other padding to increase performance
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 256)
            hidden_size = round_up(hidden_size, 256)
        elif current_platform.is_rocm() or (
                self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
                or self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16):
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 128)
            hidden_size = round_up(hidden_size, 128)
        else:
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size_per_partition, 64)

        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            prepare_moe_fp4_layer_for_marlin(layer)
        elif (self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
              or self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16):
            from flashinfer.fp4_quantization import (
                nvfp4_block_scale_interleave)
            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w2_permute_indices)
            layer.gemm1_alpha = Parameter(torch.tensor(
                [1.702] * self.num_experts, dtype=torch.float32).cuda(),
                                          requires_grad=False)
            layer.gemm1_beta = Parameter(torch.tensor(
                [1.0] * self.num_experts, dtype=torch.float32).cuda(),
                                         requires_grad=False)
            layer.gemm1_clamp_limit = Parameter(torch.tensor(
                [7.0] * self.num_experts, dtype=torch.float32).cuda(),
                                                requires_grad=False)
            sf_block_size = 32  # mxfp4 block size

            assert (layer.w13_weight.dim() == 3
                    and layer.w13_weight.shape[0] == self.num_experts
                    and layer.w13_weight.shape[1] == self.intermediate_size * 2
                    and layer.w13_weight.shape[2] == self.hidden_size // 2)
            assert (layer.w13_weight_scale.dim() == 3
                    and layer.w13_weight_scale.shape[0] == self.num_experts
                    and layer.w13_weight_scale.shape[1]
                    == self.intermediate_size * 2
                    and layer.w13_weight_scale.shape[2]
                    == self.hidden_size // sf_block_size)
            assert (layer.w2_weight.dim() == 3
                    and layer.w2_weight.shape[0] == self.num_experts
                    and layer.w2_weight.shape[1] == self.hidden_size and
                    layer.w2_weight.shape[2] == self.intermediate_size // 2)
            assert (layer.w2_weight_scale.dim() == 3
                    and layer.w2_weight_scale.shape[1] == self.hidden_size
                    and layer.w2_weight_scale.shape[2]
                    == self.intermediate_size // sf_block_size)
            assert (layer.w13_bias.dim() == 2
                    and layer.w13_bias.shape[0] == self.num_experts
                    and layer.w13_bias.shape[1] == self.intermediate_size * 2)
            assert (layer.w2_bias.dim() == 2
                    and layer.w2_bias.shape[0] == self.num_experts
                    and layer.w2_bias.shape[1] == self.hidden_size)

            w13_weight_scale = layer.w13_weight_scale.data
            w2_weight_scale = layer.w2_weight_scale.data
            w13_weight = layer.w13_weight.data
            w2_weight = layer.w2_weight.data
            w13_bias = layer.w13_bias.data.to(torch.float32)
            w2_bias = layer.w2_bias.data.to(torch.float32)

            # Swap w1 and w3 as the definition of
            # swiglu is different in the trtllm-gen
            def swap_every_two_rows(x, axis=-1):
                shape = x.shape
                if axis < 0:
                    axis = len(shape) + axis

                # Create a new shape with pairs swapped along specified axis
                new_shape = list(shape)
                new_shape[axis] = shape[axis] // 2
                new_shape.insert(axis + 1, 2)

                # Reshape to expose pairs, swap them, and reshape back
                x = x.reshape(*new_shape)
                x = x.flip(axis + 1)
                new_shape = list(shape)
                return x.reshape(*new_shape)

            w13_weight_scale = swap_every_two_rows(w13_weight_scale, -2)
            w13_weight = swap_every_two_rows(w13_weight, -2)
            w13_bias = swap_every_two_rows(w13_bias, -1)

            # Do not interleave as the checkpoint is already interleaved

            # Shuffle weights and scaling factors for transposed mma output
            gemm1_weights_mxfp4_shuffled = []
            gemm1_scales_mxfp4_shuffled = []
            gemm2_weights_mxfp4_shuffled = []
            gemm2_scales_mxfp4_shuffled = []
            gemm1_bias_shuffled = []
            gemm2_bias_shuffled = []
            epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
            for i in range(self.num_experts):
                # w13 weight shuffling
                permute_indices = _maybe_get_cached_w2_permute_indices(
                    self._cache_permute_indices,
                    w13_weight[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                gemm1_weights_mxfp4_shuffled.append(w13_weight[i].view(
                    torch.uint8)[permute_indices.to(
                        w13_weight.device)].contiguous())
                # w13 scale shuffling
                permute_sf_indices = _maybe_get_cached_w2_permute_indices(
                    self._cache_permute_indices,
                    w13_weight_scale[i].view(torch.uint8),
                    epilogue_tile_m,
                    num_elts_per_sf=16,
                )
                gemm1_scales_mxfp4_shuffled.append(
                    nvfp4_block_scale_interleave(w13_weight_scale[i].view(
                        torch.uint8)[permute_sf_indices.to(
                            w13_weight_scale.device)].contiguous()))
                # w13 bias shuffling
                permute_bias_indices = _maybe_get_cached_w2_permute_indices(
                    self._cache_permute_indices,
                    w13_bias[i].clone().reshape(-1, 1),
                    epilogue_tile_m,
                )
                gemm1_bias_shuffled.append(w13_bias[i].clone().reshape(
                    -1,
                    1)[permute_bias_indices.to(w13_bias.device)].contiguous())
                # w2 weight shuffling
                permute_indices = _maybe_get_cached_w2_permute_indices(
                    self._cache_permute_indices,
                    w2_weight[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                gemm2_weights_mxfp4_shuffled.append(w2_weight[i].view(
                    torch.uint8)[permute_indices.to(
                        w2_weight.device)].contiguous())
                # w2 scale shuffling
                permute_sf_indices = _maybe_get_cached_w2_permute_indices(
                    self._cache_permute_indices,
                    w2_weight_scale[i].view(torch.uint8),
                    epilogue_tile_m,
                    num_elts_per_sf=16,
                )
                gemm2_scales_mxfp4_shuffled.append(
                    nvfp4_block_scale_interleave(w2_weight_scale[i].view(
                        torch.uint8)[permute_sf_indices.to(
                            w2_weight_scale.device)].contiguous()))
                # w2 bias shuffling
                permute_indices = _maybe_get_cached_w2_permute_indices(
                    self._cache_permute_indices,
                    w2_bias[i].clone().reshape(-1, 1),
                    epilogue_tile_m,
                )
                gemm2_bias_shuffled.append(w2_bias[i].clone().reshape(
                    -1, 1)[permute_indices.to(w2_bias.device)].contiguous())

            w13_weight = torch.stack(gemm1_weights_mxfp4_shuffled)
            w13_weight_scale = torch.stack(
                gemm1_scales_mxfp4_shuffled).reshape(
                    self.num_experts, 2 * self.intermediate_size,
                    self.hidden_size // sf_block_size).view(
                        torch.float8_e4m3fn)

            w2_weight = torch.stack(gemm2_weights_mxfp4_shuffled)
            w2_weight_scale = torch.stack(gemm2_scales_mxfp4_shuffled).reshape(
                self.num_experts, self.hidden_size, self.intermediate_size //
                sf_block_size).view(torch.float8_e4m3fn)

            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = Parameter(w13_weight_scale,
                                               requires_grad=False)
            layer.w2_weight = Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = Parameter(w2_weight_scale,
                                              requires_grad=False)
            layer.w13_bias = Parameter(
                torch.stack(gemm1_bias_shuffled).reshape(self.num_experts, -1),
                requires_grad=False)
            layer.w2_bias = Parameter(torch.stack(gemm2_bias_shuffled).reshape(
                self.num_experts, -1),
                                      requires_grad=False)
        elif (self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
              or self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16):
            layer.gemm1_alpha = Parameter(torch.tensor(
                [1.702] * self.num_experts, dtype=torch.float32).cuda(),
                                          requires_grad=False)
            layer.gemm1_beta = Parameter(torch.tensor(
                [1.0] * self.num_experts, dtype=torch.float32).cuda(),
                                         requires_grad=False)
            layer.gemm1_clamp_limit = Parameter(torch.tensor(
                [7.0] * self.num_experts, dtype=torch.float32).cuda(),
                                                requires_grad=False)

            sf_block_size = 32  # mxfp4 block size

            # Common shape assertions
            assert (layer.w13_weight.dim() == 3
                    and layer.w13_weight.shape[0] == self.num_experts
                    and layer.w13_weight.shape[1] == self.intermediate_size * 2
                    and layer.w13_weight.shape[2] == self.hidden_size // 2)
            assert (layer.w13_weight_scale.dim() == 3
                    and layer.w13_weight_scale.shape[0] == self.num_experts
                    and layer.w13_weight_scale.shape[1]
                    == self.intermediate_size * 2
                    and layer.w13_weight_scale.shape[2]
                    == self.hidden_size // sf_block_size)
            assert (layer.w2_weight.dim() == 3
                    and layer.w2_weight.shape[0] == self.num_experts
                    and layer.w2_weight.shape[1] == self.hidden_size and
                    layer.w2_weight.shape[2] == self.intermediate_size // 2)
            assert (layer.w2_weight_scale.dim() == 3
                    and layer.w2_weight_scale.shape[1] == self.hidden_size
                    and layer.w2_weight_scale.shape[2]
                    == self.intermediate_size // sf_block_size)
            assert (layer.w13_bias.dim() == 2
                    and layer.w13_bias.shape[0] == self.num_experts
                    and layer.w13_bias.shape[1] == self.intermediate_size * 2)
            assert (layer.w2_bias.dim() == 2
                    and layer.w2_bias.shape[0] == self.num_experts
                    and layer.w2_bias.shape[1] == self.hidden_size)

            # De-interleave and swap for w13 weight, bias, and scales
            w13_w = layer.w13_weight.data
            gate_w, up_w = w13_w[:, ::2, :], w13_w[:, 1::2, :]
            deinterleaved_w13_w = torch.cat([gate_w, up_w], dim=1)
            w1_w, w3_w = torch.chunk(deinterleaved_w13_w, 2, dim=1)
            w13_weight_swapped = torch.cat([w3_w, w1_w], dim=1)

            w13_b = layer.w13_bias.data.to(torch.float32)
            gate_b, up_b = w13_b[:, ::2], w13_b[:, 1::2]
            deinterleaved_w13_b = torch.cat([gate_b, up_b], dim=1)
            b1, b3 = torch.chunk(deinterleaved_w13_b, 2, dim=-1)
            w13_bias_swapped = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)

            w13_s = layer.w13_weight_scale.data
            gate_s, up_s = w13_s[:, ::2, :], w13_s[:, 1::2, :]
            deinterleaved_w13_s = torch.cat([gate_s, up_s], dim=1)
            s1, s3 = torch.chunk(deinterleaved_w13_s, 2, dim=1)
            w13_scale_swapped = torch.cat([s3, s1], dim=1)

            if self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS:
                from flashinfer import block_scale_interleave

                orig_shape = w13_scale_swapped.shape
                w13_scale_interleaved = block_scale_interleave(
                    w13_scale_swapped.view(torch.uint8)).reshape(orig_shape)

                w2_s = layer.w2_weight_scale.data
                orig_shape = w2_s.shape
                w2_scale_interleaved = block_scale_interleave(
                    w2_s.view(torch.uint8)).reshape(orig_shape)

                layer.w13_weight = Parameter(w13_weight_swapped,
                                             requires_grad=False)
                layer.w13_weight_scale = Parameter(w13_scale_interleaved,
                                                   requires_grad=False)
                layer.w13_bias = Parameter(w13_bias_swapped,
                                           requires_grad=False)
                layer.w2_weight_scale = Parameter(w2_scale_interleaved,
                                                  requires_grad=False)
            elif self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16:

                def _interleave_mxfp4_cutlass_sm90(w):
                    w_shape = w.shape
                    w_interleaved = w.reshape(w_shape[0], w_shape[1],
                                              (w_shape[2] // 4), 4)
                    w_interleaved = w_interleaved.permute(0, 2, 1, 3)
                    w_interleaved = w_interleaved.reshape(
                        w_shape[0], w_shape[2] // 4, w_shape[1] * 4)
                    return w_interleaved

                w31_scales = w13_scale_swapped.to(torch.uint8).view(
                    torch.uint8)
                w31_scales_interleaved = _interleave_mxfp4_cutlass_sm90(
                    w31_scales)

                w2_weight_scale = layer.w2_weight_scale.data
                w2_scales = w2_weight_scale.to(torch.uint8).view(torch.uint8)
                w2_scales_interleaved = _interleave_mxfp4_cutlass_sm90(
                    w2_scales)

                layer.w13_weight = torch.nn.Parameter(torch.cat([w3_w, w1_w],
                                                                dim=1),
                                                      requires_grad=False)
                layer.w13_bias = torch.nn.Parameter(w13_bias_swapped,
                                                    requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w31_scales_interleaved, requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_scales_interleaved, requires_grad=False)
        elif self.mxfp4_backend == Mxfp4Backend.TRITON:
            from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

            w13_bias = layer.w13_bias.to(torch.float32)
            w2_bias = layer.w2_bias.to(torch.float32)

            layer.w13_bias = Parameter(w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(w2_bias, requires_grad=False)

            # FIXME warp need to be adjusted based on batch size
            # only apply to  batched mode
            if self.moe.use_ep:
                num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
            else:
                num_warps = 8

            w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
                layer.w13_weight, layer.w13_weight_scale, num_warps)
            w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
                layer.w2_weight, layer.w2_weight_scale, num_warps)

            self.w13_precision_config = PrecisionConfig(
                weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex))
            self.w2_precision_config = PrecisionConfig(
                weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex))

            self.w13_weight_triton_tensor = w13_weight
            self.w2_weight_triton_tensor = w2_weight

            # need to delete the original weights to save memory on single GPU
            del layer.w13_weight
            del layer.w2_weight
            layer.w13_weight = None
            layer.w2_weight = None
            torch.cuda.empty_cache()
        else:
            raise ValueError(f"Unsupported backend: {self.mxfp4_backend}")

    def _get_tile_tokens_dim(self, x: torch.Tensor, top_k: int):
        # Number of tokens in the input tensor.
        num_tokens = x.shape[0]
        # Factor to account for the imbalance of the experts.
        # factor equals to the
        # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
        # - 1.0 means perfect expert distribution.
        # - > 1.0 means some experts have more
        #     tokens than the perfect distribution.
        # - < 1.0 does not make sense.
        imbalance_factor = 1.3
        # Calculate the number of tokens per expert
        # assuming perfect distribution.
        num_tokens_per_expert = (num_tokens * top_k) // self.num_experts
        # Apply the imbalance factor.
        num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
        # And pad the number to the next power of 2.
        tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
        # Cap to 8-64 tokens per CTA tile
        # as it's the range supported by the kernel.
        tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

        return tile_tokens_dim

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:

        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            return None

        if self.mxfp4_backend == Mxfp4Backend.TRITON:
            w1_scale = self.w13_precision_config
            w2_scale = self.w2_precision_config
        else:
            w1_scale = layer.w13_weight_scale
            w2_scale = layer.w2_weight_scale

        return mxfp4_w4a4_moe_quant_config(
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        if (prepare_finalize.activation_format ==
                mk.FusedMoEActivationFormat.BatchedExperts):
            raise NotImplementedError(
                "Mxfp4 does not support batched experts format for EP")
        else:
            if (self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
                    or self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16):
                # B200 code-path
                kwargs = {
                    "gemm1_alpha": layer.gemm1_alpha,
                    "gemm1_beta": layer.gemm1_beta,
                    "gemm1_clamp_limit": layer.gemm1_clamp_limit,
                    # TODO(bnell): part of quant_config
                    "max_capture_size": self.max_capture_size,
                }
                assert self.moe_quant_config is not None
                return TrtLlmGenExperts(self.moe, self.moe_quant_config,
                                        **kwargs)
            else:
                # Use matmul_ogs from triton_kernels here!
                raise NotImplementedError(
                    "Mxfp4 does not support non-batched experts format for EP")

    def _route_and_experts(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            renormalize: bool,
            use_grouped_topk: bool = False,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            global_num_experts: int = -1,
            expert_map: Optional[torch.Tensor] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            activation: str = "silu",
            enable_eplb: bool = False,
            expert_load_view: Optional[torch.Tensor] = None,
            logical_to_physical_map: Optional[torch.Tensor] = None,
            logical_replica_count: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        assert isinstance(self.fused_experts, mk.FusedMoEModularKernel)

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count)

        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        if enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        if self.mxfp4_backend == Mxfp4Backend.MARLIN:
            topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias)

            return torch.ops.vllm.fused_marlin_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_bias,
                layer.w2_bias,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                router_logits,
                topk_weights,
                topk_ids,
                global_scale1=None,
                global_scale2=None,
                quant_type_id=scalar_types.float4_e2m1f.id,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                activation=activation,
                expert_map=expert_map)

        if self.fused_experts is not None:
            return self._route_and_experts(
                layer,
                x,
                router_logits,
                top_k,
                renormalize,
                use_grouped_topk,
                topk_group,
                num_expert_group,
                global_num_experts,
                expert_map,
                custom_routing_function,
                scoring_func,
                e_score_correction_bias,
                apply_router_weight_on_input,
                activation,
                enable_eplb,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )

        assert _can_support_mxfp4(
            use_grouped_topk, topk_group, num_expert_group, expert_map,
            custom_routing_function, e_score_correction_bias,
            apply_router_weight_on_input, scoring_func, activation,
            expert_load_view, logical_to_physical_map,
            logical_replica_count), (
                "MXFP4 are not supported with this configuration.")

        if (self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
                or self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16):
            from flashinfer import trtllm_fp4_block_scale_moe
            if self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16:
                assert x.dtype == torch.bfloat16
                x_quant = x
                x_scale = None
            elif self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM:
                from flashinfer import mxfp8_quantize
                x_quant, x_scale = mxfp8_quantize(x, False)  # to mxfp8
                x_scale = x_scale.view(torch.float8_e4m3fn).reshape(
                    *x.shape[:-1], -1)

            trtllm_gen_output = trtllm_fp4_block_scale_moe(
                router_logits.to(torch.bfloat16),
                None,  # routing_bias
                x_quant,
                x_scale,
                layer.w13_weight,  # uint8 (e2m1 x 2)
                layer.w13_weight_scale,  # uint8 (e4m3 x 2)
                layer.w13_bias,  # fp32 per expert per channel
                layer.gemm1_alpha,  # fp32 per expert
                layer.gemm1_beta,  # fp32 per expert
                layer.gemm1_clamp_limit,  # fp32 per expert
                layer.w2_weight,  # uint8 (e2m1 x 2)
                layer.w2_weight_scale,  # ue8m0
                layer.w2_bias,  # fp32 per expert per channel
                None,  # output1_scale_scalar
                None,  # output1_scale_gate_scalar
                None,  # output2_scale_scalar
                global_num_experts,
                top_k,
                None,  # n_group
                None,  # topk_group
                self.intermediate_size,  # padded to multiple of 256
                layer.ep_rank * layer.local_num_experts,  # local_expert_offset
                self.num_experts,  # local num experts
                None,
                self._get_tile_tokens_dim(x, top_k),
                1 if renormalize else 0,  # routing_method_type, renormalize
                True,  # do finalize
                tune_max_num_tokens=self.max_capture_size,
            )[0]
            return trtllm_gen_output
        elif (self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
              or self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16):
            from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

            topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
            )

            # Backend-specific preparation
            if self.mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS:

                from flashinfer import mxfp8_quantize

                x_quant, x_scale = mxfp8_quantize(x, True, 32)

                fake_input_scale = torch.ones(self.num_experts,
                                              device=x.device)
                quant_scales = [
                    layer.w13_weight_scale.contiguous().view(torch.int32),
                    fake_input_scale,
                    layer.w2_weight_scale.contiguous().view(torch.int32),
                    fake_input_scale,
                ]

                fi_input = x_quant
                extra_kwargs = dict(
                    use_mxfp8_act_scaling=True,
                    input_sf=x_scale,
                    fc1_expert_weights=layer.w13_weight.contiguous().view(
                        torch.long),
                    fc2_expert_weights=layer.w2_weight.contiguous().view(
                        torch.long),
                )
            elif self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16:
                assert x.dtype == torch.bfloat16

                quant_scales = [
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                ]

                fi_input = x
                extra_kwargs = dict(
                    use_w4_group_scaling=True,
                    fc1_expert_weights=layer.w13_weight,
                    fc2_expert_weights=layer.w2_weight,
                )

            output = torch.empty_like(x, dtype=torch.bfloat16)
            _ = flashinfer_cutlass_fused_moe(
                input=fi_input,
                token_selected_experts=topk_ids.to(torch.int).contiguous(),
                token_final_scales=topk_weights,
                output_dtype=torch.bfloat16,
                output=output,
                quant_scales=quant_scales,
                fc1_expert_biases=layer.w13_bias,
                fc2_expert_biases=layer.w2_bias,
                swiglu_alpha=layer.gemm1_alpha,
                swiglu_beta=layer.gemm1_beta,
                swiglu_limit=layer.gemm1_clamp_limit,
                tp_size=self.moe.tp_size,
                tp_rank=self.moe.tp_rank,
                ep_size=self.moe.ep_size,
                ep_rank=self.moe.ep_rank,
                tune_max_num_tokens=self.max_capture_size,
                **extra_kwargs,
            )

            return output
        elif self.mxfp4_backend == Mxfp4Backend.TRITON:
            from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (  # noqa: E501
                triton_kernel_moe_forward)
            return triton_kernel_moe_forward(
                hidden_states=x,
                w1=self.w13_weight_triton_tensor,
                w2=self.w2_weight_triton_tensor,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.mxfp4_backend}")
