# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 GEMM for ROCm gfx942 (MI300) with AITER's Triton ``gemm_a16w4``.

AITER's ``gemm_a16w4`` (``aiter.ops.triton.gemm.basic.gemm_a16w4``) reads
int4 weights as plain row-major nibbles along K:

  w:        [N, K // 2]  uint8   byte j of row n = q[n, 2j] | q[n, 2j + 1] << 4
  w_scales: [N, K // G]  fp16 / bf16
  w_zeros:  [N, K // G]  uint8   (None: zero point 8, i.e. uint4b8)

The GPTQ/AWQ/compressed-tensors loaders hand the kernel framework int32 words
holding 8 sequential K values (bit 4*i = value i), so ``w`` is that word
tensor in [N, K // 8] layout viewed as uint8; no nibble shuffle is needed.
"""

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class AiterW4A16LinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,  # symmetric GPTQ (zero point 8)
        scalar_types.uint4,  # asymmetric, explicit zero points (AWQ, CT)
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        # Arch gating is done by rocm_aiter_ops.is_linear_w4a16_enabled().
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "AiterW4A16LinearKernel is ROCm-only"

        if not rocm_aiter_ops.is_linear_w4a16_enabled():
            return (
                False,
                "AITER W4A16 GEMM needs VLLM_ROCM_USE_AITER=1, "
                "VLLM_ROCM_USE_AITER_LINEAR=1, gfx942 and an AITER build with "
                "gemm_a16w4",
            )

        if envs.VLLM_BATCH_INVARIANT:
            # The tuned BLOCK_SIZE_K (and split-K) changes with the M bucket.
            return False, "AiterW4A16LinearKernel is not batch invariant"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                f"AiterW4A16LinearKernel; supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "AiterW4A16LinearKernel supports fp16 and bf16 only"

        K = c.partition_weight_shape[0]
        if K % 8 != 0:
            return False, f"Input features ({K}) must be a multiple of 8"

        gs = c.group_size
        if gs != -1 and gs != K:
            if gs < 16 or gs & (gs - 1) != 0:
                return (
                    False,
                    f"Group size ({gs}) must be -1 or a power of two >= 16",
                )
            if K % gs != 0:
                return (
                    False,
                    f"Group size ({gs}) does not divide input features ({K})",
                )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        def transform_w_q(x: BasevLLMParameter) -> torch.Tensor:
            # [N, K // 8] int32, 8 sequential K values per word -> [N, K // 2]
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            return x.data.contiguous().view(torch.uint8)

        def transform_w_s(x: BasevLLMParameter) -> torch.Tensor:
            permute_param_layout_(x, input_dim=1, output_dim=0)
            return x.data.contiguous().to(c.act_type)

        def transform_w_zp(x: BasevLLMParameter) -> torch.Tensor:
            # [N // 8, K // G] int32 packed along N -> [N, K // G] uint8
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=0)
            zp = unpack_quantized_values_into_int32(x.data, c.weight_type, packed_dim=0)
            return zp.to(torch.uint8).contiguous()

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        if c.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp = self._get_weight_params(layer)
        if not c.zero_points:
            # uint4b8: the kernel's default zero point is 8. Some loaders
            # still register (unused) qzeros; ignore them.
            w_zp = None

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)
        output = rocm_aiter_ops.gemm_a16w4(x_2d, w_q, w_s, w_zp, bias)
        return output.reshape(out_shape)
