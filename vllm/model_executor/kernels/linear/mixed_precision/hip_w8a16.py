# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements


def _w8a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_dequant: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM kernel and dequant fallback.

    Registered as a custom op so torch.compile treats it as opaque,
    avoiding issues with the data-dependent branch.

    group_size: -1 = per-channel (w_s is 1-D [M]); >0 = per-group
    (w_s is 2-D [M, K/group_size]).
    """
    import vllm._custom_ops as ops

    N = x_2d.shape[0]
    K = x_2d.shape[1]

    if K * N <= LDS_CAPACITY_ELEMENTS:
        return ops.wvSplitK_int8(w_q, x_2d, w_s, cu_count, bias, group_size)

    return torch.nn.functional.linear(x_2d, w_dequant, bias)


def _w8a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_dequant: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    N = x_2d.size(0)
    M = w_q.size(0)
    return torch.empty((N, M), dtype=x_2d.dtype, device=x_2d.device)


def _register_w8a16_op():
    lib = torch.library.Library("_rocm_skinny_w8", "DEF")
    lib.define(
        "w8a16_apply(Tensor x_2d, Tensor w_q, Tensor w_s, Tensor w_dequant,"
        " Tensor? bias, int cu_count, int group_size) -> Tensor"
    )
    lib.impl("w8a16_apply", _w8a16_apply_impl, "CUDA")
    lib.impl("w8a16_apply", _w8a16_apply_fake, "Meta")
    return lib


_W8A16_LIB = _register_w8a16_op()


class HipW8A16LinearKernel(MPLinearKernel):
    """W8A16 per-channel int8 skinny GEMM for ROCm (gfx11).

    Uses the wvSplitK_int8 kernel for small batch sizes where activations
    fit in LDS. Falls back to dequant + torch.linear for larger batches.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        try:
            if not hasattr(torch.ops, "_rocm_C") or not hasattr(
                torch.ops._rocm_C, "wvSplitK_int8"
            ):
                return False, "wvSplitK_int8 op not available in this build"
        except Exception:
            return False, "ROCm ops not available"

        if c.weight_type.is_floating_point() or c.weight_type.size_bits != 8:
            return False, "requires 8-bit integer weights"

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if c.group_size != -1 and c.group_size not in (32, 64, 128):
            return False, (
                f"unsupported group_size={c.group_size}; "
                "supported: -1 (per-channel), 32, 64, 128"
            )

        if c.zero_points:
            return False, "does not support zero points (asymmetric)"

        if c.has_g_idx:
            return False, "does not support g_idx reordering"

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"

        if c.group_size != -1 and K % c.group_size != 0:
            return False, (f"K={K} must be divisible by group_size={c.group_size}")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        def transform_w_q(x: BasevLLMParameter) -> torch.Tensor:
            unpacked = unpack_quantized_values_into_int32(
                x.data, c.weight_type, packed_dim=x.packed_dim
            )
            bias_val = c.weight_type.bias
            return (unpacked - bias_val).to(torch.int8).contiguous()

        def transform_w_s(x: BasevLLMParameter) -> torch.Tensor:
            # Per-channel: collapse trailing dim to 1-D [M].
            # Per-group: expect 2-D [M, K/group_size]; ensure contiguous.
            if c.group_size == -1:
                return x.data.squeeze(-1).contiguous()
            return x.data.contiguous()

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        w_q, w_s, _, _ = self._get_weight_params(layer)
        if c.group_size == -1:
            self._w_dequant = (w_q.to(c.act_type) * w_s.unsqueeze(1)).contiguous()
        else:
            # Per-group dequant: w_q[m, g*G+i] * w_s[m, g] -> bf16 fallback.
            M, K = w_q.shape
            G = c.group_size
            w_q_g = w_q.to(c.act_type).reshape(M, K // G, G)
            self._w_dequant = (w_q_g * w_s.unsqueeze(-1)).reshape(M, K).contiguous()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        w_q, w_s, _, _ = self._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        N = x_2d.shape[0]
        K = x_2d.shape[1]
        M = w_q.shape[0]
        out_shape = x.shape[:-1] + (M,)

        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"hip_w8a16 {N}x{M}x{K}")
        )
        with ctx:
            cu_count = num_compute_units()
            output = torch.ops._rocm_skinny_w8.w8a16_apply(
                x_2d,
                w_q,
                w_s,
                self._w_dequant,
                bias,
                cu_count,
                self.config.group_size,
            )
        return output.reshape(out_shape)
