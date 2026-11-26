# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 GEMM on hipBLASLt through AITER (`aiter.hipb_w4a16_mm`).

AITER runs the GEMM on hipBLASLt's W4A16 kernels and picks the kernel per
layer and batch. Those kernels read the int4 weights in plain K order, while
the in-tree RDNA kernels read the ExLlama shuffle, so the two cannot share a
weight buffer: a layer this kernel accepts runs both prefill and decode
through AITER.

On gfx115x the weight rows are padded to an odd multiple of 128 B: these
kernels have no per-workgroup K offset, so rows on a 256 B (and worse, 1 KiB)
period send the workgroups reading one K window to the same DRAM channels.
Activations, outputs and scales stay dense. Zero-points are handed over as
vLLM packs them ([ceil(N/8), K/G] int32, row n in nibble n % 8), in one
allocation with the scales.
"""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)

# Batch bound for kernel selection outside an engine (no current VllmConfig).
_DEFAULT_MAX_BATCHED_TOKENS = 8192
# Probe at least this far: encoder towers can exceed max_num_batched_tokens.
_MIN_PROBE_M = 1 << 16


def _batch_buckets() -> tuple[int, ...]:
    """Powers of two up to the largest batch a call can bring: a layer is
    accepted only when AITER has a kernel for every bucket it can meet."""
    vllm_config = get_current_vllm_config_or_none()
    max_tokens = (
        vllm_config.scheduler_config.max_num_batched_tokens
        if vllm_config is not None
        else _DEFAULT_MAX_BATCHED_TOKENS
    )
    last = 1 << (max(max_tokens, _MIN_PROBE_M) - 1).bit_length()
    return tuple(1 << i for i in range(last.bit_length()))


def _on_gfx115x() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx115x

    return on_gfx115x()


def weight_row_bytes(k: int) -> int:
    """Row stride in bytes of the packed [N, K // 2] weight: on gfx115x the
    smallest odd multiple of 128 B that holds the row, dense elsewhere."""
    row_bytes = k // 2
    if not _on_gfx115x():
        return row_bytes
    lines = -(-row_bytes // 128)
    return (lines + (lines % 2 == 0)) * 128


def k_order_bytes(w_q: torch.Tensor, weight_type: ScalarType) -> torch.Tensor:
    """The [N, K] uint4 weight as [N, K // 2] uint8, element 2j in the low
    nibble of byte j (hipBLASLt's UNSIGNED_BIAS8 int4 encoding, no shuffle)."""
    if (
        w_q.dtype == torch.int32
        and getattr(w_q, "packed_factor", None) == 8
        and getattr(w_q, "packed_dim", None) == getattr(w_q, "input_dim", None)
    ):
        # Packed along K with element j at bits 4j of each word (GPTQ,
        # compressed-tensors): the little-endian bytes are already in K order.
        w = permute_param_layout_(w_q, input_dim=1, output_dim=0, packed_dim=1)
        return w.data.contiguous().view(torch.uint8)
    unpacked = unpack_quantized_values_into_int32(
        w_q.data, weight_type, packed_dim=w_q.packed_dim
    )
    # AWQ weights arrive as (K, N) with output_dim=1.
    if getattr(w_q, "output_dim", 0) != 0:
        unpacked = unpacked.t()
    return (unpacked[:, 0::2] | (unpacked[:, 1::2] << 4)).to(torch.uint8)


def pad_weight_rows(packed: torch.Tensor) -> torch.Tensor:
    """[N, K // 2] uint8 -> the same bytes as int8, rows `weight_row_bytes`
    apart."""
    n, row_bytes = packed.shape
    stride = weight_row_bytes(2 * row_bytes)
    buf = torch.empty((n, stride), dtype=torch.uint8, device=packed.device)
    buf[:, :row_bytes].copy_(packed)
    return buf[:, :row_bytes].view(torch.int8)


class AiterHipbW4A16LinearKernel(MPLinearKernel):
    """W4A16 on hipBLASLt through AITER, for every batch size of the layer.

    Selected only when AITER has kernels for the layer on this GPU, which
    `can_implement` asks AITER directly.
    """

    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
        scalar_types.uint4,  # asymmetric (zero_points)
    ]
    SUPPORTED_GROUP_SIZES = (32, 64, 128)

    @classmethod
    def get_min_capability(cls) -> int:
        # Arch support is AITER's answer, asked in can_implement.
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "AiterHipbW4A16LinearKernel only targets ROCm"
        from vllm.model_executor.kernels.linear import _get_linear_backend

        # `--linear-backend aiter` asks for AITER kernels whatever the flags.
        requested = _get_linear_backend(quantization="mixed_precision") == "aiter"
        if not (
            rocm_aiter_ops.is_linear_hipb_w4a16_enabled()
            or (requested and rocm_aiter_ops.has_hipb_w4a16())
        ):
            return (
                False,
                (
                    "requires an AITER with aiter.hipb_w4a16_mm and either "
                    "`--linear-backend aiter` or `VLLM_ROCM_USE_AITER=1`, "
                    "`VLLM_ROCM_USE_AITER_LINEAR=1` and "
                    "`VLLM_ROCM_USE_AITER_LINEAR_HIPB_W4A16=1`"
                ),
            )
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                (
                    f"Quant type {c.weight_type} not supported; "
                    f"supported: {cls.SUPPORTED_QUANT_TYPES}"
                ),
            )
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"
        if c.out_type is not None and c.out_type != c.act_type:
            return False, "requires the output dtype to match the activations"
        if c.group_size not in cls.SUPPORTED_GROUP_SIZES:
            return (
                False,
                (
                    f"Group size {c.group_size} not supported; "
                    f"supported: {cls.SUPPORTED_GROUP_SIZES}"
                ),
            )
        K, N = c.partition_weight_shape
        if K % c.group_size != 0:
            return False, f"K={K} must be divisible by group_size={c.group_size}"
        # A dense prefill copy beats this kernel for prefill (and
        # wvSplitK_int4_g for decode), so it keeps the layer on RDNAHybrid.
        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.kernel_config.rdna_hybrid_w4a16_prefill_dequant == "enable"
        ):
            return (
                False,
                (
                    "rdna_hybrid_w4a16_prefill_dequant=enable keeps the layer on "
                    "RDNAHybridW4A16LinearKernel"
                ),
            )
        try:
            supported = rocm_aiter_ops.hipb_w4a16_is_supported(
                c.act_type,
                N,
                K,
                c.group_size,
                c.zero_points,
                2 * weight_row_bytes(K),
                _batch_buckets(),
            )
        except Exception as e:
            # A broken AITER install (JIT build, kernel table) must not stop
            # the layer from loading on the in-tree kernels.
            logger.warning_once(
                "AITER's W4A16 support query failed, using the next W4A16 kernel: %s",
                e,
            )
            return False, f"AITER's W4A16 support query failed: {e}"
        if not supported:
            return (
                False,
                (
                    f"AITER has no W4A16 kernel on this GPU for N={N} K={K} "
                    f"group_size={c.group_size} zero_points={c.zero_points} "
                    f"{c.act_type}"
                ),
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        self._transform_param(
            layer,
            self.w_q_name,
            lambda x: pad_weight_rows(k_order_bytes(x, c.weight_type)),
        )
        self._transform_param(
            layer,
            self.w_s_name,
            lambda x: permute_param_layout_(
                x, input_dim=1, output_dim=0
            ).data.contiguous(),
        )
        if c.zero_points:
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: permute_param_layout_(
                    x, input_dim=1, output_dim=0, packed_dim=0
                ).data.contiguous(),
            )
            # The GEMM reads the scales and, 256 B after them, the zero-points
            # from one allocation: it becomes the scale parameter, so the
            # scales are stored once.
            _, w_s, w_zp = self._get_weight_params(layer)
            assert w_zp is not None
            scale = rocm_aiter_ops.hipb_w4a16_scale_buffer(w_s.data, w_zp.data)
            self._transform_param(layer, self.w_s_name, lambda x: scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q = getattr(layer, self.w_q_name)
        output = rocm_aiter_ops.hipb_w4a16_mm(
            x.reshape(-1, x.shape[-1]),
            w_q,
            getattr(layer, self.w_s_name),
            c.group_size,
            c.zero_points,
            bias,
        )
        return output.reshape(x.shape[:-1] + (w_q.shape[0],))
