# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W8A8 block-scaled FP8 GEMM on AMD RDNA4 (gfx1201).

Backed by the HIP kernels in ``csrc/rocm/w8a8_block_fp8_gemm_rdna4.cu`` and
exposed as ``torch.ops._rocm_C.w8a8_block_fp8_gemm_rdna4``. Opt in with
``VLLM_ROCM_USE_HIP_W8A8=1``; when unset ``can_implement`` reports False and
vLLM falls through to the next entry in
``_POSSIBLE_FP8_BLOCK_KERNELS[ROCM]`` (Triton).

Only the 128x128 quantisation block is supported.

Two things are fixed up once at load rather than per call:

* The weight scale is transposed to the ``[K/128, N/128]`` layout the kernels
  index. Inferring the orientation at runtime is ambiguous when the scale grid
  is square (``N/128 == K/128``, e.g. Qwen3-32B-FP8 ``qkv_proj`` at TP=2), so
  the op asserts the layout instead of guessing.
* ``VLLM_ROCM_FP8_PADDING`` leaves the weight as a non-contiguous ``[N, K]``
  view of an ``[N, K+256]`` buffer. The kernels take the row stride as a
  separate argument, so the view is widened back to its real extent rather
  than copied down to contiguous.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig

logger = init_logger(__name__)

# The quantisation block the kernels are compiled against.
BLOCK_N = 128
BLOCK_K = 128

SUPPORTED_OUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _widen_padded_weight(w: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` view of an ``[N, stride]`` buffer -> the ``[N, stride]`` tensor.

    The extra columns are zeros and contribute nothing, because the kernels
    take K from the activations, not from the weight. Falls back to a copy if
    the tensor is not the shape this expects, so an unrelated future layout
    change degrades to merely slower rather than silently wrong.
    """
    if w.is_contiguous():
        return w

    n, k = w.shape
    stride = w.stride(0)
    if w.stride(1) == 1 and stride > k:
        # as_strided reads up to offset + n*stride elements, so check the
        # storage really is that big rather than assuming F.pad's layout.
        need = w.storage_offset() + n * stride
        have = w.untyped_storage().size() // w.element_size()
        if have >= need:
            full = torch.as_strided(w, (n, stride), (stride, 1), w.storage_offset())
            # If the stride were wrong the rows would not line up. Compared as
            # raw bytes because comparison ops are not defined for every
            # float8 dtype.
            first_ok = torch.equal(
                full[0, :k].view(torch.uint8), w[0].view(torch.uint8)
            )
            last_ok = torch.equal(
                full[-1, :k].view(torch.uint8), w[-1].view(torch.uint8)
            )
            if first_ok and last_ok:
                return full

    return w.contiguous()


class RDNA4W8A8Fp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """Block-scaled FP8 GEMM using the gfx1201 fp8 WMMA kernels."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not envs.VLLM_ROCM_USE_HIP_W8A8:
            return False, "requires setting VLLM_ROCM_USE_HIP_W8A8=1"

        if not current_platform.is_rocm():
            return False, "requires ROCm"

        from vllm.platforms.rocm import on_gfx1201

        if not on_gfx1201():
            return False, "requires gfx1201 (RDNA4)"

        # Only built when gfx1201 is in the target arch list, so fall through
        # gracefully on a wheel built for other archs.
        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "w8a8_block_fp8_gemm_rdna4")
        ):
            return False, (
                "torch.ops._rocm_C.w8a8_block_fp8_gemm_rdna4 missing - "
                "rebuild the C++ extension with gfx1201 in the arch list"
            )

        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        ok, why = super().can_implement(config)
        if not ok:
            return ok, why

        ok, why = cls.is_supported()
        if not ok:
            return False, why

        if config.out_dtype not in SUPPORTED_OUT_DTYPES:
            return False, (
                f"out_dtype {config.out_dtype} unsupported; the kernels emit "
                f"fp32, fp16 or bf16"
            )

        act_group = config.activation_quant_key.scale.group_shape
        if act_group != GroupShape(1, BLOCK_K):
            return False, (
                f"activation group_shape must be (1, {BLOCK_K}), got {act_group}"
            )

        weight_group = config.weight_quant_key.scale.group_shape
        if weight_group != GroupShape(BLOCK_N, BLOCK_K):
            return False, (
                f"weight group_shape must be ({BLOCK_N}, {BLOCK_K}), got {weight_group}"
            )

        n, k = config.weight_shape
        if n % BLOCK_N != 0:
            return False, f"N={n} must be divisible by {BLOCK_N}"
        if k % BLOCK_K != 0:
            return False, f"K={k} must be divisible by {BLOCK_K}"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Transposing twice would restore the broken orientation, and on a
        # square scale grid nothing downstream could detect it.
        if getattr(layer, "_rdna4_w8a8_prepared", False):
            return

        params = self._get_layer_params(layer)
        scale_attr = params.block_scale_attr
        scale = params.block_scale
        replace_parameter(layer, scale_attr, scale.t().contiguous().to(torch.float32))
        replace_parameter(layer, params.WEIGHT, _widen_padded_weight(params.weight))
        layer._rdna4_w8a8_prepared = True

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        C = torch.empty(
            (A.shape[0], B.shape[0]), dtype=self.config.out_dtype, device=A.device
        )
        torch.ops._rocm_C.w8a8_block_fp8_gemm_rdna4(
            A.contiguous(), B, As.contiguous(), Bs, C, BLOCK_N, BLOCK_K
        )
        return C
