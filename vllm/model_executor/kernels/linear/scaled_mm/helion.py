# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.config import (
    CUDAGraphMode,
    get_current_vllm_config,
)
from vllm.kernels.helion.ops.scaled_mm import (
    scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import direct_register_custom_op

from .cutlass import CutlassFP8ScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

HELION_SCALED_MM_MAX_NUM_TOKENS = 64


class HelionFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    Hybrid Helion / Cutlass FP8 scaled_mm kernel

    Dispatches between Helion and CUTLASS based on the input batch size (M):
    - Small batches (M <= min(64, max_cudagraph_capture_size)): Use Helion.
    - Large batches: use CUTLASS.

    Restricting Helion to small batches:
    - reduces autotuning time and config space
    - avoids requiring large max_cudagraph_capture_size to cover all batch sizes
    - focuses Helion on the batch sizes where it provides the most benefit.
    """

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)
        self.fallback: CutlassFP8ScaledMMLinearKernel = CutlassFP8ScaledMMLinearKernel(
            c, layer_param_names
        )
        vllm_config = get_current_vllm_config().compilation_config
        self.max_num_tokens = min(
            vllm_config.max_cudagraph_capture_size, HELION_SCALED_MM_MAX_NUM_TOKENS
        )

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not has_helion():
            return False, "Helion kernel requires helion to be installed."

        if not current_platform.is_cuda():
            return False, "requires CUDA."

        # TODO(xiaohongchen1991): support blackwell hardwares when CuteDSL
        # backend supported by Helion
        if not current_platform.is_device_capability(90):
            return (
                False,
                "HelionFP8ScaledMMLinearKernel is only supported on "
                "SM90 (Hopper) architecture.",
            )

        # Helion kernel is disabled if there is no config exists for the hardware used.
        if scaled_mm._disabled:
            return False, "scaled_mm._disabled_reason"

        # Require CUDA graph capture and reply for Helion kernel
        vllm_config = get_current_vllm_config().compilation_config
        if (
            vllm_config.cudagraph_mode == CUDAGraphMode.NONE
            or vllm_config.max_cudagraph_capture_size == 0
        ):
            return False, "Helion kernel requires enabling CUDA Graph mode."

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def input_quant_key(self) -> QuantKey | None:
        """Only static per-tensor activation quantization is supported for external
        quantization."""
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        padded_k, padded_n = B.shape
        output_size = self.fallback.logical_output_size
        assert output_size is not None
        pad_k = padded_k - A.shape[1]
        pad_n = padded_n - output_size

        if pad_k > 0:
            A = self.fallback._pad_to_alignment(A, dim=1, alignment=16)
        if pad_n > 0 and bias is not None:
            bias = self.fallback._pad_to_alignment(bias, dim=0, alignment=16)

        cutlass_compatible_b = padded_k % 16 == 0 and padded_n % 16 == 0
        if not cutlass_compatible_b:
            from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
                triton_scaled_mm,
            )

            output = triton_scaled_mm(A, B, As, Bs, out_dtype, bias)
        else:
            output = torch.empty(
                (A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device
            )
            torch.ops.vllm.helion_scale_mm(
                output, A, B, As, Bs, bias, self.max_num_tokens
            )

        if pad_n > 0:
            output = output[..., :output_size].contiguous()

        return output.view(*output_shape[:-1], output_size)


def _helion_scale_mm_impl(
    output: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None,
    max_num_tokens: int,
) -> None:
    if A.shape[0] <= max_num_tokens:
        scaled_mm(output, A, B, As, Bs, bias)
    else:
        torch.ops._C.cutlass_scaled_mm(output, A, B, As, Bs, bias)


def _helion_scale_mm_fake(
    output: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None,
    max_num_tokens: int,
) -> None:
    return


direct_register_custom_op(
    "helion_scale_mm",
    _helion_scale_mm_impl,
    mutates_args=["output"],
    fake_impl=_helion_scale_mm_fake,
)
