# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Port of upstream vllm-project/vllm#38815 (commit df1e30e74) adapted to the
# cohere fork. Upstream calls `init_mxfp8_linear_kernel()` introduced by
# vllm-project/vllm#39205 which refactors MXFP8 linear ops into a kernel
# abstraction. The cohere fork is still on the pre-#39205 `Mxfp8LinearOp`
# API, so this scheme wires into `Mxfp8LinearOp` directly (mirroring what
# `Mxfp8OnlineLinearMethod` does in `vllm/model_executor/layers/quantization/mxfp8.py`).
#
# Delete this file after this fork merges upstream past df1e30e74 and its
# prerequisite #39205 (MxFp8LinearKernel refactor); at that point upstream's
# own version of this scheme applies cleanly.
#
# Upstream-PR: vllm-project/vllm#38815
# Upstream-Commit: df1e30e74be91bd48fd015c417e7699310a64119
# Drop-After-Upstream-Merged: vllm-project/vllm#39205

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    Mxfp8LinearBackend,
    Mxfp8LinearOp,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import replace_parameter

__all__ = ["CompressedTensorsW8A8Mxfp8"]

logger = init_logger(__name__)


class CompressedTensorsW8A8Mxfp8(CompressedTensorsScheme):
    """
    Compressed tensors scheme for MXFP8 quantization (W8A8).

    Loads pre-quantized MXFP8 weights from compressed-tensors checkpoints.
    Activations are dynamically quantized to MXFP8 at runtime.

    MXFP8 format:
    - 8-bit float weights (E4M3) stored as float8_e4m3fn
    - Per-group E8M0 scales (uint8) with group_size=32
    - Activations dynamically quantized to MXFP8 during inference
    """

    def __init__(self):
        self.out_dtype = torch.get_default_dtype()
        self.mxfp8_linear = Mxfp8LinearOp(self._select_backend())
        logger.info_once(
            "Using %s backend for MXFP8 GEMM", self.mxfp8_linear.backend.value
        )

    @staticmethod
    def _select_backend() -> Mxfp8LinearBackend:
        try:
            from vllm.utils import flashinfer as fi

            _ = fi.mm_mxfp8
            return Mxfp8LinearBackend.FLASHINFER_CUTLASS
        except Exception:
            logger.warning(
                "FlashInfer mm_mxfp8 not available, "
                "falling back to MXFP8 emulation backend."
            )
            return Mxfp8LinearBackend.EMULATION

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires input_size_per_partition "
                f"({input_size_per_partition}) to be divisible by "
                f"{MXFP8_BLOCK_SIZE}."
            )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // MXFP8_BLOCK_SIZE,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data

        if self.mxfp8_linear.backend == Mxfp8LinearBackend.FLASHINFER_CUTLASS:
            N, K = weight.shape[0], weight.shape[1]
            weight_scale = swizzle_mxfp8_scale(weight_scale, N, K)

        layer.input_scale = None
        replace_parameter(layer, "weight", weight.contiguous())
        replace_parameter(layer, "weight_scale", weight_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mxfp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            bias=bias,
        )
