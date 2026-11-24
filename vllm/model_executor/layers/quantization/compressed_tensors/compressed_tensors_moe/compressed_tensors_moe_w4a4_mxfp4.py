# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    FLASHINFER_MOE_EP_MXFP4_BACKENDS,
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.mxfp4 import PackedMxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


class CompressedTensorsW4A4Mxfp4MoEMethod(PackedMxfp4MoEMethod):
    """Packed MXFP4 checkpoints with optional MXFP4 activations."""

    def __init__(self, moe: FusedMoEConfig, activation_key: QuantKey | None = None):
        super().__init__(moe, activation_key=activation_key)
        self.group_size = 32

    def _select_backend(
        self,
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts] | None]:
        return select_mxfp4_moe_backend(self.moe, activation_key=self.activation_key)

    @property
    def supports_eplb(self) -> bool:
        return self.mxfp4_backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS
