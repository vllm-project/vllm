# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.kernel import FLASHINFER_MOE_EP_BACKENDS
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    FLASHINFER_MOE_EP_MXFP4_BACKENDS,
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.mxfp4 import PackedMxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic


class INCMxfp4MoEMethod(PackedMxfp4MoEMethod):
    """Packed MXFP4 MoE for AutoRound checkpoints."""

    def __init__(self, moe: FusedMoEConfig):
        if moe.moe_backend in FLASHINFER_MOE_EP_BACKENDS:
            raise ValueError(
                f"moe_backend={moe.moe_backend!r} is not supported "
                "for AutoRound (INC) MXFP4 MoE checkpoints"
            )
        super().__init__(moe, activation_key=kMxfp4Dynamic)
        self.group_size = 32

    def _select_backend(
        self,
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts] | None]:
        return select_mxfp4_moe_backend(self.moe, activation_key=self.activation_key)

    @property
    def supports_eplb(self) -> bool:
        return self.mxfp4_backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS
