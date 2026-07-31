# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)


class MoEPrepareFinalizeFactory(ABC):
    """Factory for a MoE Prepare/Finalize backend."""

    backend_name: ClassVar[str]
    supports_sequence_parallel: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def create(
        cls,
        *,
        moe: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig | None,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        allow_new_interface: bool,
        use_monolithic: bool,
        eep_stage: bool,
        all2all_manager: Any,
    ) -> FusedMoEPrepareAndFinalize: ...
