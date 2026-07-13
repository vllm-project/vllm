# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.hw_agnostic.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )
    from vllm.model_executor.hw_agnostic.layers.fused_moe.runner.shared_experts import (  # noqa: E501
        SharedExperts,
    )

logger = init_logger(__name__)


class FusedMoEMethodBase(QuantizeMethodBase):
    """ABC for hw-agnostic FusedMoE quant methods.

    Concrete subclasses in this tree: ``UnquantizedFusedMoEMethod`` and
    ``Fp8MoEMethod`` (offline + ``Fp8OnlineMoEMethod`` online variant).
    Subclasses build their experts kernel in
    ``process_weights_after_loading`` and run it via ``fused_moe_forward``.
    """

    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe: FusedMoEConfig = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.experts: FusedMoEExpertsModular | None = None

    @abstractmethod
    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        return hidden_size, intermediate_size_per_partition

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        # The AllGather/ReduceScatter transport places no dtype constraint on
        # topk_ids; the router keeps its native dtype.
        return None

    @property
    def skip_forward_padding(self) -> bool:
        return False

    @property
    def has_unpadded_output(self) -> bool:
        return False

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def method_name(self) -> str:
        return self.__class__.__name__

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Modular MoE forward (router runs first; pre-computed topk in)."""
        raise NotImplementedError
