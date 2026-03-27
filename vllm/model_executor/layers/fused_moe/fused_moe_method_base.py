# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )

logger = init_logger(__name__)


class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe: FusedMoEConfig = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.moe_kernel: mk.FusedMoEKernel | None = None

    @property
    def supports_internal_mk(self) -> bool:
        # NOTE(rob): temporary attribute to indicate support for
        # completed migration to the new internal MK interface.
        return self.moe_kernel is not None

    @property
    def mk_owns_shared_expert(self) -> bool:
        # NOTE(rob): temporary attribute to indicate support for
        # completed migration to the new internal MK interface.
        return self.moe_kernel is not None and self.moe_kernel.owns_shared_experts

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

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        Returns True if this quantization method uses 'weight_scale_2' pattern
        for per-tensor weight scales (e.g., FP4 variants), False otherwise.

        This method should be overridden by subclasses that use the
        'weight_scale_2' pattern instead of the standard 'weight_scale' pattern.
        """
        return False

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        from .all2all_utils import maybe_make_prepare_finalize

        pf = maybe_make_prepare_finalize(
            self.moe, self.moe_quant_config, routing_tables
        )
        assert pf is None or isinstance(pf, FusedMoEPrepareAndFinalizeModular)
        return pf

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: "RoutedExperts",
    ) -> FusedMoEExpertsModular:
        # based on the all2all implementation, select the appropriate
        # gemm implementation
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        if self.moe_kernel is not None:
            return self.moe_kernel.prepare_finalize.topk_indices_dtype()
        return None

    @property
    def skip_forward_padding(self) -> bool:
        """Whether to skip the padding in the forward before applying the moe method."""
        return False

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def method_name(self) -> str:
        return self.__class__.__name__

    @property
    def is_monolithic(self) -> bool:
        if self.moe_kernel is None:
            if hasattr(self, "experts_cls"):
                return self.experts_cls.is_monolithic()
            else:
                return False
        return self.moe_kernel.is_monolithic

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Apply the MoE operation using modular kernels.

        Args:
            layer: RoutedExperts instance containing weight parameters
            x: Input tensor
            topk_weights: Expert weights from router
            topk_ids: Selected expert IDs from router
            shared_experts_input: Input for shared experts (if any)

        Returns:
            Output tensor from routed experts
        """
        raise NotImplementedError

    def apply_monolithic(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the MoE operation using monolithic kernels.

        Args:
            layer: RoutedExperts instance containing weight parameters
            x: Input tensor
            router_logits: Router logits (routing done internally)

        Returns:
            Output tensor from routed experts
        """
        raise NotImplementedError
