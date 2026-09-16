# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

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

    def rebuild_moe_kernel(self, layer, dry_run: bool = False) -> None:
        """Rebuild this layer's MoE kernel for the current all2all backend.

        Re-selects the experts class for the current activation format
        (Standard vs BatchedExperts) and rebuilds the prepare/finalize pair
        around it. Used by P/D role switching.

        The rebuild reads its parameters off ``layer.moe_config``, so the
        caller changes that first and nothing here checks it. Two shapes:

        - A budget change on one backend (deepep_v2): set
          ``moe_config.max_num_tokens``. The prepare/finalize built here asks
          the all2all manager for a handle at that budget, so the buffer is
          sized for the new role while the experts class stays the same.
        - A backend change (the high-throughput/low-latency pair): set
          ``layer.moe_config.moe_parallel_config.all2all_backend``, the
          per-layer copy -- ``use_batched_activation_format`` and its
          siblings are read-only properties over that field, and flipping
          the engine-wide parallel config alone reaches no layer's copy, so
          this would re-select the same backend and rebuild an identical
          kernel, a silent no-op that reports success. Also point the EP
          group's device communicator at the new manager first: the rebuild
          asks whichever manager is installed for a handle, and the outgoing
          one refuses the incoming role's arguments.

        Implementations must NOT touch weights. The batched and standard
        variants of one backend share a weight layout, which is what allows the
        switch to be cheap; re-running the load-time format conversion would
        both cost a full-size allocation and re-process weights that are
        already in kernel format.

        With ``dry_run``, run the selection and the compatibility checks
        (weight layout, activation format, hidden-size padding) but assign
        nothing. Role switching uses this to refuse an impossible switch
        before it has mutated anything, and running the real code path beats
        a second copy of the rules that could drift from it. Buffer
        allocation is not covered -- it happens inside the real rebuild, per
        layer, so a caller switching a whole model owns the rollback if a
        later layer fails to allocate.

        A method wrapped in ``FusedMoEModularMethod`` (LoRA, or after an
        elastic-EP scale) does not forward this and refuses; the wrapper runs
        its own kernel, so forwarding alone would not be enough.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support rebuilding its MoE kernel "
            f"for a different all2all backend."
        )

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        # NOTE(rob): temporary attribute to indicate support for
        # completed migration to the new internal MK interface.
        return (
            self.moe_kernel is not None and self.moe_kernel.can_overlap_shared_experts
        )

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

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        """
        Given layer hidden size and intermediate size per partition and MoE
        configurations, round up hidden_size and intermediate_size_per_partition
        if necessary.

        Args:
            hidden_size: Layer hidden-size
            intermediate_size_per_partition: Intermediate size per partition for
                the layer.
            act_dtype: Data type of the layer activations.
            moe_parallel_config: Fused MoE parallelization strategy configuration.

        Return:
            A tuple of (rounded_hidden_size, rounded_intermediate_size_per_partition),
            where:
                - rounded_hidden_size is the possibly rounded up hidden size.
                - rounded_intermediate_size_per_partition is the possibly rounded
                  up intermediate size per partition.
        """
        from .all2all_utils import maybe_roundup_layer_hidden_size

        return maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype, moe_parallel_config
        ), intermediate_size_per_partition

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
    def has_unpadded_output(self) -> bool:
        """
        Indicates that the hidden_states output might be the unpadded
        hidden_states shape rather than the full padded shape.
        """
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
        shared_experts: "SharedExperts | None",
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
            Output tensor from routed experts.
        """
        raise NotImplementedError

    def apply_monolithic(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        """
        Apply the MoE operation using monolithic kernels.

        Args:
            layer: RoutedExperts instance containing weight parameters
            x: Input tensor
            router_logits: Router logits (routing done internally)

        Returns:
            Finalized routed states or a deferred-finalize output.
        """
        raise NotImplementedError
