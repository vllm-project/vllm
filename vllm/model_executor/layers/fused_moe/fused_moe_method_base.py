# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
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
        layer: torch.nn.Module,
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

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalizeModular | None:
        pf = self._make_prepare_finalize(routing_tables=routing_tables)
        assert pf is None or isinstance(pf, FusedMoEPrepareAndFinalizeModular)
        return pf

    def _make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        moe_config: FusedMoEConfig | None = None,
        *,
        eep_stage: bool = False,
    ) -> mk.FusedMoEPrepareAndFinalize | None:
        from .all2all_utils import maybe_make_prepare_finalize

        return maybe_make_prepare_finalize(
            moe_config or self.moe,
            self.moe_quant_config,
            routing_tables,
            allow_new_interface=eep_stage,
            use_monolithic=self.is_monolithic,
            eep_stage=eep_stage,
        )

    def _make_eep_experts(
        self,
        source_experts: FusedMoEExpertsModular,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEExpertsModular:
        experts_cls = source_experts.__class__
        assert self.moe_quant_config is not None
        experts_kwargs: dict[str, Any] = {
            "moe_config": moe_config,
            "quant_config": self.moe_quant_config,
        }
        if (
            prepare_finalize.activation_format
            == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens is not None
            experts_kwargs.update(
                max_num_tokens=max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
            )

        # Expert kernels with extra init params need explicit EEP support.
        generic_arg_names = set(signature(mk.FusedMoEExperts.__init__).parameters)
        ctor_arg_names = set(signature(experts_cls.__init__).parameters)
        unsupported_args = ctor_arg_names - generic_arg_names
        missing_args = set(experts_kwargs) - ctor_arg_names
        if unsupported_args or missing_args:
            raise NotImplementedError(
                f"{experts_cls.__name__} experts do not support Elastic EP."
            )

        return experts_cls(**experts_kwargs)

    def eep_make_staged_quant_method(
        self,
        _layer: torch.nn.Module,
        moe: FusedMoEConfig,
    ) -> "FusedMoEMethodBase | None":
        if self.moe_kernel is None:
            return None
        if self.moe_kernel.is_monolithic:
            raise NotImplementedError(
                "Elastic EP full modular-kernel staging is not supported for "
                "monolithic fused MoE kernels."
            )
        if self.moe_quant_config is None:
            raise ValueError(
                "Elastic EP full modular-kernel staging requires initialized "
                "MoE quant config."
            )

        prepare_finalize = self._make_prepare_finalize(
            routing_tables=None,
            moe_config=moe,
            eep_stage=True,
        )
        assert prepare_finalize is not None
        assert isinstance(prepare_finalize, FusedMoEPrepareAndFinalizeModular)

        source_experts = self.moe_kernel.fused_experts
        assert isinstance(source_experts, FusedMoEExpertsModular)

        experts = self._make_eep_experts(
            source_experts,
            prepare_finalize,
            moe,
        )

        from .fused_moe_modular_method import FusedMoEModularMethod

        if isinstance(self, FusedMoEModularMethod):
            base_quant_method = self.old_quant_method
        else:
            base_quant_method = self
        return FusedMoEModularMethod(
            base_quant_method,
            mk.FusedMoEKernel(
                prepare_finalize,
                experts,
                inplace=self.moe_kernel.inplace,
            ),
        )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular:
        # based on the all2all implementation, select the appropriate
        # gemm implementation
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
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
        layer: "RoutedExperts",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def apply_monolithic(
        self,
        layer: "RoutedExperts",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
