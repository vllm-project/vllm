# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

logger = init_logger(__name__)


class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe: FusedMoEConfig = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None

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

    def maybe_make_prepare_finalize(self) -> FusedMoEPrepareAndFinalize | None:
        from .all2all_utils import maybe_make_prepare_finalize

        return maybe_make_prepare_finalize(self.moe, self.moe_quant_config)

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        # based on the all2all implementation, select the appropriate
        # gemm implementation
        raise NotImplementedError(
            f"{self.__class__.__name__} must select appropriate gemm "
            "implementation based on the prepare_finalize"
        )

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def allow_inplace(self) -> bool:
        return False

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
