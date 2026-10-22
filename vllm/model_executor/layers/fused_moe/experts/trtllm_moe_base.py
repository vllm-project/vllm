# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared base for the trtllm-gen fused-MoE backends."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExperts,
)
from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_trtllm_fused_moe_workspace_size,
    has_flashinfer_trtllm_fused_moe,
    has_flashinfer_trtllm_fused_moe_workspace,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEQuantConfig,
    )

logger = init_logger(__name__)

# Workspace size per (problem, quant mode) configuration. The trtllm-gen size
# query walks the launcher's allocation code for every tile size, so it is
# cached process-wide instead of once per MoE layer.
_workspace_nbytes: dict[tuple, int] = {}


class TrtLlmExpertsBase(FusedMoEExperts):
    """Shared base of the trtllm-gen fused-MoE backends.

    Reserves one trtllm-gen MoE workspace and shares it across MoE layers.
    Without a caller-owned workspace, FlashInfer's trtllm-gen launchers
    allocate and free their routing maps, intermediate GEMM outputs and
    batched-GEMM workspaces through the PyTorch caching allocator on every MoE
    layer call, which fragments the allocator during large prefills. Size one
    buffer for the largest batch once, during memory profiling, and share it
    across all MoE layers that run sequentially in the current ubatch and lane.

    The constructor caches the config-derived fields common to every backend
    and initializes the workspace; subclasses call
    ``super().__init__(moe_config, quant_config)`` and add their own fields.
    """

    moe_config: "FusedMoEConfig"
    local_num_experts: int

    def __init__(
        self,
        moe_config: "FusedMoEConfig",
        quant_config: "FusedMoEQuantConfig",
    ) -> None:
        super().__init__(moe_config, quant_config)
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.routing_method_type = moe_config.routing_method
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self._init_trtllm_moe_workspace()

    @staticmethod
    def activation_format() -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        """Supports only Blackwell-family GPUs."""
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and has_flashinfer_trtllm_fused_moe()
        )

    def _init_trtllm_moe_workspace(self) -> None:
        # The tuning ceiling already includes all contributing DP ranks.
        self._workspace_rows = fi_moe_largest_bucket(self.moe_config)

    def _get_workspace(
        self,
        hidden_states: torch.Tensor,
        *,
        do_finalize: bool = True,
        quant_mode: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation_type: int,
        use_shuffled_weight: bool = True,
        weight_layout: int | None = None,
        use_routing_scales_on_input: bool = False,
        use_per_token_scale: bool = False,
        valid_hidden_size: int | None = None,
        valid_intermediate_size: int | None = None,
        num_fused_shared_experts: int = 0,
    ) -> torch.Tensor | None:
        """Return the persistent trtllm-gen MoE workspace, or ``None``.

        The buffer is sized for the largest batch, so smaller batches reuse it
        without another reservation. When FlashInfer lacks the workspace API,
        ``None`` keeps FlashInfer's per-call workspace allocation.
        """
        if (
            # A caller-owned workspace requires finalized output.
            not do_finalize
            or not has_flashinfer_trtllm_fused_moe_workspace()
            or not is_workspace_manager_initialized()
            # Unexpectedly large batches keep the per-call allocation rather
            # than pinning one more persistent buffer per distinct size.
            or hidden_states.size(0) > self._workspace_rows
        ):
            return None

        rows = self._workspace_rows
        device = hidden_states.device
        size_key = (
            rows,
            quant_mode,
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            self.local_num_experts,
            num_fused_shared_experts,
            activation_type,
            use_shuffled_weight,
            weight_layout,
            use_routing_scales_on_input,
            use_per_token_scale,
            valid_hidden_size,
            valid_intermediate_size,
            device,
        )
        nbytes = _workspace_nbytes.get(size_key)
        if nbytes is None:
            nbytes = int(
                flashinfer_trtllm_fused_moe_workspace_size(
                    rows,
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    top_k,
                    quant_mode=quant_mode,
                    local_num_experts=self.local_num_experts,
                    num_fused_shared_experts=num_fused_shared_experts,
                    activation_type=activation_type,
                    use_shuffled_weight=use_shuffled_weight,
                    weight_layout=weight_layout,
                    use_routing_scales_on_input=use_routing_scales_on_input,
                    use_per_token_scale=use_per_token_scale,
                    valid_hidden_size=valid_hidden_size,
                    valid_intermediate_size=valid_intermediate_size,
                    device=device,
                )
            )
            _workspace_nbytes[size_key] = nbytes
        if nbytes <= 0:
            return None

        key = ("trtllm_gen_moe_workspace", device, nbytes)

        def _allocate() -> torch.Tensor:
            logger.info_once(
                "Reserving a persistent %.2f GiB trtllm-gen MoE workspace "
                "for up to %d tokens.",
                nbytes / (1 << 30),
                rows,
            )
            return torch.empty((nbytes,), dtype=torch.uint8, device=device)

        try:
            return current_workspace_manager().get_persistent_resource(key, _allocate)
        except AssertionError:
            # The workspace is locked and this size was not reserved in warmup.
            logger.warning_once(
                "trtllm-gen MoE workspace of %d bytes was not reserved during "
                "warmup; falling back to per-call allocation.",
                nbytes,
            )
            return None
