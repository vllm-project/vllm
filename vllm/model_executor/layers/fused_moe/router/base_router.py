# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if current_platform.is_cuda_alike() or current_platform.is_xpu():

    @triton.jit
    def _eplb_map_and_record_i32_kernel(
        topk_ids_ptr,
        logical_replica_count_ptr,
        logical_to_physical_ptr,
        out_ids_ptr,
        out_ptr,
        record_enabled_ptr,
        num_unpadded_tokens_ptr,
        num_logical_experts,
        map_slots,
        out_size,
        numel,
        num_active_experts,
        HAS_NUM_UNPADDED: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
        valid_expert = (expert_id >= 0) & (expert_id < num_logical_experts)
        safe_expert_id = tl.where(valid_expert, expert_id, 0)

        # 1. Convert the logical expert ids to physical expert ids
        replica_count = tl.load(
            logical_replica_count_ptr + safe_expert_id,
            mask=mask & valid_expert,
            other=1,
        )
        # Avoid invalid modulo/div by forcing at least 1.
        replica_count = tl.maximum(replica_count, 1)
        # floor(2^32 / phi), classic Knuth multiplicative hash multiplier.
        KNUTH_MULTIPLIER = 2654435769
        token_idx = (offs // num_active_experts).to(tl.int64)
        hashed = (token_idx * KNUTH_MULTIPLIER) & 0xFFFFFFFF
        replica_idx = hashed % replica_count
        map_index = safe_expert_id * map_slots + replica_idx
        physical_id = tl.load(
            logical_to_physical_ptr + map_index,
            mask=mask & valid_expert,
            other=-1,
        )
        tl.store(out_ids_ptr + offs, physical_id, mask=mask)

        # 2. Record expert load metrics.

        # TODO(bowen): When using `FusedMoEModularKernel`, this
        # can be done in a more unified way, since
        # `FusedMoEPrepareAndFinalize` will return the expert
        # token count, in some cases directly from the kernel.
        # However, now there are many code paths not using
        # the modular kernel, e.g. calling `fused_experts`,
        # so we decide to keep the logic here.
        #
        # If later refactor moved all the MoE kernel calls
        # to the modular kernel, we can move this logic there
        # to achieve better efficiency.

        record_enabled = tl.load(record_enabled_ptr) != 0
        # Skip padded tokens when recording.
        if HAS_NUM_UNPADDED:
            num_unpadded_tokens = tl.load(num_unpadded_tokens_ptr)
            is_unpadded = offs < num_unpadded_tokens * num_active_experts
        else:
            is_unpadded = True
        valid = (
            mask
            & record_enabled
            & is_unpadded
            & (physical_id >= 0)
            & (physical_id < out_size)
        )
        safe_physical_id = tl.where(physical_id >= 0, physical_id, 0)
        tl.atomic_add(out_ptr + safe_physical_id, 1, mask=valid)

    def _eplb_map_and_record_triton(
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_load_view: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        topk_ids_in = topk_ids.contiguous().to(dtype=torch.int32)
        numel = topk_ids_in.numel()
        if numel == 0:
            return topk_ids
        num_active_experts = topk_ids_in.shape[-1]
        out_flat = torch.empty((numel,), device=topk_ids.device, dtype=topk_ids.dtype)
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        assert expert_load_view.is_contiguous()
        _eplb_map_and_record_i32_kernel[grid](
            topk_ids_in,
            logical_replica_count.contiguous(),
            logical_to_physical_map.contiguous(),
            out_flat,
            expert_load_view,
            record_enabled,
            num_unpadded_tokens,
            logical_replica_count.shape[0],
            logical_to_physical_map.shape[1],
            expert_load_view.shape[0],
            numel,
            num_active_experts,
            HAS_NUM_UNPADDED=num_unpadded_tokens is not None,
            BLOCK_SIZE=256,
        )
        return out_flat.reshape(topk_ids.shape)

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Fused triton implementation: mapping + optional recording in one kernel.
        return _eplb_map_and_record_triton(
            topk_ids=topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_view=expert_load_view,
            record_enabled=record_enabled,
            num_unpadded_tokens=num_unpadded_tokens,
        )
else:

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return topk_ids


class BaseRouter(FusedMoERouter):
    """Base class providing common functionality for all router implementations.

    This class implements the template method pattern where select_experts() handles
    common pre-processing and post-processing, delegating the actual routing logic
    to the abstract _compute_routing() method.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState | None = None,
    ):
        """Args:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts
        eplb_state: Optional EPLBLayerState for load balancing

        """
        super().__init__(eplb_state=eplb_state)
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        # EMA state for VLLM_MOE_BALANCE_ROUTING_EMA; lazy-initialized on
        # first forward pass once the device and ep_size are known.
        self._ema_load: torch.Tensor | None = None
        self._ema_prev_topk_ids: torch.Tensor | None = None
        self._ema_expert_to_rank: torch.Tensor | None = None
        # True once we have seen at least one CUDA-graph-capture pass.
        # Used to detect the capture→eager transition and reset stale state.
        self._ema_capturing_seen: bool = False
        # EP topology for EMA; read from the current VllmConfig at init time
        # so we use the layer's actual ep_size rather than the EP group size,
        # which differs when TP is run without --enable-expert-parallel.
        self._moe_use_ep: bool = False
        self._moe_ep_size: int = 1
        self._moe_placement: str = "linear"
        try:
            from vllm.config import get_current_vllm_config
            _cfg = get_current_vllm_config()
            _pc = _cfg.parallel_config
            # FusedMoEParallelConfig.make() sets use_ep only when EP is
            # active; we mirror that logic using the same source fields.
            _tp = _pc.tensor_parallel_size
            _dp = _pc.data_parallel_size
            _ep = getattr(_pc, "enable_expert_parallel", False)
            self._moe_use_ep = bool(_ep and _tp * _dp > 1)
            if self._moe_use_ep:
                self._moe_ep_size = _tp * _dp
            self._moe_placement = getattr(
                _pc, "expert_placement_strategy", "linear"
            )
        except Exception:
            pass

    def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:
        """Set a capture callback for logical routed expert IDs."""
        self.capture_fn = capture_fn

    def _validate_eplb_state(self) -> None:
        """Validate that EPLB state is properly initialized if EPLB is enabled."""
        if self.eplb_state is not None:
            eplb_state = self.eplb_state
            if eplb_state.expert_load_view is None:
                raise ValueError("EPLB requires expert_load_view != None")
            if eplb_state.logical_to_physical_map is None:
                raise ValueError("EPLB requires logical_to_physical_map != None")
            if eplb_state.logical_replica_count is None:
                raise ValueError("EPLB requires logical_replica_count != None")
            if eplb_state.should_record_tensor is None:
                raise ValueError("EPLB requires should_record_tensor != None")
            if eplb_state.num_unpadded_tokens_tensors is None:
                raise ValueError("EPLB requires num_unpadded_tokens_tensors != None")

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Apply EPLB mapping to convert logical expert IDs to physical expert IDs."""
        if self.eplb_state is not None:
            eplb_state = self.eplb_state
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_to_physical_map is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            assert eplb_state.num_unpadded_tokens_tensors is not None
            return eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                expert_load_view=eplb_state.expert_load_view,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ],
            )
        return topk_ids

    def _convert_indices_dtype(
        self, topk_ids: torch.Tensor, indices_type: torch.dtype | None
    ) -> torch.Tensor:
        """Convert topk_ids to the desired dtype if needed."""
        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None
        return topk_ids

    @abstractmethod
    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the actual routing logic.

        This method must be implemented by subclasses to provide the specific
        routing algorithm (e.g., grouped_topk, fused_topk, custom routing, etc.).

        Args:
            hidden_states: Input hidden states
            router_logits: Router logits for expert selection
            indices_type: Desired dtype for expert indices (may be None)
            input_ids: Token ids, for routers that condition on them

        Returns:
            tuple of (topk_weights, topk_ids)

        """
        raise NotImplementedError

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route the input hidden states to the top-k experts based on the
        router logits.

        This method implements the template method pattern:
        1. Validates EPLB state
        2. Calls _compute_routing() to get topk_weights and topk_ids
        3. Applies EPLB mapping if enabled
        4. Converts indices dtype if needed

        Returns:
            (topk_weights, topk_ids)
            (tuple[torch.Tensor, torch.Tensor]):
            The weights and expert ids computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.

        """
        # Step 1: Validate EPLB state
        self._validate_eplb_state()

        # Step 2: Compute routing (delegated to subclass).
        # If EMA load-aware routing is enabled, adjust logits before top-k
        # to proactively steer tokens away from overloaded EP ranks.
        if envs.VLLM_MOE_BALANCE_ROUTING_EMA and self._moe_use_ep:
            _capturing = (
                torch.compiler.is_compiling()
                or torch.cuda.is_current_stream_capturing()
            )
            # Detect the first eager step after CUDA graph capture and reset
            # stale EMA state.  Profile and warmup passes that precede capture
            # use all-zero dummy inputs, routing all tokens to the same top-k
            # experts and leaving a biased _ema_load.  _ema_prev_topk_ids
            # stored during capture points into graph-pool memory that replay
            # later overwrites, so it must also be cleared.
            if self._ema_capturing_seen and not _capturing:
                self._ema_load = None
                self._ema_prev_topk_ids = None
                self._ema_expert_to_rank = None
                self._ema_capturing_seen = False
            elif _capturing:
                self._ema_capturing_seen = True

            if not _capturing:
                if self.eplb_state is not None:
                    raise RuntimeError(
                        "VLLM_MOE_BALANCE_ROUTING_EMA is not compatible with "
                        "EPLB (--enable-eplb): the expert-to-rank mapping "
                        "is invalidated by EPLB weight rearrangement. "
                        "Disable one of them."
                    )
                from vllm.model_executor.layers.fused_moe.router \
                        .balance_routing_ema import (
                    adjust_logits_ema,
                    build_expert_to_rank,
                )
                if self._ema_load is None:
                    dev = router_logits.device
                    self._ema_load = torch.zeros(
                        self._moe_ep_size, dtype=torch.float32, device=dev
                    )
                    self._ema_expert_to_rank = build_expert_to_rank(
                        num_experts=self.global_num_experts,
                        ep_size=self._moe_ep_size,
                        placement=self._moe_placement,
                        device=dev,
                    )
                router_logits = adjust_logits_ema(
                    router_logits=router_logits,
                    ema_load=self._ema_load,
                    prev_topk_ids=self._ema_prev_topk_ids,
                    expert_to_rank=self._ema_expert_to_rank,
                    num_experts=self.global_num_experts,
                    alpha=envs.VLLM_MOE_BALANCE_ROUTING_EMA_ALPHA,
                    lambda_=envs.VLLM_MOE_BALANCE_ROUTING_EMA_LAMBDA,
                )

        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids
        )

        # Store logical topk_ids for the next step's EMA update.
        # Skipped during capture: topk_ids would point into graph-pool memory
        # that replay overwrites, corrupting the next eager step's update.
        if (
            envs.VLLM_MOE_BALANCE_ROUTING_EMA
            and self._moe_use_ep
            and not torch.compiler.is_compiling()
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._ema_prev_topk_ids = topk_ids.detach()

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Step 3: Apply EPLB mapping
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # Step 4: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, topk_indices_dtype)

        return topk_weights, topk_ids
