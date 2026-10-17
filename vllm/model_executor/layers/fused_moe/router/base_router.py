# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.shared_expert_fusion import (
    SharedExpertFusion,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)

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
        replica_prob_ptr,
        physical_ids_ptr,
        num_logical_experts,
        map_slots,
        out_size,
        numel,
        num_active_experts,
        HAS_NUM_UNPADDED: tl.constexpr,
        HAS_REPLICA_PROB: tl.constexpr,
        HAS_PHYSICAL_IDS: tl.constexpr,
        PROB_SLOTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
        valid_expert = (expert_id >= 0) & (expert_id < num_logical_experts)
        safe_expert_id = tl.where(valid_expert, expert_id, 0)

        # 1. Convert the logical expert ids to physical expert ids.
        # A policy that resolved the replica itself needs none of this:
        # the count load and the hash exist only to pick a column, and
        # leaving them in charged the direct path for a choice it had
        # already made.
        if not HAS_PHYSICAL_IDS:
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

        if HAS_REPLICA_PROB:
            # A load-balancing policy supplied per-expert replica shares. Use
            # the same hash as a uniform draw and invert the row's CDF, so the
            # choice stays a pure function of the token index: every rank
            # computes the same replica for the same token without exchanging
            # anything, and no RNG state enters the CUDA graph.
            row = safe_expert_id * PROB_SLOTS
            total = tl.zeros_like(hashed).to(tl.float32)
            for j in tl.static_range(PROB_SLOTS):
                total += tl.load(
                    replica_prob_ptr + row + j, mask=mask & valid_expert, other=0.0
                )
            # 2^32; hashed is already reduced modulo it.
            draw = (hashed.to(tl.float32) / 4294967296.0) * total
            acc = tl.zeros_like(total)
            sampled = tl.zeros_like(replica_idx)
            for j in tl.static_range(PROB_SLOTS):
                acc += tl.load(
                    replica_prob_ptr + row + j, mask=mask & valid_expert, other=0.0
                )
                # Index = how many prefix sums stay at or below the draw.
                sampled += tl.where(acc <= draw, 1, 0)
            sampled = tl.minimum(sampled, replica_count - 1)
            # An all-zero row means "no preference"; keep the hash's uniform
            # pick rather than collapsing every token onto replica 0.
            replica_idx = tl.where(total > 0.0, sampled, replica_idx)

        if HAS_PHYSICAL_IDS:
            # The policy resolved the replica itself. Selection and mapping are
            # skipped; recording below is not, which is the whole reason this
            # path exists rather than the caller writing out_ids and losing the
            # load accounting that shares this kernel.
            physical_id = tl.load(
                physical_ids_ptr + offs, mask=mask & valid_expert, other=-1
            ).to(tl.int64)
        else:
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
        replica_prob: torch.Tensor | None = None,
        physical_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_ids_in = topk_ids.contiguous().to(dtype=torch.int32)
        numel = topk_ids_in.numel()
        if numel == 0:
            return topk_ids
        num_active_experts = topk_ids_in.shape[-1]
        out_flat = torch.empty((numel,), device=topk_ids.device, dtype=topk_ids.dtype)
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        assert expert_load_view.is_contiguous()
        map_slots = logical_to_physical_map.shape[1]
        # The share table is scanned with a compile-time loop, so its width has
        # to be the number of replicas that can exist -- not the width of the
        # candidate map, which is padded to MAX_EXPERT_REDUNDANCY + 1 (1024)
        # however few redundant experts are configured. Unrolling that twice is
        # 2048 loads per element and takes Triton well past vLLM's RPC timeout
        # to compile. A narrower table is a prefix of the same columns, so a
        # sampled index still resolves against the full-width map below.
        prob_slots = map_slots
        if replica_prob is not None:
            assert (
                replica_prob.shape[0] == logical_to_physical_map.shape[0]
                and replica_prob.shape[1] <= map_slots
            ), (
                "replica shares must be a column prefix of the candidate map: "
                f"{tuple(replica_prob.shape)} vs "
                f"{tuple(logical_to_physical_map.shape)}"
            )
            prob_slots = replica_prob.shape[1]
            logger.info_once(
                "MLB replica-share routing active: table width %d, map width %d",
                prob_slots,
                map_slots,
            )
            replica_prob = replica_prob.contiguous().to(torch.float32)
        if physical_ids is not None:
            assert physical_ids.shape == topk_ids.shape, (
                "resolved physical ids must match the TopK shape: "
                f"{tuple(physical_ids.shape)} vs {tuple(topk_ids.shape)}"
            )
            physical_ids = physical_ids.contiguous().to(torch.int32)
            logger.info_once(
                "MLB direct-dispatch routing active: the policy resolves "
                "replicas itself; load recording stays in this kernel"
            )
        _eplb_map_and_record_i32_kernel[grid](
            topk_ids_in,
            logical_replica_count.contiguous(),
            logical_to_physical_map.contiguous(),
            out_flat,
            expert_load_view,
            record_enabled,
            num_unpadded_tokens,
            replica_prob,
            physical_ids,
            logical_replica_count.shape[0],
            map_slots,
            expert_load_view.shape[0],
            numel,
            num_active_experts,
            HAS_NUM_UNPADDED=num_unpadded_tokens is not None,
            HAS_REPLICA_PROB=replica_prob is not None,
            HAS_PHYSICAL_IDS=physical_ids is not None,
            PROB_SLOTS=prob_slots,
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
        *,
        layer_state: EplbLayerState | None = None,
        topk_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # A pluggable load balancer decides how each logical expert's traffic
        # should be split across its replicas; it does not need to own the
        # per-token step. Ask it for that split and keep the fused kernel --
        # which means load recording, the record gate and the padding mask all
        # stay exactly as they are, with nothing reimplemented alongside them.
        replica_prob = None
        physical_ids = None
        if layer_state is not None and topk_weights is not None:
            from vllm.distributed.eplb.mlb_runtime import get_mlb_routing

            routing = get_mlb_routing()
            # A policy that decides from the committed placement alone has
            # nothing to add per forward: its answer was already materialised
            # into a one-column candidate map when the placement committed, so
            # substitute that map and skip the call. This is the same shape as
            # SGLang's `static`, which is one tensor index inside its own TopK
            # path -- there is no external call there to pay for, and there
            # should not be one here either. Measured at 426 us per layer per
            # forward, or 26 ms of a ~1 s forward, which is what made `static`
            # trail the built-in selection it reproduces exactly.
            _layer_id = getattr(layer_state, "moe_layer_idx", None)
            fixed = (
                routing.fixed_dispatch_maps(_layer_id)
                if routing is not None
                and getattr(routing, "dispatch_fixed_by_placement", False)
                and _layer_id is not None
                else None
            )
            if fixed is not None:
                logical_to_physical_map, logical_replica_count = fixed
            elif routing is not None and routing.requires_post_topk_routing:
                replica_prob, physical_ids = routing.resolve_routing(
                    topk_ids, topk_weights, layer_state, num_unpadded_tokens
                )

        # Fused triton implementation: mapping + optional recording in one kernel.
        return _eplb_map_and_record_triton(
            topk_ids=topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_view=expert_load_view,
            record_enabled=record_enabled,
            num_unpadded_tokens=num_unpadded_tokens,
            replica_prob=replica_prob,
            physical_ids=physical_ids,
        )
else:

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None = None,
        *,
        layer_state: EplbLayerState | None = None,
        topk_weights: torch.Tensor | None = None,
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
        # Set by FusedMoEFactory when the layer dispatches its shared expert.
        # None -- the default -- leaves every routing path byte-identical to
        # what it was, which is what makes this safe to leave in the template
        # method rather than behind a separate router subclass.
        self.shared_expert_fusion: SharedExpertFusion | None = None

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

    def _apply_eplb_mapping(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
                layer_state=eplb_state,
                topk_weights=topk_weights,
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

        # Step 2: Compute routing (delegated to subclass)
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids
        )

        # Capture logical ids before EPLB mapping.
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # Step 3: Apply EPLB mapping
        topk_ids = self._apply_eplb_mapping(topk_ids, topk_weights)

        # Step 3.5: Give the shared expert a home rank and a top-k column.
        #
        # After the EPLB mapping, not before: the shared slot is already
        # physical and has no logical identity to map, and EPLB's expert-load
        # accounting covers routed experts only -- recording a shared-slot id
        # in it would attribute traffic to a logical expert that does not
        # exist.
        if self.shared_expert_fusion is not None:
            topk_ids, topk_weights = self._append_shared_expert(topk_ids, topk_weights)

        # Step 4: Convert indices dtype
        topk_ids = self._convert_indices_dtype(topk_ids, topk_indices_dtype)

        return topk_weights, topk_ids

    def _append_shared_expert(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-map routed ids into the widened space and append the shared slot.

        The home rank comes from an L2 policy when one routes the shared
        expert (MLB's Waterfill), and is this rank otherwise. Falling back to
        this rank rather than refusing to run keeps the fusion testable on its
        own: it places the shared expert exactly where the replicated MLP did,
        so any output difference is the dispatch detour and not a routing
        decision.
        """
        fusion = self.shared_expert_fusion
        assert fusion is not None

        shared_rank = None
        eplb_state = self.eplb_state
        if (
            eplb_state is not None
            and eplb_state.num_unpadded_tokens_tensors is not None
        ):
            from vllm.distributed.eplb.mlb_runtime import get_mlb_routing

            routing = get_mlb_routing()
            if routing is not None and routing.routes_shared_expert:
                shared_rank = routing.resolve_shared_expert_rank(
                    topk_ids,
                    topk_weights,
                    eplb_state,
                    eplb_state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()],
                )

        return fusion.append_shared_expert(topk_ids, topk_weights, shared_rank)
