# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoonEP (https://github.com/MoonshotAI/MoonEP) prepare/finalize.

BF16 correctness-first proof of concept on top of vLLM's modular kernel
interface.

MoonEP differs from DeepEP-style backends in two ways that shape this
integration:

- ``dispatch`` returns tokens already grouped by expert *row* (a fixed
  ``[NvS, H]`` layout with ``NvS = S x K`` real slots plus padding) together
  with a ``cu_seqlens[E+B]`` segment table and an opaque ``plan``. There is
  no per-token topk id tensor after dispatch; the expert compute must be a
  grouped GEMM over ``cu_seqlens`` segments.
- Rows ``[E, E+B)`` of the weight/segment space are dynamic redundant-expert
  prefetch slots. ``plan.experts_to_copy`` names the source expert of each
  slot and ``Buffer.prefetch_weight`` must run between dispatch and expert
  compute.

PoC limitations:
- BF16 / unquantized only, eager only.
- Expert weights are replicated in global-expert order on every rank
  (memory-heavy). Production Kimi-K3 serving requires sharded
  symmetric-memory expert ownership, where rows ``[0, E)`` physically alias
  each home rank's parameter memory.
- Route weights are applied inside the expert compute and MoonEP's
  ``combine`` performs the K-sum, so ``finalize`` requires
  ``TopKWeightAndReduceNoOP``.
"""

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

MOONEP_DEFAULT_NUM_PREFETCH_SLOTS = 4
MOONEP_DEFAULT_TOKEN_PADDING = 128
MOONEP_DEFAULT_NUM_SMS = 32
# MoonEP's weight-prefetch kernel tiles hidden and intermediate dims by 128.
MOONEP_WEIGHT_TILE = 128


class MoonEPExpertWeightLayout(NamedTuple):
    """Contiguous BF16 expert weights in MoonEP ``[E+B, ...]`` layout.

    Rows ``[0, E)`` hold expert weights in global expert order; rows
    ``[E, E+B)`` are mutable prefetch slots filled by
    ``Buffer.prefetch_weight``.

    The prefetch slots never need re-zeroing between calls: the planner
    marks unused slots with ``-1`` in ``plan.experts_to_copy`` (skipped by
    ``prefetch_weight``) and gives them an empty ``cu_seqlens`` segment, so
    stale slot contents are never read; used slots are fully overwritten.
    """

    full_gate_weight: torch.Tensor
    full_up_weight: torch.Tensor
    full_down_weight: torch.Tensor
    num_prefetch_slots: int


def make_moonep_weight_layout(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    num_prefetch_slots: int,
) -> MoonEPExpertWeightLayout:
    """Build the replicated ``[E+B, ...]`` PoC weight layout.

    ``w13_weight`` must be ``[E, 2I, H]`` (gate rows first) and ``w2_weight``
    ``[E, H, I]``, both BF16 in global expert order.
    """
    if num_prefetch_slots <= 0:
        raise ValueError(
            f"num_prefetch_slots must be positive, got {num_prefetch_slots}"
        )
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise NotImplementedError("MoonEP PoC supports BF16 weights only.")
    num_experts, two_i, hidden_size = w13_weight.shape
    intermediate_size = two_i // 2
    if tuple(w2_weight.shape) != (num_experts, hidden_size, intermediate_size):
        raise ValueError(
            f"w2_weight shape {tuple(w2_weight.shape)} does not match "
            f"w13_weight shape {tuple(w13_weight.shape)}"
        )
    if hidden_size % MOONEP_WEIGHT_TILE or intermediate_size % MOONEP_WEIGHT_TILE:
        raise ValueError(
            "MoonEP weight prefetch requires hidden_size and intermediate_size "
            f"to be multiples of {MOONEP_WEIGHT_TILE}; got H={hidden_size}, "
            f"I={intermediate_size}"
        )

    full_gate_weight = torch.empty(
        num_experts + num_prefetch_slots,
        intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=w13_weight.device,
    )
    full_up_weight = torch.empty_like(full_gate_weight)
    full_down_weight = torch.empty(
        num_experts + num_prefetch_slots,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=w2_weight.device,
    )

    full_gate_weight[:num_experts].copy_(w13_weight[:, :intermediate_size, :])
    full_up_weight[:num_experts].copy_(w13_weight[:, intermediate_size:, :])
    full_down_weight[:num_experts].copy_(w2_weight)
    full_gate_weight[num_experts:].zero_()
    full_up_weight[num_experts:].zero_()
    full_down_weight[num_experts:].zero_()

    return MoonEPExpertWeightLayout(
        full_gate_weight=full_gate_weight.contiguous(),
        full_up_weight=full_up_weight.contiguous(),
        full_down_weight=full_down_weight.contiguous(),
        num_prefetch_slots=num_prefetch_slots,
    )


def gather_moonep_weight_layout(
    w13_local: torch.Tensor,
    w2_local: torch.Tensor,
    num_global_experts: int,
    num_prefetch_slots: int,
) -> MoonEPExpertWeightLayout:
    """Build the replicated ``[E+B, ...]`` layout from this rank's local experts.

    PoC bridge: each EP rank loads only its own experts (linear placement),
    so all-gather them once at load time into global expert order on every
    rank. Production MoonEP instead maps rows ``[0, E)`` onto each home
    rank's parameter memory via symmetric memory (RFC #52095 item 5).
    """
    from vllm.distributed import get_ep_group

    ep_group = get_ep_group()
    ep_size = ep_group.world_size
    if ep_size == 1:
        return make_moonep_weight_layout(w13_local, w2_local, num_prefetch_slots)
    if w13_local.size(0) * ep_size != num_global_experts:
        raise NotImplementedError(
            "MoonEP PoC requires num_experts to be evenly divisible across EP "
            f"ranks: {num_global_experts} experts, {ep_size} ranks"
        )
    w13_global = ep_group.all_gather(w13_local.contiguous(), dim=0)
    w2_global = ep_group.all_gather(w2_local.contiguous(), dim=0)
    return make_moonep_weight_layout(w13_global, w2_global, num_prefetch_slots)


class MoonEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using MoonEP balanced dispatch/combine.

    ``prepare`` pads the batch to the buffer's static token capacity,
    dispatches, runs ``prefetch_weight`` for the planned redundant experts,
    and stashes the ``plan`` for ``finalize`` (the same pattern DeepEP-HT
    uses for its handle). Downstream expert compute must consume the
    expert-grouped ``[NvS, H]`` layout via ``cu_seqlens``.
    """

    def __init__(
        self,
        buffer: Any,  # moonep.Buffer
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_global_experts: int,
        weight_layout: MoonEPExpertWeightLayout | None = None,
    ):
        super().__init__()
        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        self.num_global_experts = num_global_experts
        self.weight_layout = weight_layout
        self._fused_experts: Any = None

        # dispatch state consumed by finalize (and the expert runner)
        self._plan: Any = None
        self._cu_seqlens: torch.Tensor | None = None
        self._num_tokens: int = 0

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts) -> None:
        # The [E+B] weight layout is attached to the experts by their
        # process_weights_after_loading hook (after this runs), so keep the
        # reference and resolve the layout lazily in prepare().
        self._fused_experts = fused_experts

    def _resolve_weight_layout(self) -> MoonEPExpertWeightLayout:
        if self.weight_layout is None and self._fused_experts is not None:
            layout = getattr(self._fused_experts, "weight_layout", None)
            if isinstance(layout, MoonEPExpertWeightLayout):
                self.weight_layout = layout
        # Redundant experts' weights must be in rows [E, E+B) before the
        # expert compute reads them; skipping prefetch silently corrupts
        # output.
        assert self.weight_layout is not None, (
            "MoonEPPrepareAndFinalize: weight layout not available (the "
            "experts' process_weights_after_loading has not run)"
        )
        return self.weight_layout

    @property
    def num_dispatched_slots(self) -> int:
        """``NvS``: static number of dispatched token slots per rank."""
        return int(self.buffer._ctx["NvS"])

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        # combine returns the fully weighted+reduced token-major output
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    @property
    def cu_seqlens(self) -> torch.Tensor:
        assert self._cu_seqlens is not None, "prepare() has not been called"
        return self._cu_seqlens

    @property
    def plan(self) -> Any:
        assert self._plan is not None, "prepare() has not been called"
        return self._plan

    def _pad_to_capacity(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_tokens = a1.shape[0]
        capacity = self.max_tokens_per_rank
        if num_tokens > capacity:
            raise ValueError(
                f"MoonEP static buffer capacity exceeded: {num_tokens} tokens "
                f"> S={capacity}."
            )
        a1 = a1.contiguous()
        topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
        topk_weights = topk_weights.to(dtype=torch.float32).contiguous()
        if num_tokens == capacity:
            return a1, topk_ids, topk_weights, num_tokens
        pad = capacity - num_tokens
        return (
            F.pad(a1, (0, 0, 0, pad)),
            F.pad(topk_ids, (0, 0, 0, pad)),
            F.pad(topk_weights, (0, 0, 0, pad)),
            num_tokens,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if a1.dtype != torch.bfloat16:
            raise NotImplementedError("MoonEP PoC supports BF16 hidden states only.")
        if quant_config.quant_dtype is not None:
            raise NotImplementedError("MoonEP PoC does not support quantized dispatch.")
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "MoonEP PoC applies router weights in the expert runner."
            )
        assert num_experts == self.num_global_experts
        assert self._plan is None, (
            "MoonEPPrepareAndFinalize.prepare() called again before finalize()"
        )

        a1, topk_ids, topk_weights, self._num_tokens = self._pad_to_capacity(
            a1, topk_ids, topk_weights
        )
        tokens_per_expert = torch.bincount(
            topk_ids.reshape(-1).to(dtype=torch.int64),
            minlength=num_experts,
        ).to(dtype=torch.int32)

        hidden_nvsh, route_weights_nvs, cu_seqlens, plan = self.buffer.dispatch(
            a1,
            topk_weights,
            topk_ids,
            tokens_per_expert,
        )
        self._plan = plan
        self._cu_seqlens = cu_seqlens

        weight_layout = self._resolve_weight_layout()
        self.buffer.prefetch_weight(
            plan=plan,
            full_gate_weight=weight_layout.full_gate_weight,
            full_up_weight=weight_layout.full_up_weight,
            full_down_weight=weight_layout.full_down_weight,
        )

        # Segment sizes per [E+B] weight row. NOTE: this is per *row*, not
        # per local expert, and NvS rows are expert-grouped rather than
        # token-major — only MoonEP-aware expert implementations can consume
        # this activation layout.
        expert_num_tokens = torch.diff(cu_seqlens, prepend=cu_seqlens.new_zeros(1))
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        )

        # MoonEP has no post-dispatch per-token topk ids/weights; route
        # weights come back in NvS order.
        return hidden_nvsh, None, expert_tokens_meta, None, route_weights_nvs

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        # Route weights are applied in the expert compute and MoonEP's
        # combine performs the K-sum, so any weight/reduce work in finalize
        # means MoonEP was paired with the wrong kind of experts.
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP), (
            "MoonEP requires TopKWeightAndReduceNoOP, got "
            f"{type(weight_and_reduce_impl).__name__}"
        )
        combined, _, _ = self.buffer.combine(
            plan=self.plan,
            hidden_nvsh=fused_expert_output,
            route_weights_nvs=None,
        )
        output.copy_(combined[: self._num_tokens])
        self._plan = None
        self._cu_seqlens = None
