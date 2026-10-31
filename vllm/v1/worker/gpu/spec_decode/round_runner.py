# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculative round as the GPU model runner drives it: plan, lay out,
restore, commit and observe one verify step."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.gpu.spec_decode.capabilities import (
    SpecCapabilities,
    SpecLayerKind,
    SpecLimit,
    attn_layer_kind,
    attn_layout,
    layer_spec_limits,
    mamba_state_mode,
    spec_layers,
)
from vllm.v1.worker.gpu.spec_decode.commit import accept_result_of, accepted_rows
from vllm.v1.worker.gpu.spec_decode.kernels import (
    CommitStep,
    KernelContext,
    KernelRegistry,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.plan_kernels import (
    PlanLayout,
    chain_parents,
    chain_path,
)
from vllm.v1.worker.gpu.spec_decode.round import (
    AcceptResult,
    PlanBound,
    PlanInputs,
    SpecResources,
    VerifyPlan,
)
from vllm.v1.worker.gpu.spec_decode.step_observation import StepObservation

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class SpecRound:
    """What the GPU model runner holds for speculative decoding besides the
    speculator: the chain plan, the plan layout when the speculator branches,
    the resources and capabilities it runs with, and the spec kernels. Built
    only with speculative decoding."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        resources: SpecResources,
        max_num_reqs: int,
        max_num_tokens: int,
        num_speculative_steps: int,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_rows = num_speculative_steps + 1
        self.device = device
        # The plan of a step that verifies drafts: today always a chain, with its
        # counts and tokens in the runner's buffers. Other plans are laid out by
        # layout, built by init_tree_verify.
        self.chain_plan = VerifyPlan(
            parent=chain_parents(max_num_reqs, self.max_rows, device),
            count=None,
            is_chain=True,
        )
        # The accepted path of every chain round, path[r, j] == j.
        self.chain_path = chain_path(max_num_reqs, self.max_rows, device)
        self.layout: PlanLayout | None = None
        # What speculation holds (rows, decode graphs, branching), declared by
        # the speculator's config. Memory is sized for it; set_resources lowers
        # it to what the model's layers verify before capture.
        self.resources = resources
        self.capabilities: SpecCapabilities | None = None
        # Above 1 the speculator verifies trees: every verify batch (chains and
        # capture batches too) is laid out, so the attention metadata always
        # carries tree_parents / tree_ancestors and one graph serves every plan.
        self.max_branching = self.resources.max_branching
        # The rows per request of a chain the runner lays out (max_branching > 1).
        self.chain_count: torch.Tensor | None = None
        if self.max_branching > 1:
            self.chain_count = torch.ones(
                max_num_reqs, dtype=torch.int32, device=device
            )
        # The spec verify and commit kernels (registry), bound by set_resources.
        self.kernels: KernelRegistry | None = None
        self.kernel_context: KernelContext | None = None
        # Steps passed to BaseSpeculator.observe_step so far.
        self.num_observed_steps = 0

    def init_tree_verify(self, input_buffers: "InputBuffers") -> None:
        """Buffers of a speculator whose plans branch."""
        if self.max_branching == 1:
            return
        self.layout = PlanLayout(self.max_num_tokens, self.max_rows, self.device)
        input_buffers.tree_parents = self.layout.tree_parents
        input_buffers.tree_ancestors = self.layout.tree_ancestors

    def registry(self) -> KernelRegistry:
        """The speculator's spec kernels, built once per worker."""
        if self.kernels is None:
            self.kernels = KernelRegistry.from_config(self.vllm_config)
        return self.kernels

    def resolve_capabilities(self, runner: "GPUModelRunner") -> SpecCapabilities:
        """What this rank's layers verify, after the attention backends are
        built: the lowest plan kind of any target layer (draft layers do not
        count), lowered to ``"chain"`` by each runner setting that keeps rows in
        chain order."""
        limits = layer_spec_limits(
            runner.attn_groups,
            runner.compilation_config.static_forward_context,
            mamba_state_mode(runner.cache_config),
            kernels=self.registry(),
        )
        settings = {
            "pipeline parallelism (no tree path broadcast)": runner.use_pp,
            "prefill context parallelism": runner.pcp_manager is not None,
            "decode context parallelism": runner.dcp_size > 1,
            "batch-sharded sampling": runner.batch_sharder is not None,
            "micro-batching": runner.parallel_config.use_ubatching,
            f"model state {type(runner.model_state).__name__}": (
                not type(runner.model_state).supports_tree_verify
            ),
        }
        limits += [
            SpecLimit("chain", f"{what}: keeps verify rows in chain order")
            for what, applies in settings.items()
            if applies
        ]
        self.capabilities = SpecCapabilities.from_limits(limits)
        return self.capabilities

    def set_resources(
        self,
        resources: SpecResources,
        input_buffers: "InputBuffers",
        layers: list[SpecLayerSpec],
    ) -> None:
        """Run speculation with ``resources``, the engine core's adaptation of
        the declared ones to what every rank verifies, and bind the spec kernels
        to ``layers``. The memory stays sized for the declared resources."""
        self.resources = resources
        if resources.max_branching == 1 < self.max_branching:
            # Lowered to chains: batches keep the chain layout.
            self.max_branching = 1
            self.chain_count = None
            self.layout = None
            input_buffers.tree_parents = None
            input_buffers.tree_ancestors = None
        self.kernel_context = KernelContext(
            device=self.device,
            plan_kind=resources.plan_kind,
            max_num_reqs=self.max_num_reqs,
            max_candidates=resources.max_candidates,
            capture_shapes=resources.capture_shapes,
        )
        self.registry().bind(self.kernel_context, layers)

    def capture(self) -> None:
        """Capture the bound spec kernels' graphs, if they have any."""
        if self.kernel_context is not None:
            self.registry().capture(self.kernel_context)

    def plan_batch(
        self,
        speculator: "BaseSpeculator",
        scheduler_output: "SchedulerOutput",
        sampler: "Sampler | None",
        req_ids: list[str],
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_draft_tokens: np.ndarray,
        has_prefill: bool,
    ) -> tuple[torch.Tensor, VerifyPlan | None, tuple[np.ndarray, np.ndarray] | None]:
        """Ask the speculator for the batch's plan. Returns the device copy of
        ``idx_mapping_np`` it planned with, the plan (None: verify the scheduled
        drafts as a chain), and the per-request token and draft counts when the
        plan verifies fewer drafts than were scheduled (else None)."""
        idx_mapping = async_tensor_h2d(
            idx_mapping_np, device=self.device, dtype=torch.int64
        )
        chain_only = self._chain_only(
            scheduler_output, sampler, idx_mapping_np, has_prefill
        )
        planned = speculator.plan(
            PlanInputs(
                req_ids=req_ids,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                scheduled_rows=num_draft_tokens + 1,
                chain_only=chain_only,
            )
        )
        verify_plan, bound = None, PlanBound()
        if planned is not None:
            verify_plan, bound = planned
            self._check_plan(verify_plan, len(req_ids), chain_only)
        if bound.rows is None:
            return idx_mapping, verify_plan, None
        # Unverified drafts leave the batch; they count as rejected.
        trimmed = trim_draft_tokens(
            num_scheduled_tokens, num_draft_tokens, bound.rows - 1
        )
        return idx_mapping, verify_plan, trimmed

    def _chain_only(
        self,
        scheduler_output: "SchedulerOutput",
        sampler: "Sampler | None",
        idx_mapping_np: np.ndarray,
        has_prefill: bool,
    ) -> bool:
        """Whether this step can only verify chains: a prefill shares the batch
        (a branching plan must be its request's whole query), structured output
        (the grammar bitmask of row ``i`` assumes the rows before it as its
        prefix), or logits processing that reads each row's prefix."""
        if self.max_branching == 1:
            return True
        if has_prefill or scheduler_output.has_structured_output_requests:
            return True
        assert sampler is not None
        return sampler.reads_row_prefix(idx_mapping_np)

    def _check_plan(self, plan: VerifyPlan, num_reqs: int, chain_only: bool) -> None:
        if plan.is_chain:
            return
        if chain_only:
            raise ValueError(
                "the speculator returned a plan that is not a chain for a batch "
                "that can only verify chains (PlanInputs.chain_only)"
            )
        resources = self.resources
        if plan.mask is not None and not resources.custom_mask:
            raise ValueError(
                "the speculator returned a plan with a mask, but runs with plan "
                f"kind {resources.plan_kind!r} (the model verifies no masks)"
            )
        if plan.parent.shape[1] > resources.max_candidates:
            raise ValueError(
                f"a plan has {plan.parent.shape[1]} rows per request; the "
                f"speculator runs with max_candidates {resources.max_candidates}"
            )
        if plan.parent.shape[0] < num_reqs:
            raise ValueError("a plan must cover every request of the batch")

    def lay_out(self, input_batch: "InputBatch") -> None:
        """Lay out the batch's plan: move its rows' positions to
        ``num_computed_tokens + depth``, write its candidate tokens and fill the
        tree metadata the attention backends read.

        A speculator that never branches verifies chains, which are laid out
        already, so nothing runs for it. With ``max_branching > 1`` every batch is
        laid out (a chain, or a batch without drafts, as the chain of its query
        rows), so the tree metadata is current whenever a captured graph that
        reads it replays: one graph serves every plan."""
        if self.max_branching == 1:
            return
        assert self.layout is not None
        plan = input_batch.verify_plan
        num_reqs = input_batch.num_reqs
        if plan is None or plan.count is None:
            assert self.chain_count is not None
            count = self.chain_count[:num_reqs]
            torch.diff(input_batch.query_start_loc[: num_reqs + 1], out=count)
            count.clamp_(1, self.max_rows)
            plan = self.chain_plan
        else:
            count = plan.count
        self.layout.apply(
            plan,
            count,
            input_batch.query_start_loc,
            input_batch.positions,
            input_batch.input_ids,
            num_reqs,
            input_batch.num_tokens,
            input_batch.num_tokens_after_padding,
        )
        num_tokens = input_batch.num_tokens_after_padding
        input_batch.tree_parents = self.layout.tree_parents[:num_tokens]
        input_batch.tree_ancestors = self.layout.tree_ancestors[:num_tokens]
        input_batch.tree_pos_delta = self.layout.pos_delta[:num_tokens]

    def restore(self, input_batch: "InputBatch") -> None:
        """Undo :meth:`lay_out` after the forward: the sampler, the prompt
        logprobs and the drafter read chain positions (accept gets the plan
        positions as ``VerifyOutput.positions``)."""
        if self.layout is not None:
            self.layout.restore_positions(input_batch.positions, input_batch.num_tokens)

    def accept_result(
        self, sampler_output: "SamplerOutput", input_batch: "InputBatch"
    ) -> AcceptResult | None:
        """The accepted path: the speculator's accept chose it, or the chain
        sampler's arange (a view of a constant buffer). None without drafts."""
        if input_batch.verify_plan is None:
            return None
        return accept_result_of(sampler_output, self.chain_path)

    def commit(
        self, input_batch: "InputBatch", result: AcceptResult | None
    ) -> torch.Tensor | None:
        """Commit the accepted path with the bound commit kernels (move its KV
        to the chain slots, gather its conv inputs into the window the next step
        reads) and return the verify row each request's recurrent state resumes
        from. A chain's accepted rows already sit where the next step reads them
        and its state rows follow from ``num_sampled``, so a chain commits
        nothing here and returns None."""
        plan = input_batch.verify_plan
        if result is None or is_chain(plan):
            return None
        assert plan is not None and plan.count is not None
        assert self.kernels is not None and self.kernel_context is not None, (
            "set_resources binds the commit kernels"
        )
        num_reqs = input_batch.num_reqs
        self.kernels.commit(
            CommitStep(result, num_reqs, plan.count, input_batch.seq_lens)
        )
        return accepted_rows(result, num_reqs)

    def observe(
        self,
        speculator: "BaseSpeculator",
        input_batch: "InputBatch",
        batch_desc: "BatchExecutionDescriptor | None",
        start_ns: int,
        num_spec_tokens: int,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """Pass the step to ``speculator.observe_step``."""
        assert batch_desc is not None
        num_reqs = input_batch.num_reqs
        num_prefill_tokens = 0
        if input_batch.has_prefill:
            prefilling = input_batch.is_prefilling_np[:num_reqs]
            scheduled = input_batch.num_scheduled_tokens[:num_reqs]
            num_prefill_tokens = int(scheduled[prefilling].sum())
        step = self.num_observed_steps
        self.num_observed_steps = step + 1
        speculator.observe_step(
            StepObservation(
                step=step,
                start_ns=start_ns,
                req_ids=input_batch.req_ids,
                num_reqs=num_reqs,
                num_reqs_padded=input_batch.num_reqs_after_padding,
                num_tokens=input_batch.num_tokens,
                num_tokens_padded=input_batch.num_tokens_after_padding,
                cudagraph_mode=batch_desc.cg_mode,
                uniform_token_count=batch_desc.uniform_token_count,
                num_ubatches=batch_desc.num_ubatches,
                num_prefill_tokens=num_prefill_tokens,
                num_spec_tokens=num_spec_tokens,
                num_scheduled_tokens=input_batch.num_scheduled_tokens,
                num_computed_tokens=input_batch.num_computed_tokens_np,
                is_prefilling=input_batch.is_prefilling_np,
                num_draft_tokens=input_batch.num_draft_tokens_per_req,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
            )
        )


def spec_kernel_layers(runner: "GPUModelRunner") -> list[SpecLayerSpec]:
    """The target layers (``spec_layers``) the spec kernels may run on, with
    their KV cache group's block table and block size.

    Block sizes come from the KV cache group, as the cache was allocated and
    the block table is kept: a hybrid model's group blocks are larger than its
    layers' own specs say (attention pages grow to the recurrent state's).
    """
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    forward_context = runner.compilation_config.static_forward_context
    state_mode = mamba_state_mode(runner.cache_config)
    layers = []
    for group, name, spec, layer in spec_layers(runner.attn_groups, forward_context):
        if getattr(layer, "kv_sharing_target_layer_name", None):
            # It reads its target's cache, which the target's commit moves.
            continue
        gid = group.kv_cache_group_id
        placement = dict(
            block_table=runner.block_tables.input_block_tables[gid],
            block_size=spec.block_size,
            num_blocks=runner.kv_cache_config.num_blocks
            * (spec.block_size // runner.kernel_block_sizes[gid]),
        )
        if isinstance(layer, MambaBase):
            variant = type(layer).spec_layer_variant
            if variant is None:
                continue
            kind = SpecLayerKind("mamba", variant, state_mode)
            layers.append(SpecLayerSpec(name, kind, layer, "mamba_state", **placement))
        elif isinstance(spec, AttentionSpec):
            layout = attn_layout(layer, spec)
            layers.append(
                SpecLayerSpec(name, attn_layer_kind(spec), layer, layout, **placement)
            )
    return layers


def trim_draft_tokens(
    num_scheduled_tokens: np.ndarray,
    num_draft_tokens: np.ndarray,
    num_verified_drafts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep each request's first ``num_verified_drafts`` scheduled drafts.

    Returns the per-request token and draft counts of the trimmed batch.
    """
    num_verified_drafts = np.asarray(num_verified_drafts, dtype=num_draft_tokens.dtype)
    in_range = num_verified_drafts.shape == num_draft_tokens.shape and bool(
        ((num_verified_drafts >= 0) & (num_verified_drafts <= num_draft_tokens)).all()
    )
    if not in_range:
        raise ValueError(
            "a plan must verify, for each request, between 1 and its scheduled "
            f"rows; got {num_verified_drafts + 1} for {num_draft_tokens + 1}."
        )
    num_trimmed = num_draft_tokens - num_verified_drafts
    return num_scheduled_tokens - num_trimmed, num_verified_drafts


def is_chain(plan: VerifyPlan | None) -> bool:
    """Whether a batch's plan keeps chain positions (no plan: no drafts)."""
    return plan is None or plan.is_chain
