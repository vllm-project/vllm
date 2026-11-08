# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculative round as the GPU model runner drives it: plan, lay out,
restore and commit one verify step."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.spec_decode.resources import SpecResources
from vllm.v1.worker.gpu.spec_decode.capabilities import (
    SpecCapabilities,
    SpecLayerKind,
    SpecLimit,
    attn_layer_kind,
    attn_layout,
    layer_spec_limits,
    spec_layers,
)
from vllm.v1.worker.gpu.spec_decode.commit import accept_result_of, accepted_rows
from vllm.v1.worker.gpu.spec_decode.kernels import (
    CommitStep,
    KernelContext,
    KernelRegistry,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.plan_kernels import PlanLayout, arange_table
from vllm.v1.worker.gpu.spec_decode.round import AcceptResult, PlanInputs, VerifyPlan

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class SpecRound:
    """The runner's speculative round state besides the speculator."""

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
        # The plan of a step whose speculator returns None: the scheduled chain.
        self.chain_plan = VerifyPlan(
            parent=arange_table(max_num_reqs, self.max_rows, device, start=-1),
            is_chain=True,
        )
        # The accepted path of a chain round, path[r, j] == j.
        self.chain_path = arange_table(max_num_reqs, self.max_rows, device)
        # Whether this rank's speculator overrides ``plan``; resolved at startup.
        self.plans = False
        self.layout: PlanLayout | None = None
        # Declared; set_resources lowers it before capture.
        self.resources = resources
        self.capabilities: SpecCapabilities | None = None
        # With max_branching > 1: the rows each request verifies.
        self.rows: torch.Tensor | None = None
        if resources.max_branching > 1:
            self.rows = torch.ones(max_num_reqs, dtype=torch.int32, device=device)
        self.kernels: KernelRegistry | None = None

    def registry(self) -> KernelRegistry:
        """The speculator's spec kernels, built once per worker."""
        if self.kernels is None:
            self.kernels = KernelRegistry.from_config(self.vllm_config)
        return self.kernels

    def resolve_capabilities(self, runner: "GPUModelRunner") -> SpecCapabilities:
        """What this rank's target layers and runner settings verify."""
        speculator = runner.speculator
        self.plans = speculator is not None and "plan" in speculator.overridden_hooks()
        if runner.adaptive_verification is not None and self.plans:
            raise ValueError(
                "Adaptive verification sets each request's verified drafts, so it "
                "cannot run a speculator that overrides plan. Pass "
                "enable_adaptive_verification=false in the speculative config."
            )
        limits = layer_spec_limits(
            runner.attn_groups,
            runner.compilation_config.static_forward_context,
            kernels=self.registry(),
        )
        why = runner.model_state.tree_decline_reason()
        if why is not None:
            limits.append(SpecLimit("chain", f"model state: {why}"))
        settings = {
            "a speculator without tree_accept": runner.speculator is not None
            and self.resources.max_branching > 1
            and "tree_accept" not in runner.speculator.overridden_hooks(),
            "pipeline parallelism (no tree path broadcast)": runner.use_pp,
            "prefill context parallelism": runner.pcp_manager is not None,
            "decode context parallelism": runner.dcp_size > 1,
            "batch-sharded sampling": runner.batch_sharder is not None,
            "micro-batching": runner.parallel_config.use_ubatching,
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
        """Run with the engine core's ``resources`` and bind the spec kernels
        to ``layers``."""
        self.resources = resources
        if resources.max_branching == 1:
            # Chains (perhaps lowered to): batches keep the chain layout.
            self.rows = None
        layout = None
        if resources.max_branching > 1:
            layout = PlanLayout(self.max_num_tokens, self.max_rows, self.device)
        ctx = KernelContext(
            device=self.device,
            plan_kind=resources.plan_kind,
            max_num_reqs=self.max_num_reqs,
            max_rows=resources.max_rows,
            capture_rows=resources.capture_rows,
            tree_parents=None if layout is None else layout.tree_parents,
        )
        self.registry().bind(ctx, layers)
        # Laid out after binding: until then (the memory profile) dummy batches
        # verify chains, which the layers' own kernels run.
        if layout is not None:
            self.layout = layout
            input_buffers.tree_parents = layout.tree_parents

    def plan_batch(
        self,
        speculator: "BaseSpeculator | None",
        scheduler_output: "SchedulerOutput",
        sampler: "Sampler | None",
        req_ids: list[str],
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        num_draft_tokens: np.ndarray,
        has_prefill: bool,
    ) -> tuple[torch.Tensor | None, VerifyPlan, tuple[np.ndarray, np.ndarray] | None]:
        """Ask the speculator for the batch's plan; a speculator that does not
        plan verifies the scheduled chain, without a step of its own. Returns the
        device ``idx_mapping`` (None when not copied yet), the plan, and the
        trimmed token and draft counts when the plan verifies fewer rows."""
        if not self.plans:
            return None, self.chain_plan, None
        assert speculator is not None
        idx_mapping = async_tensor_h2d(
            idx_mapping_np, device=self.device, dtype=torch.int64
        )
        chain_only = self._chain_only(
            scheduler_output, sampler, idx_mapping_np, has_prefill
        )
        verify_plan = speculator.plan(
            PlanInputs(
                req_ids=req_ids,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                scheduled_rows=num_draft_tokens + 1,
                chain_only=chain_only,
            )
        )
        if verify_plan is None:
            return idx_mapping, self.chain_plan, None
        rows = verify_plan.rows
        self._check_plan(verify_plan, num_draft_tokens + 1 if rows is None else rows)
        assert verify_plan.is_chain or not chain_only, (
            "the speculator returned a plan that is not a chain for a batch "
            "that can only verify chains (PlanInputs.chain_only)"
        )
        if rows is None:
            return idx_mapping, verify_plan, None
        # Unverified drafts leave the batch; they count as rejected.
        trimmed = trim_draft_tokens(num_scheduled_tokens, num_draft_tokens, rows - 1)
        return idx_mapping, verify_plan, trimmed

    def _chain_only(
        self,
        scheduler_output: "SchedulerOutput",
        sampler: "Sampler | None",
        idx_mapping_np: np.ndarray,
        has_prefill: bool,
    ) -> bool:
        """``PlanInputs.chain_only`` for this step."""
        if self.resources.max_branching == 1:
            return True
        if has_prefill or scheduler_output.has_structured_output_requests:
            return True
        assert sampler is not None
        trace = sampler.trace_replay_state
        if trace is not None and trace.has_trace(idx_mapping_np):
            # Only the chain rejection sampler applies the trace.
            return True
        return sampler.reads_row_prefix(idx_mapping_np)

    def _check_plan(self, plan: VerifyPlan, rows: np.ndarray) -> None:
        num_reqs, width = plan.parent.shape
        assert width <= self.max_rows, (
            f"a plan has {width} rows per request, above "
            f"num_speculative_tokens + 1 = {self.max_rows}"
        )
        assert num_reqs >= len(rows), "a plan must cover every request of the batch"
        assert not len(rows) or width >= rows.max(), (
            f"a plan has {width} rows per request but verifies up to {rows.max()}"
        )

    def lay_out(self, input_batch: "InputBatch") -> None:
        """Lay out the batch's plan: positions at depth, candidate tokens and
        tree metadata. With ``max_branching > 1`` every batch is laid out, so
        one captured graph serves every plan."""
        if self.layout is None:
            return
        num_reqs = input_batch.num_reqs
        # A prefill's rows are clamped and laid out as a chain.
        assert self.rows is not None
        rows = self.rows[:num_reqs]
        torch.diff(input_batch.query_start_loc[: num_reqs + 1], out=rows)
        rows.clamp_(1, self.max_rows)
        self.layout.apply(
            input_batch.verify_plan or self.chain_plan,
            rows,
            input_batch.query_start_loc,
            input_batch.positions,
            input_batch.input_ids,
            num_reqs,
            input_batch.num_tokens,
            input_batch.num_tokens_after_padding,
        )
        num_tokens = input_batch.num_tokens_after_padding
        input_batch.tree_parents = self.layout.tree_parents[:num_tokens]
        input_batch.tree_pos_delta = self.layout.pos_delta[:num_tokens]

    def restore(self, input_batch: "InputBatch") -> None:
        """Undo :meth:`lay_out` after the forward: the sampler, the prompt
        logprobs and the drafter read chain positions."""
        if self.layout is not None:
            self.layout.restore_positions(input_batch.positions, input_batch.num_tokens)

    def accept_result(
        self, sampler_output: "SamplerOutput", input_batch: "InputBatch"
    ) -> AcceptResult | None:
        """The accepted path; None without drafts. A chain's is a view of
        ``chain_path``."""
        if input_batch.verify_plan is None:
            return None
        return accept_result_of(sampler_output, self.chain_path)

    def commit(
        self, input_batch: "InputBatch", result: AcceptResult | None
    ) -> torch.Tensor | None:
        """Run the bound commit kernels on the accepted path and return the
        verify row each request's recurrent state resumes from; None for a
        chain, which commits nothing."""
        plan = input_batch.verify_plan
        if result is None or plan is None or plan.is_chain:
            return None
        assert self.rows is not None
        assert self.kernels is not None, "set_resources binds the commit kernels"
        num_reqs = input_batch.num_reqs
        self.kernels.commit(
            CommitStep(result, num_reqs, self.rows[:num_reqs], input_batch.seq_lens)
        )
        return accepted_rows(result, num_reqs)


def spec_kernel_layers(runner: "GPUModelRunner") -> list[SpecLayerSpec]:
    """The target layers the spec kernels may run on, with their KV cache
    group's block table and block size (a hybrid group's blocks can be larger
    than a layer's own spec says)."""
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    forward_context = runner.compilation_config.static_forward_context
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
            if layer.tree_decline_reason() is not None:
                continue
            kind = SpecLayerKind("recurrent", type(layer).__name__)
            layers.append(
                SpecLayerSpec(name, kind, layer, "recurrent_state", **placement)
            )
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
    """The token and draft counts that keep each request's first
    ``num_verified_drafts`` drafts."""
    num_verified_drafts = np.asarray(num_verified_drafts, dtype=num_draft_tokens.dtype)
    in_range = num_verified_drafts.shape == num_draft_tokens.shape and bool(
        ((num_verified_drafts >= 0) & (num_verified_drafts <= num_draft_tokens)).all()
    )
    assert in_range, (
        "a plan must verify, for each request, between 1 and its scheduled "
        f"rows; got {num_verified_drafts + 1} for {num_draft_tokens + 1}."
    )
    num_trimmed = num_draft_tokens - num_verified_drafts
    return num_scheduled_tokens - num_trimmed, num_verified_drafts
