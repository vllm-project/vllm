# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sublayer boundaries of the specialized decode executor.

A decoder block (attention or MoE) of a tensor-parallel Qwen4Exp step ends with
an all-reduce of the ranks' partial outputs, and the next hyper-connection read
starts with the Triton combine-norm (`hc_combine_norm`): the residual streams
take the block output and are normalized for the read's projections. After a
MoE block, the shared expert's sigmoid gate, its product and the add of the
routed output run first, as three PyTorch kernels. `Boundaries` leaves all of
it to one kernel, `SublayerBoundary` (`..boundary`):

- the block's output projection (`RowParallelLinear`) skips its all-reduce
  for the call (`partial_output`), and the block output stays pending as
  this rank's partial (`PendingBlock`);
- the MoE layer runs its runner's body (`MoERunner._forward_impl`) without
  the final all-reduce, with the shared expert's gate left out of the shared
  expert (`ungated_shared_expert`): the shared expert returns its output
  before the gate, and the gate's logits are computed on the same stream as
  the shared expert, as the stock gate is;
- the next read (`Boundaries.read`) runs the kernel for the reduction, the
  gate and the combine-norm, then its own projections.

Which sites. At TP > 1, every attention and MoE block whose layer passes
`moe_unsupported` (the MoE layer of Qwen3.8-Flash-Next); at TP 1 only the MoE
blocks, since an attention boundary has no reduction to fold and the stock
combine-norm is as fast. A MoE block before a layer with per-layer embeddings
(PLE), which adds to the streams between the combine and the norm, and every MoE
block of a step with deepstack embeddings keep the stock path. Every decision
reads build-time state or host state that all TP ranks share, so the ranks run
the same boundaries in the same order, as the kernel's exchange requires; the
ranks compare their MoE sites when the boundaries are built and all raise if
they differ, rather than hang in the first step.

The kernel's exchange holds its own symmetric-memory mailbox and device-side
generation state, separate from the stock all-reduce's workspace, which the
rest of the step (embedding, LM head, MTP drafter) keeps using.

Row split. Built with `split`, the kernel also has the row-split calls of
the executor's large TP4 steps (`split.py`): `read` with `split` reduces
and combines this rank's quarter of the rows only, and the read's projections
(`project`) all-gather the block input.

Folded finalize. With FlashInfer's TRT-LLM NVFP4 experts, the routed experts
stop after their second GEMM (GEMM2) and the boundary kernel runs their
*finalize*, the weighted sum over each token's top-10 experts
(`SublayerBoundary.moe_folded`), bit for bit as FlashInfer's finalize kernel
would. The experts' `post_gemm2_hook` (`DeferredFinalize`) keeps their
unfinalized output: GEMM2's rows, the routing weights and each token's GEMM2
row per expert (-1 for an expert another rank holds under expert parallelism).
The runner's own deferred finalize (`FusedMoEConfig.use_deferred_moe_finalize`)
excludes expert parallelism and is not used. `deferred_finalize` also hands
FlashInfer an `output` of the finalized shape, which a deferred call never
writes: FlashInfer's autotuner keys its tactics by every input's shape,
`output` included, and a deferred call without one would miss the tactics
tuned for the finalized call, giving GEMM2 another tactic and other bits.

The fold applies to steps of `FOLD_MIN_ROWS` (32) rows or more; smaller steps
run FlashInfer's finalize kernel and the boundary kernel's unfolded MoE variant,
exactly as without the fold. In small steps the shared expert on S ends with
the routed chain or after it, so dropping the finalize kernel shortens nothing,
while the fold adds its two dependent loads after the wait (the routing
weights and slots, then the rows) to the boundary kernel: about 2.2 us against
the finalize kernel's 1.5 us, measured at 4 and 16 rows at TP 4. From 32 rows
the routed chain is the longer branch and the fold gains. A step's row count is
fixed per captured CUDA graph and equal on every rank, so the choice is static
per captured shape and every rank makes the same one.

The order the folded finalize relies on, on the main stream M (the shared
expert and its gate run on the side stream S, which M joins before the
boundary, as without the fold):

1. Who waits on what. FlashInfer's routing kernel writes the weights and
   slots, GEMM1 reads them, GEMM2 writes the rows; the boundary kernel B
   follows GEMM2 directly. Every kernel of M launched with programmatic
   dependent launch (PDL) waits for its predecessor (`griddepcontrol.wait`)
   before it reads that predecessor's outputs or writes anything, and ends
   only after its wait returned; a kernel launched without PDL starts after its
   predecessor completed. So a kernel's wait returns only after every earlier
   kernel of M completed. B reads the rows, weights and slots only after its
   wait, which returns after GEMM2, and so the routing kernel, completed.
2. No reuse before B has read them. The rows, weights and slots are tensors
   that FlashInfer allocates on M; the pending block references them until B
   is enqueued, so no allocation made before that, B's own outputs included,
   can take their memory. An allocation after it that does is made on M (the
   caching allocator reuses a block only on the stream that freed it), so the
   kernel W that writes it is enqueued after B. W writes only after its own
   wait (or, without PDL, starts after its predecessor completed), hence after
   B completed (point 1).
3. FlashInfer's `output` stand-in and the hook's output tensor are never
   written.

`tests/models/qwen4_exp/specialized_decode/test_boundary_fold.py` checks
points 1 and 2 by simulation on what `moe_output` and `read` enqueue and
release, with the boundary kernel's accesses read from its source, and shows
that the simulation reports injected defects (a missing wait, an early
release). During a CUDA graph capture, `DeferredFinalize` and `read` check
the captured graph: when the hook runs, M's only pending node is GEMM2 (no
finalize kernel ran), and the boundary kernel's node depends on that GEMM2 node
(nothing ran on M between them). A violation raises during the capture.

Routed front (TP 1). At TP 1 the target's MoE layers' calls of up to 32 rows
run their routing and input quantization as one kernel (`RoutedPath`,
`moe.py`) and the expert GEMMs on its record, folded or finalized as above,
bitwise the stock output. With the folded finalize, the routing weights and
slots the boundary kernel reads are the layer's record, which only that
layer's front writes, before GEMM1 (`moe.py`, ordering point 4); the hook
and the capture checks are the same.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist
from torch import nn

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPMonolithic,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.model_executor.layers.linear import RowParallelLinear

from .. import HC_HIDDEN
from .gdn import capture_dependencies

if TYPE_CHECKING:
    from ...hyperconnection import GatedResidual
    from ...model import Qwen4ExpDecoderLayer, Qwen4ExpSparseMoeBlock
    from ..boundary import SublayerBoundary
    from .moe import RoutedPath

logger = init_logger(__name__)

# A read's projections from its normalized streams: the block input and the
# injection logits (``hc_project`` or ``stock_project``).
Project = Callable[
    ["GatedResidual", torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]
]


# The routed experts per token whose finalize the boundary kernel folds in.
FOLDED_TOP_K = 10
# The smallest step (token rows) whose MoE boundaries fold the finalize in.
FOLD_MIN_ROWS = 32
# The name prefix of FlashInfer's TRT-LLM NVFP4 GEMM2 kernels (BF16 output).
GEMM2_KERNEL = "bmm_Bfloat16_"


class PendingBlock(NamedTuple):
    """A block output left to the next boundary kernel: this rank's partial
    output, or for a MoE block its routed output, the shared expert's output
    before the gate and the gate's logits. With the folded finalize,
    `partial` holds the routed experts' GEMM2 rows, with the routing weights
    and slots and, during a capture, GEMM2's graph node."""

    partial: torch.Tensor
    shared: torch.Tensor | None = None
    gate_logit: torch.Tensor | None = None
    expert_weights: torch.Tensor | None = None
    expert_slots: torch.Tensor | None = None
    gemm2_node: int | None = None


@contextmanager
def partial_output(projection: RowParallelLinear) -> Iterator[None]:
    """Within the block, `projection` returns this rank's partial output
    rather than all-reducing it."""
    saved = projection.reduce_results
    projection.reduce_results = False
    try:
        yield
    finally:
        projection.reduce_results = saved


class GateLogits(nn.Module):
    """Stands in for a MoE layer's shared expert in the runner: computes the
    gate's logits, then the shared expert's output with its gate left out
    (`ungated_shared_expert`). Runs wherever the runner runs the shared
    expert (its side stream when it overlaps it)."""

    def __init__(self, mlp: "Qwen4ExpSparseMoeBlock") -> None:
        """The stand-in for `mlp`'s shared expert."""
        super().__init__()
        # A tuple, so that the layer's modules are not registered twice.
        self._mlp = (mlp,)
        self.logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The shared expert's output for `x` before the gate; keeps the
        gate's logits in `logits`."""
        (mlp,) = self._mlp
        self.logits, _ = mlp.shared_expert_gate(x)
        shared_expert = mlp.shared_expert
        assert shared_expert is not None
        return shared_expert(x)


@contextmanager
def ungated_shared_expert(
    mlp: "Qwen4ExpSparseMoeBlock", logits: GateLogits
) -> Iterator[None]:
    """Within the block, `mlp`'s runner runs `logits` for the shared
    expert, which returns its output before the gate."""
    shared_expert = mlp.shared_expert
    wrapper = mlp.experts.shared_experts
    assert shared_expert is not None and wrapper is not None
    gate = shared_expert.expert_gate
    layer = wrapper._layer
    shared_expert.expert_gate = None
    wrapper._layer = logits
    try:
        yield
    finally:
        shared_expert.expert_gate = gate
        wrapper._layer = layer


def moe_unsupported(mlp: "Qwen4ExpSparseMoeBlock") -> str:
    """What leaving `mlp`'s output to the boundary kernel needs that it
    misses, or `""`."""
    runner = mlp.experts
    config = runner.moe_config
    shared_expert = mlp.shared_expert
    requirements = {
        "a TP-sharded shared expert with a sigmoid gate, run by the experts' runner": (
            shared_expert is not None
            and not mlp.replicate_shared_expert
            and shared_expert.expert_gate is mlp.shared_expert_gate
            and runner.shared_experts is not None
            and runner.shared_experts._layer is shared_expert
        ),
        "routed experts whose output the runner adds to the shared output "
        "before one all-reduce": (
            runner.routed_scaling_factor == 1.0
            and runner.routed_input_transform is None
            and runner.routed_output_transform is None
            and not isinstance(runner.router, ZeroExpertRouter)
            and not runner._fused_output_is_reduced
            and not runner.do_naive_dispatch_combine
            and not config.is_sequence_parallel
            and config.pcp_size == 1
            and config.hidden_dim == HC_HIDDEN
        ),
    }
    return "; ".join(need for need, ok in requirements.items() if not ok)


def fold_unsupported(mlp: "Qwen4ExpSparseMoeBlock") -> str:
    """What folding `mlp`'s routed finalize into the boundary kernel needs
    that it misses, or `""` (beyond `moe_unsupported`)."""
    quant_method = mlp.experts.routed_experts.quant_method
    impl = getattr(getattr(quant_method, "moe_kernel", None), "impl", None)
    experts = getattr(impl, "fused_experts", None)
    requirements = {
        "FlashInfer's TRT-LLM NVFP4 experts with routing and quantized inputs": (
            isinstance(experts, TrtLlmNvFp4ExpertsMonolithic)
            and not experts.expects_unquantized_inputs
            and experts.topk == FOLDED_TOP_K
            and experts.post_gemm2_hook is None
        ),
        "a prepare step that passes the experts' output through": isinstance(
            getattr(impl, "prepare_finalize", None),
            MoEPrepareAndFinalizeNoDPEPMonolithic,
        ),
    }
    return "; ".join(need for need, ok in requirements.items() if not ok)


def folding_experts(mlp: "Qwen4ExpSparseMoeBlock") -> TrtLlmNvFp4ExpertsMonolithic:
    """The routed experts of `mlp`, which `fold_unsupported` accepted."""
    moe_kernel = mlp.experts.routed_experts.quant_method.moe_kernel
    assert moe_kernel is not None
    experts = moe_kernel.impl.fused_experts
    assert isinstance(experts, TrtLlmNvFp4ExpertsMonolithic)
    return experts


def kernel_name(node: int) -> str:
    """The function name of a captured kernel node, `""` for another node."""
    # Imported here: CUDA builds only.
    from cuda.bindings import driver

    handle = driver.CUgraphNode(node)
    error, kind = driver.cuGraphNodeGetType(handle)
    _check_driver(error)
    if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
        return ""
    error, params = driver.cuGraphKernelNodeGetParams(handle)
    _check_driver(error)
    if int(params.func):
        error, name = driver.cuFuncGetName(params.func)
    else:
        error, name = driver.cuKernelGetName(params.kern)
    _check_driver(error)
    return name.decode() if isinstance(name, bytes) else str(name)


def node_dependencies(node: int) -> frozenset[int]:
    """The nodes a captured graph node depends on."""
    from cuda.bindings import driver

    # (error, dependencies, ...): CUDA 13 also returns the edge data. Entries
    # past the node's dependencies are null.
    result = driver.cuGraphNodeGetDependencies(driver.CUgraphNode(node), 64)
    _check_driver(result[0])
    return frozenset(int(other) for other in result[1] if int(other))


def _check_driver(error: object) -> None:
    """Raise on a failed CUDA driver call."""
    from cuda.bindings import driver

    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {error}")


def require_follows(gemm2: int) -> None:
    """Raise unless the node just captured on the current stream (the boundary
    kernel) is a kernel node that depends on `gemm2`."""
    nodes = capture_dependencies(torch.cuda.current_stream())
    if len(nodes) != 1 or not kernel_name(min(nodes)):
        follows = False
    else:
        follows = gemm2 in node_dependencies(min(nodes))
    if not follows:
        raise RuntimeError(
            "Qwen4Exp specialized decode: the boundary kernel with the folded "
            "finalize was not captured right after the routed experts' GEMM2."
        )


class DeferredFinalize:
    """The routed experts' `post_gemm2_hook` for the folded finalize: keeps
    their unfinalized output for the boundary kernel and leaves the hook's
    output tensor unwritten. During a capture, checks that the experts' call
    ended with GEMM2 (M's only pending node), and keeps that node."""

    def __init__(self) -> None:
        """An empty hook, to be called once."""
        self.output: UnfinalizedMoEOutput | None = None
        self.gemm2_node: int | None = None

    def __call__(self, unfinalized: UnfinalizedMoEOutput, output: torch.Tensor) -> None:
        """Keep `unfinalized`; `output` stays unwritten."""
        self.keep(unfinalized)

    def keep(self, unfinalized: UnfinalizedMoEOutput) -> None:
        """Keep `unfinalized`, right after the GEMM2 that wrote its rows was
        enqueued on the current stream (the check above)."""
        assert self.output is None
        if torch.cuda.is_current_stream_capturing():
            nodes = capture_dependencies(torch.cuda.current_stream())
            names = [kernel_name(node) for node in nodes]
            if len(nodes) != 1 or not names[0].startswith(GEMM2_KERNEL):
                raise RuntimeError(
                    "Qwen4Exp specialized decode: the routed experts' deferred "
                    f"call did not end with GEMM2 (pending: {names})."
                )
            (self.gemm2_node,) = nodes
        self.output = unfinalized


@contextmanager
def deferred_finalize(
    experts: TrtLlmNvFp4ExpertsMonolithic, hook: DeferredFinalize
) -> Iterator[None]:
    """Within the block, `experts` stop after GEMM2 and hand their output to
    `hook`; their FlashInfer call gets an `output` of the finalized shape
    (module docstring), which it does not write."""
    # Imported here: CUDA builds only.
    import flashinfer.fused_moe as fused_moe

    run = fused_moe.trtllm_fp4_block_scale_moe

    def run_deferred(**kwargs: object) -> list[torch.Tensor]:
        """FlashInfer's call with the finalized-shape `output`."""
        hidden_states = kwargs["hidden_states"]
        assert isinstance(hidden_states, torch.Tensor)
        assert kwargs["do_finalize"] is False and kwargs.get("output") is None
        kwargs["output"] = torch.empty(
            (hidden_states.shape[0], HC_HIDDEN),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        return run(**kwargs)

    assert experts.post_gemm2_hook is None
    experts.post_gemm2_hook = hook
    fused_moe.trtllm_fp4_block_scale_moe = run_deferred
    try:
        yield
    finally:
        fused_moe.trtllm_fp4_block_scale_moe = run
        experts.post_gemm2_hook = None


def agree_across_ranks(sites: Sequence[bool], group: dist.ProcessGroup) -> None:
    """Raise on every rank unless all ranks of `group` (a CPU group) pass
    the same `sites`."""
    flags = torch.tensor(sites, dtype=torch.int32)
    low, high = flags.clone(), flags.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(low, high):
        differing = torch.nonzero(low != high).flatten().tolist()
        raise RuntimeError(
            "Qwen4Exp specialized decode: the tensor parallel ranks disagree on "
            f"which MoE layers the boundary kernel serves (positions {differing})."
        )


class Boundaries:
    """The boundary kernel of the executor's steps and the sites it serves.

    Args:
        kernel: The boundary kernel (`SublayerBoundary`), over the TP group.
        attention: Whether the attention blocks' outputs go to the kernel.
        moe_layers: Per position among the executor's layers, its MoE layer
            when that layer's output goes to the kernel, else None.
        fold: Per position, whether the kernel also runs that MoE layer's
            routed finalize (`fold_unsupported`) in steps of
            `FOLD_MIN_ROWS` rows or more; None for no layer.
        routed: The routed front of those MoE layers (TP 1), or None.

    """

    def __init__(
        self,
        kernel: "SublayerBoundary",
        attention: bool,
        moe_layers: Sequence["Qwen4ExpSparseMoeBlock | None"],
        fold: Sequence[bool] | None = None,
        routed: "RoutedPath | None" = None,
    ) -> None:
        self.kernel = kernel
        if fold is None:
            fold = [False] * len(moe_layers)
        self.attention = attention
        self.routed = routed
        # Per layer position: the MoE layer, its gate stand-in and, with the
        # folded finalize, its routed experts; or None.
        self.moe_layers: list[
            tuple[
                Qwen4ExpSparseMoeBlock,
                GateLogits,
                TrtLlmNvFp4ExpertsMonolithic | None,
            ]
            | None
        ] = [
            None
            if mlp is None
            else (mlp, GateLogits(mlp), folding_experts(mlp) if folded else None)
            for mlp, folded in zip(moe_layers, fold, strict=True)
        ]

    @classmethod
    def build(
        cls,
        layers: Sequence["Qwen4ExpDecoderLayer"],
        max_rows: int,
        owner: str = "the target model",
        split: bool = False,
        routed_front: bool = True,
    ) -> "Boundaries":
        """The boundaries of `layers` for steps of up to `max_rows` rows,
        with their own kernel and mailbox. Collective over the TP group (the
        kernel maps every rank's mailbox).

        Args:
            layers: The decoder layers, in the order of their calls.
            max_rows: The token rows of the largest call.
            owner: Whose layers they are, for the log.
            split: Also build the kernel's row-split calls (TP > 1 only;
                module docstring, Row split).
            routed_front: At TP 1, run the served MoE layers' small calls on
                the routed front (`RoutedPath`), which reads the experts'
                processed weights: only for layers whose quantization methods
                have processed them.

        """
        # Imported here: CuTe DSL kernels (CUDA builds only).
        from ..boundary import SublayerBoundary

        tp_group = get_tp_group()
        tp_size = tp_group.world_size
        device = layers[0].mlp_hyper_connection.hc_norm.weight.device
        kernel = SublayerBoundary(
            tp_group.device_group if tp_size > 1 else None,
            max_rows,
            device,
            split=split and tp_size > 1,
        )
        moe_layers: list[Qwen4ExpSparseMoeBlock | None] = []
        fold: list[bool] = []
        missing: dict[str, None] = {}
        unfolded: dict[str, None] = {}
        for position, layer in enumerate(layers):
            mlp = layer.mlp
            reason = moe_unsupported(mlp)
            if reason:
                missing[reason] = None
            # PLE adds to the streams between the combine and the next norm.
            precedes_ple = (
                position + 1 < len(layers) and layers[position + 1].ple is not None
            )
            served = not (reason or precedes_ple)
            moe_layers.append(mlp if served else None)
            fold_reason = fold_unsupported(mlp) if served else ""
            if fold_reason:
                unfolded[fold_reason] = None
            fold.append(served and not fold_reason)
        if tp_size > 1:
            agree_across_ranks(
                [mlp is not None for mlp in moe_layers], tp_group.cpu_group
            )
        if missing:
            logger.info_once(
                "Qwen4Exp specialized decode: some MoE layers of %s keep the "
                "stock all-reduce (the boundary kernel needs %s).",
                owner,
                "; ".join(missing),
            )
        if unfolded:
            logger.info_once(
                "Qwen4Exp specialized decode: some MoE layers of %s finalize "
                "their routed output themselves (folding it needs %s).",
                owner,
                "; ".join(unfolded),
            )
        routed = None
        if routed_front and tp_size == 1:
            from .moe import RoutedPath

            routed = RoutedPath.build(moe_layers, owner)
        boundaries = cls(kernel, tp_size > 1, moe_layers, fold, routed)
        logger.info_once(
            "Qwen4Exp specialized decode: the boundary kernel of %s serves %s "
            "and %d of %d MoE blocks, %d with the routed finalize folded in "
            "(steps of %d rows or more).",
            owner,
            "the attention blocks" if boundaries.attention else "no attention block",
            sum(entry is not None for entry in boundaries.moe_layers),
            len(layers),
            sum(fold),
            FOLD_MIN_ROWS,
        )
        return boundaries

    def warm_up(self, hc: "GatedResidual") -> None:
        """Compile and run each kernel variant the steps use once, on one row
        of zeros with `hc`'s norm; collective over the TP group."""
        entries = [entry for entry in self.moe_layers if entry is not None]
        # Steps below FOLD_MIN_ROWS run the unfolded MoE variant on every
        # served layer.
        self.kernel.warm_up(
            hc.hc_norm.weight,
            hc.config.rms_norm_eps,
            self.attention,
            bool(entries),
            self.kernel.max_rows >= FOLD_MIN_ROWS
            and any(experts is not None for _, _, experts in entries),
        )
        if self.routed is not None:
            self.routed.warm_up()

    @contextmanager
    def attention_partial(self, projection: RowParallelLinear) -> Iterator[bool]:
        """Within the block, the attention block whose output projection is
        `projection` returns this rank's partial output when the kernel
        reduces it; yields whether it does."""
        if not self.attention:
            yield False
            return
        with partial_output(projection):
            yield True

    def moe_output(
        self, position: int, x: torch.Tensor, fused: bool
    ) -> PendingBlock | None:
        """The pending output of the MoE layer at `position` on its input
        `x`, or None when that layer (or the step, `fused` False) runs the
        stock MoE. The routed finalize is folded in when the layer allows it
        and the step has `FOLD_MIN_ROWS` rows or more. Calls the routed
        front takes (`RoutedPath.tactic`) run on it."""
        entry = self.moe_layers[position]
        if entry is None or not fused:
            return None
        mlp, logits, experts = entry
        fold = experts is not None and x.shape[0] >= FOLD_MIN_ROWS
        if self.routed is not None:
            tactic = self.routed.tactic(position, x.shape[0])
            if tactic is not None:
                return self._routed_output(position, mlp, logits, x, tactic, fold)
        if not fold:
            # The runner's custom op returns tensors only: run its body directly.
            with ungated_shared_expert(mlp, logits):
                shared, routed = _unpack(mlp.experts._forward_impl(x, x, x, None))
            gate_logit = logits.logits
            logits.logits = None
            assert isinstance(routed, torch.Tensor) and shared is not None
            assert gate_logit is not None
            return PendingBlock(routed, shared, gate_logit)
        assert experts is not None
        hook = DeferredFinalize()
        with ungated_shared_expert(mlp, logits), deferred_finalize(experts, hook):
            # The routed output is the hook's unwritten output tensor.
            shared, _ = _unpack(mlp.experts._forward_impl(x, x, x, None))
        gate_logit = logits.logits
        logits.logits = None
        unfinalized = hook.output
        assert unfinalized is not None and not unfinalized.route_major
        assert shared is not None and gate_logit is not None
        return PendingBlock(
            unfinalized.gemm2_permuted,
            shared,
            gate_logit,
            unfinalized.expert_weights,
            unfinalized.expanded_idx_to_permuted_idx,
            hook.gemm2_node,
        )

    def _routed_output(
        self,
        position: int,
        mlp: "Qwen4ExpSparseMoeBlock",
        logits: GateLogits,
        x: torch.Tensor,
        tactic: list[int],
        fold: bool,
    ) -> PendingBlock:
        """`moe_output` on the routed front (`RoutedPath.forward`) with
        `tactic`; with `fold`, the experts stop after GEMM2 and the same
        hook as the stock folded call keeps their output and checks the
        capture."""
        assert self.routed is not None
        hook = DeferredFinalize() if fold else None
        with ungated_shared_expert(mlp, logits):
            shared, routed = self.routed.forward(
                position, mlp, x, tactic, None if hook is None else hook.keep
            )
        gate_logit = logits.logits
        logits.logits = None
        assert shared is not None and gate_logit is not None
        if hook is None:
            assert isinstance(routed, torch.Tensor)
            return PendingBlock(routed, shared, gate_logit)
        unfinalized = hook.output
        assert routed is None and unfinalized is not None
        return PendingBlock(
            unfinalized.gemm2_permuted,
            shared,
            gate_logit,
            unfinalized.expert_weights,
            unfinalized.expanded_idx_to_permuted_idx,
            hook.gemm2_node,
        )

    def read(
        self,
        hc: "GatedResidual",
        hidden_states: torch.Tensor,
        pending: PendingBlock,
        injection: torch.Tensor | None,
        project: Project,
        split: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """`hc.combine_and_mix` of the reduced `pending` block: the kernel
        for the reduction and the combine-norm, then `project`. Returns the
        streams, the block input and the read's injection logits. With
        `split` (module docstring, Row split), `hidden_states` and
        `injection` hold this rank's rows, the streams and injection logits
        returned too, and `project` all-gathers the block input."""
        assert injection is not None
        weight = hc.hc_norm.weight
        eps = hc.config.rms_norm_eps
        if pending.shared is None:
            hidden_states, xn = self.kernel(
                pending.partial, hidden_states, injection, weight, eps, split
            )
        elif pending.expert_weights is None:
            assert pending.gate_logit is not None
            hidden_states, xn = self.kernel.moe(
                pending.partial,
                pending.shared,
                pending.gate_logit,
                hidden_states,
                injection,
                weight,
                eps,
                split,
            )
        else:
            assert pending.gate_logit is not None
            assert pending.expert_slots is not None
            hidden_states, xn = self.kernel.moe_folded(
                pending.partial,
                pending.expert_weights,
                pending.expert_slots,
                pending.shared,
                pending.gate_logit,
                hidden_states,
                injection,
                weight,
                eps,
                split,
            )
            if pending.gemm2_node is not None:
                require_follows(pending.gemm2_node)
        block_input, injection = project(hc, xn)
        return hidden_states, block_input, injection
