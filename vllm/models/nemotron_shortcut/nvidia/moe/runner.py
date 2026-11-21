# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A MoE runner whose routing and expert compute are separate graph nodes.

A Nemotron Shortcut shortcut block routes on its input, then runs its mixer, and
only then needs the routed contribution. `begin_staged`, `run_staged_experts`
and `finish_staged` are three opaque custom ops linked by dependency tensors,
so torch.compile keeps routing before the mixer and the experts after it.

With data parallelism on the all-gather/reduce-scatter transport, the stages
also take that transport off the compute stream. A side stream runs the
routing and the all-gather while the mixer runs, and the reduce-scatter while
the caller's shared experts run; the experts stage and the finish wait for
it. The collectives, kernels and results are those of the synchronous path.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple, TypeVar

import torch

from vllm.forward_context import in_piecewise_cudagraph
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _resolve_layer_name,
    get_layer_from_name,
)
from vllm.models.nemotron_shortcut.nvidia.moe.experts import (
    DispatchedTokens,
    combine_tokens,
    dispatch_tokens,
    run_experts,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    current_stream,
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from typing import TypeAlias

    _LayerNameType: TypeAlias = str | LayerName
else:
    _LayerNameType = LayerName if _USE_LAYERNAME else str

logger = init_logger(__name__)

_T = TypeVar("_T")


@dataclass
class _Routed:
    """Routed experts' input, kept from `begin_staged` to `run_staged_experts`."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    pre_transform_trunc_size: int | None
    post_transform_trunc_size: int | None
    # Set when the side stream dispatched the tokens.
    tokens: DispatchedTokens | None = None
    # Whether the compute stream has yet to wait for that dispatch.
    dispatch_pending: bool = False


@dataclass
class _Combined:
    """Routed experts' output, kept from `run_staged_experts` to the finish."""

    output: torch.Tensor
    pre_transform_trunc_size: int | None
    post_transform_trunc_size: int | None
    # Set while the side stream reduce-scatters it into `output`: it must live
    # until the compute stream waits for that.
    fused_output: torch.Tensor | None = None


class StagedDispatch(NamedTuple):
    """`begin_staged`'s result, passed on to the later stages.

    The side stream may still read the routing inputs after `begin_staged`
    returns. Passed to the experts op, which waits for it, they keep inductor
    from reusing their memory before then.
    """

    dependency: torch.Tensor
    hidden_states: torch.Tensor
    router_logits: torch.Tensor


# One per device, shared by all layers, so their collectives keep their order.
_side_streams: dict[torch.device, torch.cuda.Stream] = {}


def _launch_on_side_stream(
    device: torch.device, done: torch.cuda.Event, launch: Callable[[], _T]
) -> _T:
    """Run `launch` on the side stream and record `done` after it.

    The side stream first waits for the compute stream's work so far (marked
    with `done` too), so it never overtakes a compute-stream read of memory
    freed to it. Tensors both streams use stay referenced until the compute
    stream waits for `done`, which keeps this safe under CUDA graph capture
    (no `record_stream`).
    """
    side = _side_streams.get(device)
    if side is None:
        side = _side_streams[device] = torch.cuda.Stream(device)
    done.record(current_stream())
    side.wait_event(done)
    with torch.cuda.stream(side):
        result = launch()
    done.record(side)
    return result


def _capturing_graph_piece() -> bool:
    """Whether a piecewise CUDA graph piece is being captured.

    A piece must join every stream it forks before it ends. A full CUDA graph
    is one capture of the whole forward pass, in runtime mode NONE or FULL.
    """
    return torch.cuda.is_current_stream_capturing() and in_piecewise_cudagraph()


class NemotronShortcutMoERunner(MoERunner):
    """`MoERunner` with the forward split into three caller-placed stages.

    Shared experts are not part of the runner: the caller runs them on its
    own input and passes their (unreduced) output to `finish_staged`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.shared_experts is not None:
            raise ValueError("pass the shared experts output to finish_staged")
        if self.moe_config.is_sequence_parallel:
            raise NotImplementedError("staged MoE does not support sequence parallel")
        slots = 2 if self.enable_dbo else 1
        self._routed: list[_Routed | None] = [None] * slots
        self._combined: list[_Combined | None] = [None] * slots

    def _op_layer_name(self) -> _LayerNameType:
        # Always the real name: the shared "from_forward_context" counter
        # counts one call per MoE layer.
        return LayerName(self.layer_name) if _USE_LAYERNAME else self.layer_name

    @property
    def _slot(self) -> int:
        return dbo_current_ubatch_id() if self.enable_dbo else 0

    # The side stream's events, reused across forwards: the compute stream
    # waits for each before the next forward records it again.
    @cached_property
    def _dispatch_done(self) -> torch.cuda.Event:
        return torch.cuda.Event()

    @cached_property
    def _combine_done(self) -> torch.cuda.Event:
        return torch.cuda.Event()

    @property
    def overlaps_transport(self) -> bool:
        """Whether the dispatch and the combine run on the side stream.

        Only the all-gather/reduce-scatter transport, the one these experts
        use with DP. DBO overlaps it with the other microbatch instead, and
        PCP adds collectives outside the kernel.
        """
        kernel = self._quant_method.moe_kernel
        return (
            current_platform.is_cuda()
            and not self.enable_dbo
            and self.moe_config.pcp_size == 1
            and kernel is not None
            and isinstance(
                kernel.prepare_finalize, MoEPrepareAndFinalizeNaiveDPEPModular
            )
        )

    def begin_staged(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> StagedDispatch:
        """Route and dispatch `hidden_states`; returns the later stages' input."""
        dependency = torch.ops.vllm.nemotron_shortcut_moe_begin(
            hidden_states, router_logits, self._op_layer_name()
        )
        return StagedDispatch(dependency, hidden_states, router_logits)

    def receive_staged(
        self, dispatch: StagedDispatch, projections: Sequence[torch.Tensor]
    ) -> StagedDispatch:
        """Wait for the dispatch if a piecewise CUDA graph piece is captured.

        Callers place this after `projections` and before the op that ends the
        piece (an attention or GDP op), so the dispatch still overlaps the
        projections. Otherwise the experts stage waits, after the whole mixer.
        """
        dependency = torch.ops.vllm.nemotron_shortcut_moe_receive(
            dispatch.dependency, list(projections), self._op_layer_name()
        )
        return StagedDispatch(
            dependency, dispatch.hidden_states, dispatch.router_logits
        )

    def run_staged_experts(
        self, dispatch: StagedDispatch, current_path_output: torch.Tensor
    ) -> torch.Tensor:
        """Run the routed experts after `current_path_output` is computed."""
        return torch.ops.vllm.nemotron_shortcut_moe_experts(
            dispatch.dependency,
            current_path_output,
            dispatch.hidden_states,
            dispatch.router_logits,
            self._op_layer_name(),
        )

    def finish_staged(
        self,
        dependency: torch.Tensor,
        output_template: torch.Tensor,
        shared_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reduce, transform and add the shared output.

        `output_template` gives the result's shape and dtype.
        """
        return torch.ops.vllm.nemotron_shortcut_moe_finish(
            dependency, output_template, shared_output, self._op_layer_name()
        )

    def _begin_impl(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        slot = self._slot
        if self._routed[slot] is not None or self._combined[slot] is not None:
            raise RuntimeError(f"{self.layer_name}: a staged forward is pending")
        if self.overlaps_transport:
            logger.info_once(
                "Nemotron Shortcut MoE: all-gather and reduce-scatter run on a "
                "side stream, overlapped with the mixer and the shared experts"
            )
            # The mixer reads neither the routing nor the dispatch.
            routed = _launch_on_side_stream(
                hidden_states.device,
                self._dispatch_done,
                lambda: self._route(hidden_states, router_logits, dispatch=True),
            )
            routed.dispatch_pending = True
        else:
            routed = self._route(hidden_states, router_logits, dispatch=False)
        self._routed[slot] = routed
        return hidden_states.new_empty(0)

    def _route(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, dispatch: bool
    ) -> _Routed:
        block_input = hidden_states
        hidden_states, _ = self.apply_routed_input_transform(hidden_states)
        # The block input sizes the final output: the transform returns to it.
        hidden_states, pre_trunc, post_trunc = self._maybe_pad_hidden_states(
            block_input, hidden_states
        )
        self.routed_experts._ensure_moe_quant_config_init()
        with self._sequence_parallel_context():
            hidden_states, router_logits, hidden_states_scale = self._maybe_dispatch(
                hidden_states, router_logits
            )
            # Only the PCP gather pre-quantizes, and only for monolithic kernels.
            assert hidden_states_scale is None
            topk_weights, topk_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
            )
            tokens = None
            if dispatch:
                kernel = self._quant_method.moe_kernel
                assert kernel is not None
                tokens = dispatch_tokens(
                    kernel, self.routed_experts, hidden_states, topk_weights, topk_ids
                )
        return _Routed(
            hidden_states, topk_weights, topk_ids, pre_trunc, post_trunc, tokens
        )

    def _receive_impl(self) -> None:
        routed = self._routed[self._slot]
        if routed is None:
            raise RuntimeError(f"{self.layer_name}: begin_staged was not called")
        if routed.dispatch_pending and _capturing_graph_piece():
            current_stream().wait_event(self._dispatch_done)
            routed.dispatch_pending = False

    def _experts_impl(self, dependency: torch.Tensor) -> torch.Tensor:
        slot = self._slot
        routed = self._routed[slot]
        if routed is None:
            raise RuntimeError(f"{self.layer_name}: begin_staged was not called")
        self._routed[slot] = None
        with self._sequence_parallel_context():
            if routed.tokens is None:
                combined = self._run_experts(routed)
            else:
                combined = self._run_dispatched_experts(routed, routed.tokens)
        self._combined[slot] = combined
        return torch.empty_like(dependency)

    def _run_experts(self, routed: _Routed) -> _Combined:
        output = self.routed_experts.forward_modular(
            x=routed.hidden_states,
            topk_weights=routed.topk_weights,
            topk_ids=routed.topk_ids,
        )
        if isinstance(output, UnfinalizedMoEOutput):
            raise NotImplementedError("staged MoE needs a finalized output")
        output = self._maybe_combine(None, output)
        assert isinstance(output, torch.Tensor)
        return _Combined(
            output, routed.pre_transform_trunc_size, routed.post_transform_trunc_size
        )

    def _run_dispatched_experts(
        self, routed: _Routed, tokens: DispatchedTokens
    ) -> _Combined:
        if routed.dispatch_pending:
            current_stream().wait_event(self._dispatch_done)
        kernel = self._quant_method.moe_kernel
        assert kernel is not None
        fused_output = run_experts(
            kernel, self.routed_experts, tokens, self.routed_experts.activation
        )
        output = fused_output.new_empty(tokens.output_shape)
        # The caller's shared experts overlap the reduce-scatter.
        _launch_on_side_stream(
            output.device,
            self._combine_done,
            lambda: combine_tokens(kernel, tokens, fused_output, output),
        )
        return _Combined(
            output,
            routed.pre_transform_trunc_size,
            routed.post_transform_trunc_size,
            fused_output,
        )

    def _finish_impl(self, shared_output: torch.Tensor | None) -> torch.Tensor:
        slot = self._slot
        combined = self._combined[slot]
        if combined is None:
            raise RuntimeError(f"{self.layer_name}: run_staged_experts was not called")
        self._combined[slot] = None
        if combined.fused_output is not None:
            current_stream().wait_event(self._combine_done)
        # The same steps as MoERunner.forward after its custom op.
        output = combined.output
        if combined.pre_transform_trunc_size is not None:
            output = output[..., : combined.pre_transform_trunc_size]
        output, reduced = self._maybe_reduce_routed_output_before_transform(
            output, self._fused_output_is_reduced
        )
        shared_output = self._maybe_reduce_shared_expert_output(shared_output, reduced)
        shared_output, output = self._maybe_apply_routed_scale_to_output(
            shared_output, output
        )
        output = self.apply_routed_output_transform(output)
        if shared_output is not None:
            output = shared_output + output
        return self._maybe_reduce_final_output(
            output, combined.post_transform_trunc_size, reduced
        )


def _runner(layer_name: _LayerNameType) -> NemotronShortcutMoERunner:
    runner = get_layer_from_name(_resolve_layer_name(layer_name))
    assert isinstance(runner, NemotronShortcutMoERunner)
    return runner


def _begin(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    return _runner(layer_name)._begin_impl(hidden_states, router_logits)


def _begin_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    return hidden_states.new_empty(0)


def _receive(
    dependency: torch.Tensor,
    projections: list[torch.Tensor],
    layer_name: _LayerNameType,
) -> torch.Tensor:
    # `projections` is only an ordering input.
    _runner(layer_name)._receive_impl()
    return torch.empty_like(dependency)


def _receive_fake(
    dependency: torch.Tensor,
    projections: list[torch.Tensor],
    layer_name: _LayerNameType,
) -> torch.Tensor:
    return torch.empty_like(dependency)


def _experts(
    dependency: torch.Tensor,
    current_path_output: torch.Tensor,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    # Only `dependency` is read: `current_path_output` orders the op after the
    # mixer, and the routing inputs stay alive while the side stream reads them.
    return _runner(layer_name)._experts_impl(dependency)


def _experts_fake(
    dependency: torch.Tensor,
    current_path_output: torch.Tensor,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    return torch.empty_like(dependency)


def _finish(
    dependency: torch.Tensor,
    output_template: torch.Tensor,
    shared_output: torch.Tensor | None,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    output = _runner(layer_name)._finish_impl(shared_output)
    # The fake returns the template's dtype; keep the real op to it.
    return output.to(output_template.dtype)


def _finish_fake(
    dependency: torch.Tensor,
    output_template: torch.Tensor,
    shared_output: torch.Tensor | None,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    return torch.empty_like(output_template)


direct_register_custom_op(
    op_name="nemotron_shortcut_moe_begin",
    op_func=_begin,
    fake_impl=_begin_fake,
)
direct_register_custom_op(
    op_name="nemotron_shortcut_moe_receive",
    op_func=_receive,
    fake_impl=_receive_fake,
)
direct_register_custom_op(
    op_name="nemotron_shortcut_moe_experts",
    op_func=_experts,
    fake_impl=_experts_fake,
)
direct_register_custom_op(
    op_name="nemotron_shortcut_moe_finish",
    op_func=_finish,
    fake_impl=_finish_fake,
)
