# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture the forward of one exact batch as a one-shot CUDA graph and time it.

The graph is captured for a single batch and only replays that batch, so the
host-side decisions baked in at capture (attention plans, kernel choices by
query length, chunk splits) are correct by construction. The runner's CUDA
graph manager still dispatches the batch as serving would: padded token and
request counts, `is_padding`, and the graph mode seen by attention metadata
and the model. The bench captures that forward itself instead of replaying
the manager's graphs, which aren't captured. Each replay first restores the
KV blocks the step writes, so state the forward mutates in place (e.g.
compressor state) is the same on every replay.
"""

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch

import vllm.envs as envs
from vllm.benchmarks.forward.batch import BenchBatch
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    is_breakable_cudagraph_enabled,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_world_group, graph_capture
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.benchmarks.forward.speculator import DraftBench


class _BreakableCapture(BreakableCUDAGraphCapture):
    """Replays each eager segment in the forward context it was captured in.

    Serving replays a step's segments inside that step's forward context; the bench
    replays outside any, and one captured step spans several contexts (target,
    drafter), so eager ops reading attention metadata need theirs restored.
    """

    def add_eager(self, fn: Callable[[], Any]) -> Any:
        if not is_forward_context_available():
            return super().add_eager(fn)
        context = get_forward_context()

        def in_context() -> Any:
            with override_forward_context(context):
                return fn()

        return super().add_eager(in_context)


class _EagerReplay:
    """Serving's eager (NONE) steps run uncaptured: each "replay" reruns the step
    eagerly, launch overhead included, and copies the output into the first run's
    tensor so callers holding it see the latest result."""

    def __init__(self, step: Callable[[], Any], before: Callable[[], None]):
        self.step, self.before = step, before
        before()
        self.output = step()

    def replay(self) -> None:
        self.before()
        _first_tensor(self.output).copy_(_first_tensor(self.step()))


class _HostRendezvousReplay:
    """A breakable capture whose replays start after a host-side rendezvous."""

    def __init__(self, capture: BreakableCUDAGraphCapture, before: Callable[[], None]):
        self.capture, self.before = capture, before

    @property
    def num_eager_breaks(self) -> int:
        return self.capture.num_eager_breaks

    def replay(self) -> None:
        self.before()
        self.capture.replay()


_Capture: TypeAlias = torch.cuda.CUDAGraph | BreakableCUDAGraphCapture
_Graph: TypeAlias = _Capture | _EagerReplay | _HostRendezvousReplay


def _new_capture(vllm_config: Any, mode: str) -> _Capture:
    """The target's FULL steps capture whole, as serving's FULL graphs (drafter
    included); other modes break where serving's breakable graphs do."""
    if mode != CUDAGraphMode.FULL.name and is_breakable_cudagraph_enabled(
        vllm_config.compilation_config
    ):
        return _BreakableCapture(pool=torch.cuda.graph_pool_handle())
    return torch.cuda.CUDAGraph()


@functools.cache
def _capture_stream() -> torch.cuda.Stream:
    """One capture stream for every case, as serving captures on one stream. A
    new stream per case cycles through PyTorch's fixed stream pool; once it
    wrapped, later cases hit illegal memory accesses or NaNs."""
    return torch.cuda.Stream()


@contextmanager
def _capturing(graph: _Capture, stream: torch.cuda.Stream) -> Iterator[None]:
    if isinstance(graph, torch.cuda.CUDAGraph):
        with torch.cuda.graph(graph, stream=stream):
            yield
    else:
        with torch.cuda.stream(stream), graph:
            yield


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    assert isinstance(output, torch.Tensor), type(output)
    return output


def _compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - expected.float()
    scale = expected.float().square().mean().sqrt().clamp(min=1e-12)
    return {
        "max_abs": delta.abs().max().item(),
        "relative_rms": (delta.square().mean().sqrt() / scale).item(),
    }


def check_sampled(sampled: torch.Tensor, vocab_size: int) -> dict[str, Any]:
    # Rejection sampling leaves rows past each request's accepted tokens unset.
    first = sampled[:, 0]
    return {"valid_token_ids": bool(((first >= 0) & (first < vocab_size)).all())}


def as_dispatched(runner: Any) -> tuple[Callable[[], CUDAGraphMode], Callable]:
    """Make the runner dispatch as serving does but take its eager branch.

    The target's CUDA graph descriptor (padded token and request counts, graph
    mode) reaches each consumer that reads it before the forward; the runner
    then calls the model eagerly, and the caller restores the dispatched mode
    on the forward context. Returns a getter for the dispatched mode and a
    function that undoes the shadowing.
    """
    manager = runner.cudagraph_manager
    dispatched: list[Any] = [None]
    if manager is None or manager.cudagraph_mode == CUDAGraphMode.NONE:
        return lambda: CUDAGraphMode.NONE, lambda: None
    # (object, method, argument index, descriptor -> argument).
    consumers: list[tuple[Any, str, int, Callable]] = [
        (runner.model_state, "prepare_attn", 1, lambda desc: desc.cg_mode)
    ]
    if runner.pcp_manager is not None:
        consumers.append((runner.pcp_manager, "partition_batch", 1, lambda d: d))
    if runner.fast_prefill is not None:
        consumers.append((runner.fast_prefill, "prepare", 4, lambda d: d))
    pending: set[str] = set()
    dispatch = manager.dispatch
    import vllm.v1.worker.gpu.model_runner as runner_module

    sync_dispatch = runner_module.dispatch_cg_and_sync_dp

    def eager_dispatch(*args, **kwargs):
        # Other dispatchers (drafters) run eagerly, unpadded.
        return replace(dispatch(*args, **kwargs), cg_mode=CUDAGraphMode.NONE)

    def eager_sync_dispatch(*args, **kwargs):
        # The target's descriptor as serving agrees on it across DP ranks (the
        # lowest mode any rank needs, a common padded size), run eagerly.
        manager.dispatch = dispatch
        try:
            desc, dp_sync_state = sync_dispatch(*args, **kwargs)
        finally:
            manager.dispatch = eager_dispatch
        dispatched[0] = desc
        pending.update(name for _, name, _, _ in consumers)
        return replace(desc, cg_mode=CUDAGraphMode.NONE), dp_sync_state

    def substitute(fn: Callable, name: str, index: int, value: Callable):
        def wrapper(*args, **kwargs):
            # Once per dispatch: drafters may share these objects.
            if name in pending:
                pending.discard(name)
                args = (*args[:index], value(dispatched[0]), *args[index + 1 :])
            return fn(*args, **kwargs)

        return wrapper

    manager.dispatch = eager_dispatch
    runner_module.dispatch_cg_and_sync_dp = eager_sync_dispatch
    for obj, name, index, value in consumers:
        setattr(obj, name, substitute(getattr(obj, name), name, index, value))

    def undo() -> None:
        del manager.dispatch
        runner_module.dispatch_cg_and_sync_dp = sync_dispatch
        for obj, name, _, _ in consumers:
            delattr(obj, name)

    def mode() -> CUDAGraphMode:
        return CUDAGraphMode.NONE if dispatched[0] is None else dispatched[0].cg_mode

    return mode, undo


def write_input_ids(
    batch: BenchBatch, input_batch: InputBatch, input_ids: torch.Tensor | None
) -> None:
    """Decode/verify inputs otherwise come from sampler state."""
    if input_ids is None:
        return
    # PCP can split a request into two disjoint rows or replicate it.
    segments = []
    for i, req_id in enumerate(input_batch.req_ids):
        start = int(input_batch.num_computed_tokens_np[i]) - batch.expected[req_id][1]
        count = int(input_batch.num_scheduled_tokens[i])
        tokens = batch.query_token_ids[req_id]
        assert 0 <= start <= start + count <= len(tokens)
        segments.append(tokens[start : start + count])
    ids = np.concatenate(segments)
    assert len(ids) == input_batch.num_tokens
    input_ids[: len(ids)].copy_(torch.from_numpy(ids))


class ForwardBench:
    """Runs one batch through the runner's `execute_model`, timing its forward.

    `execute_model` prepares the inputs, attention metadata and forward context
    as in serving; the bench takes over the eager model call. Two steps: the
    first ("warmup") runs the model eagerly, which triggers JIT, autotuning and
    lazy init. The second ("capture") captures, validates and times the graph.
    """

    def __init__(
        self,
        runner: Any,
        batch: BenchBatch,
        iters: int,
        samples: int,
        state_views: list[torch.Tensor],
        name: str,
        profile_prefix: str | None = None,
        *,
        include_drafter: bool = True,
        include_sampler: bool = True,
    ):
        self.runner = runner
        self.batch = batch
        self.device = runner.device
        self.iters = iters
        self.samples = samples
        self.state_views = state_views
        self.name = name
        self.profile_prefix = profile_prefix
        self.phase = "warmup"
        self.result: dict[str, Any] = {}
        # Called with the sample index before each timing sample (e.g. a routing
        # re-draw), and with 0 after timing.
        self.before_sample: Callable[[int], None] | None = None
        self.include_drafter = include_drafter
        self.drafter: DraftBench | None = None
        self.include_sampler = include_sampler
        self.sampled: torch.Tensor | None = None

    def run(self) -> dict[str, Any]:
        runner, model = self.runner, self.runner.model
        assert runner.ubatch_runner is None
        prepare_inputs = runner.prepare_inputs
        self._model_forward = model.forward

        def prepare_and_check(*args, **kwargs) -> InputBatch:
            self._input_batch = prepare_inputs(*args, **kwargs)
            pcp = runner.pcp_manager
            self._check_batch(pcp.global_batch if pcp else self._input_batch)
            self.result["num_local_tokens"] = self._input_batch.num_tokens
            self.result["num_tokens_after_padding"] = (
                self._input_batch.num_tokens_after_padding
            )
            self.result["num_reqs_after_padding"] = (
                self._input_batch.num_reqs_after_padding
            )
            if envs.VLLM_MOE_SKIP_PADDING:
                self.result["num_padding_tokens"] = int(
                    self._input_batch.is_padding.sum()
                )
            self.result["batch_order"] = self._input_batch.req_ids
            return self._input_batch

        # Instance attributes shadow the methods for these two steps only.
        runner.prepare_inputs = prepare_and_check
        model.forward = self._forward
        self._cudagraph_mode, undo_dispatch = as_dispatched(runner)
        try:
            for self.phase in ("warmup", "capture"):
                runner.execute_model(self.batch.scheduler_output)
        finally:
            del runner.prepare_inputs, model.forward
            undo_dispatch()
            # No sample_tokens follows.
            runner.execute_model_state = None
        return self.result

    def _forward(self, **inputs) -> Any:
        write_input_ids(self.batch, self._input_batch, inputs["input_ids"])
        self._run_model = self._make_forward(inputs)
        if self.phase == "warmup":
            return self._run_model()
        return self._capture_and_time()

    def _make_forward(self, inputs):
        # Models may update the forward context in place (e.g. sequence
        # parallelism shards is_padding); every call must start from the
        # context the runner built, in the mode serving would run.
        context = get_forward_context()
        context.cudagraph_runtime_mode = self._cudagraph_mode()
        self.result["cudagraph_mode"] = context.cudagraph_runtime_mode.name
        saved = dict(vars(context))
        if self.include_drafter and self.runner.speculator is not None:
            from vllm.benchmarks.forward.speculator import DraftBench

            if self.drafter is None:
                self.drafter = DraftBench(self.runner)
                # Drafting reuses the target's slot mappings and PCP block tables.
                self.state_views.extend(
                    [self.runner.block_tables.slot_mappings]
                    + self.runner.block_tables.input_block_tables
                )
            self.drafter.prepare(self._input_batch, context)
        # Only the last PP rank samples.
        sample = self.include_sampler and self.runner.is_last_pp_rank
        self.result["timing_scope"] = {
            (False, False): "target_only",
            (True, False): "target_and_sampler",
            (False, True): "target_and_drafter",
            (True, True): "target_sampler_and_drafter",
        }[(sample, self.drafter is not None)]

        def run_model() -> Any:
            for name, value in saved.items():
                setattr(context, name, value)
            with override_forward_context(context):
                output = self._model_forward(**inputs)
            handler = self.runner.pp_handler
            if not self.runner.is_last_pp_rank:
                # The pipeline bench sends this output to the next stage itself.
                output = handler.relay_aux_hidden_states(
                    inputs["intermediate_tensors"], output
                )
            else:
                if sample:
                    self._sample(output)
                if self.drafter is not None:
                    # The runner completes the returned output itself; only the
                    # drafter needs the full aux hidden states here.
                    draft_input = output
                    if self.runner.use_aux_hidden_state_outputs and handler is not None:
                        draft_input = handler.add_upstream_aux_hidden_states(
                            inputs["intermediate_tensors"], output
                        )
                    self.drafter.propose(draft_input)
            return output

        return run_model

    def _sample(self, output: Any) -> None:
        """Sample, or rejection-sample verify rows, as `sample_tokens` does.

        The drafter keeps its fixed fully-accepted inputs, so the sampled
        tokens feed nothing.
        """
        runner = self.runner
        hidden = _first_tensor(output)
        input_batch = self._input_batch
        if runner.pcp_manager is not None:
            hidden, _, input_batch = runner.pcp_manager.restore_for_sampling(
                hidden, None
            )
        self.sampled = runner.sample(hidden, input_batch, None)[0].sampled_token_ids

    def _real_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rows of real tokens: CUDA graph padding rows are undefined."""
        batch = self._input_batch
        if tensor.shape[0] == batch.num_tokens_after_padding:
            return tensor[: batch.num_tokens]
        return tensor

    def _check_batch(self, input_batch: InputBatch) -> None:
        """Fail unless the runner built exactly the requested batch."""
        expected = self.batch.expected
        assert sorted(input_batch.req_ids) == sorted(expected), input_batch.req_ids
        for i, req_id in enumerate(input_batch.req_ids):
            actual = (
                int(input_batch.num_scheduled_tokens[i]),
                int(input_batch.num_computed_tokens_np[i]),
                not bool(input_batch.is_prefilling_np[i]),
            )
            assert actual == expected[req_id], (
                f"{req_id}: got (scheduled, computed, decode) = {actual}, "
                f"wanted {expected[req_id]}"
            )
        assert input_batch.num_tokens == self.batch.num_tokens

    def _capture_and_time(self) -> Any:
        world = get_world_group()
        # The forward mutates state it reads (e.g. compressor state), so
        # every run starts by restoring the KV blocks the step writes.
        snapshots = [(view, view.clone()) for view in self.state_views]

        def restore() -> None:
            for view, saved in snapshots:
                view.copy_(saved)

        # A rendezvous and the timing events live inside the graph, after the
        # restore: neither rank skew, the restore nor host launch gaps reach
        # the timed interval.
        flag = torch.zeros(1, device=self.device)
        # TODO(woosuk): Support ROCm. The bench hangs there, likely on these
        # in-graph events.
        # `external=True` lets the event be queried outside the graph without
        # being captured itself; CUDA-only, XPU graphs don't need it.
        event_kwargs = (
            {"enable_timing": True, "external": True}
            if current_platform.is_cuda_alike()
            else {"enable_timing": True}
        )
        self._events = (
            torch.cuda.Event(**event_kwargs),
            torch.cuda.Event(**event_kwargs),
        )
        if world.world_size > 1:
            # Initialize the communicator outside capture.
            torch.distributed.all_reduce(flag, group=world.device_group)
        with graph_capture(self.device, _capture_stream()) as capture_context:
            stream = capture_context.stream
            # Some kernels set up per-stream state on first use, outside capture.
            eager = self._real_rows(_first_tensor(self._run_model())).clone()
            restore()
            reference = self._real_rows(_first_tensor(self._run_model())).clone()
            draft_reference = self.drafter.tokens.clone() if self.drafter else None
            stream.synchronize()
            # As the runner captures: with breakable graphs on, ops marked as
            # eager breaks (e.g. attention in PIECEWISE steps) run eagerly
            # between graph segments at replay.

            # Ranks start each timed step together. FULL graphs rendezvous in the
            # graph (a device all-reduce); steps with eager work rendezvous on the
            # host: a device collective queued ahead of eager MegaMoE (which
            # occupies every SM waiting for its peers) can deadlock across ranks
            # once the host runs ahead of the GPU.
            in_graph = self.result["cudagraph_mode"] == CUDAGraphMode.FULL.name

            def host_rendezvous() -> None:
                if world.world_size > 1:
                    torch.accelerator.synchronize()
                    torch.distributed.barrier(group=world.cpu_group)

            def step() -> Any:
                restore()
                if in_graph and world.world_size > 1:
                    torch.distributed.all_reduce(flag, group=world.device_group)
                self._events[0].record()
                output = self._run_model()
                self._events[1].record()
                return output

            graph: _Graph
            if self.result["cudagraph_mode"] == CUDAGraphMode.NONE.name:
                graph = _EagerReplay(step, host_rendezvous)
                output = graph.output
            else:
                capture = _new_capture(
                    self.runner.vllm_config, self.result["cudagraph_mode"]
                )
                with _capturing(capture, stream):
                    output = step()
                graph = (
                    capture
                    if in_graph or isinstance(capture, torch.cuda.CUDAGraph)
                    else _HostRendezvousReplay(capture, host_rendezvous)
                )
        hidden = self._real_rows(_first_tensor(output))
        if isinstance(graph, (BreakableCUDAGraphCapture, _HostRendezvousReplay)):
            self.result["eager_breaks"] = graph.num_eager_breaks

        sampled = self.sampled
        graph.replay()
        torch.accelerator.synchronize()
        first = hidden.clone()
        first_sampled = sampled.clone() if sampled is not None else None
        graph.replay()
        torch.accelerator.synchronize()
        if sampled is not None:
            assert first_sampled is not None
            self.result["sampler"] = {
                **check_sampled(sampled, self.runner.model_config.get_vocab_size()),
                "graph_deterministic": bool(torch.equal(first_sampled, sampled)),
            }
        self.result.update(
            # Same step from the same restored state: should match exactly.
            eager_repeat=_compare(reference, eager),
            graph_vs_eager=_compare(first, reference),
            graph_deterministic=bool(torch.equal(first, hidden)),
            finite=bool(hidden.isfinite().all()),
            state_reset_bytes=sum(
                v.numel() * v.element_size() for v in self.state_views
            ),
        )
        if self.drafter is not None:
            self.result["drafter"] = self.drafter.check(draft_reference)
        self.result["latency_us"] = self._time(graph)
        from vllm.benchmarks.forward.profile import profile_graph

        # Each replay restores the state views and, in FULL graphs, rendezvouses
        # before the timed forward.
        skip = len(self.state_views) + (world.world_size > 1 and in_graph)
        summary = profile_graph(graph, self.name, self.profile_prefix, skip)
        if summary is not None:
            self.result["profile"] = {
                "summary": f"{self.profile_prefix}.kernels.json",
                "span_us": summary["span_us"],
                "busy_us": summary["busy_us"],
                "activities": summary["activities"],
                "top_kernels": summary["kernels"][:10],
            }
        del graph
        return output

    def _time(self, graph: _Graph) -> dict[str, Any]:
        """Per-replay forward time between in-graph events, MAX across ranks."""
        world = get_world_group()
        start, end = self._events
        self._warmup(graph)
        times = []
        for sample in range(self.samples):
            if self.before_sample is not None and sample > 0:
                self.before_sample(sample)
                for _ in range(2):  # untimed: settle on the new draw
                    graph.replay()
                torch.accelerator.synchronize()
            for _ in range(self.iters):
                graph.replay()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1000)
        if self.before_sample is not None:
            self.before_sample(0)
        arr = torch.tensor(times, device=self.device)
        if world.world_size > 1:
            torch.distributed.all_reduce(
                arr, op=torch.distributed.ReduceOp.MAX, group=world.device_group
            )
        arr = arr.cpu().numpy()
        sample_medians = np.median(arr.reshape(self.samples, -1), axis=1)
        median = float(np.median(arr))
        return {
            "timer": "graph_events" + ("_rank_max" if world.world_size > 1 else ""),
            "median": median,
            # Stability: spread of per-sample medians relative to the median.
            "sample_spread_pct": float(
                100 * (sample_medians.max() - sample_medians.min()) / median
            ),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p90": float(np.percentile(arr, 90)),
            "count": len(arr),
            "samples_us": [round(float(t), 2) for t in arr],
        }

    def _warmup(self, graph: _Graph, seconds: float = 1.0) -> None:
        """Replay until clocks and caches reach a steady state."""
        world = get_world_group()
        graph.replay()
        torch.accelerator.synchronize()
        begin = time.perf_counter()
        graph.replay()
        torch.accelerator.synchronize()
        per_replay = max(time.perf_counter() - begin, 1e-6)
        # Same count on every rank: the graph may contain collectives.
        count = torch.tensor([max(self.iters, int(seconds / per_replay))])
        torch.distributed.all_reduce(
            count, op=torch.distributed.ReduceOp.MAX, group=world.cpu_group
        )
        for _ in range(int(count.item())):
            graph.replay()
        torch.accelerator.synchronize()
