# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture the forward of one exact batch as a one-shot CUDA graph and time it.

The graph is captured for a single batch and only replays that batch, so the
host-side decisions baked in at capture (attention plans, kernel choices by
query length, chunk splits) are correct by construction. This deliberately
bypasses vLLM's CUDA graph manager: no capture sizes, padding or dummy
metadata. Each replay first restores the KV blocks the step writes, so state the
forward mutates in place (e.g. compressor state) is the same on every replay.
"""

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.benchmarks.forward.batch import BenchBatch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_world_group, graph_capture
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.benchmarks.forward.speculator import DraftBench


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
        self.include_drafter = include_drafter
        self.drafter: DraftBench | None = None

    def run(self) -> dict[str, Any]:
        runner, model = self.runner, self.runner.model
        # The runner's eager branch calls `model(**inputs)`.
        assert runner.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
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
            self.result["batch_order"] = self._input_batch.req_ids
            return self._input_batch

        # Instance attributes shadow the methods for these two steps only.
        runner.prepare_inputs = prepare_and_check
        model.forward = self._forward
        try:
            for self.phase in ("warmup", "capture"):
                runner.execute_model(self.batch.scheduler_output)
        finally:
            del runner.prepare_inputs, model.forward
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
        # context the runner built.
        context = get_forward_context()
        saved = dict(vars(context))
        saved["cudagraph_runtime_mode"] = CUDAGraphMode.FULL
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
        self.result["timing_scope"] = (
            "target_and_drafter" if self.drafter is not None else "target_only"
        )

        def run_model() -> Any:
            for name, value in saved.items():
                setattr(context, name, value)
            with override_forward_context(context):
                output = self._model_forward(**inputs)
            if not self.runner.is_last_pp_rank:
                output = self.runner.pp_handler.relay_aux_hidden_states(
                    inputs["intermediate_tensors"], output
                )
            if self.drafter is not None:
                self.drafter.propose(output)
            return output

        return run_model

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
        with graph_capture(self.device) as capture_context:
            stream = capture_context.stream
            # Some kernels set up per-stream state on first use, outside capture.
            eager = _first_tensor(self._run_model()).clone()
            restore()
            reference = _first_tensor(self._run_model()).clone()
            draft_reference = self.drafter.tokens.clone() if self.drafter else None
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                restore()
                if world.world_size > 1:
                    torch.distributed.all_reduce(flag, group=world.device_group)
                self._events[0].record()
                output = self._run_model()
                self._events[1].record()
        hidden = _first_tensor(output)

        graph.replay()
        torch.accelerator.synchronize()
        first = hidden.clone()
        graph.replay()
        torch.accelerator.synchronize()
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

        # Each replay restores the state views, then (multi-rank) rendezvouses,
        # before the timed forward.
        skip = len(self.state_views) + (world.world_size > 1)
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

    def _time(self, graph: torch.cuda.CUDAGraph) -> dict[str, Any]:
        """Per-replay forward time between in-graph events, MAX across ranks."""
        world = get_world_group()
        start, end = self._events
        self._warmup(graph)
        times = []
        for _ in range(self.iters * self.samples):
            graph.replay()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000)
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

    def _warmup(self, graph: torch.cuda.CUDAGraph, seconds: float = 1.0) -> None:
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
