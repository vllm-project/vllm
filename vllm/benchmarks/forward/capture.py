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
from typing import Any

import numpy as np
import torch

from vllm.benchmarks.forward.batch import BenchBatch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_world_group, graph_capture
from vllm.forward_context import get_forward_context
from vllm.v1.worker.gpu.input_batch import InputBatch


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
    ):
        self.runner = runner
        self.batch = batch
        self.device = runner.device
        self.iters = iters
        self.samples = samples
        self.state_views = state_views
        self.phase = "warmup"
        self.result: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        runner, model = self.runner, self.runner.model
        # The runner's eager branch calls `model(**inputs)`.
        assert runner.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        assert runner.ubatch_runner is None
        prepare_inputs = runner.prepare_inputs
        self._model_forward = model.forward

        def prepare_and_check(*args, **kwargs) -> InputBatch:
            self._input_batch = prepare_inputs(*args, **kwargs)
            self._check_batch(self._input_batch)
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
        self._write_input_ids(self._input_batch, inputs["input_ids"])
        # Models may update the forward context in place (e.g. sequence
        # parallelism shards is_padding); every call must start from the
        # context the runner built.
        context = get_forward_context()
        saved = dict(vars(context))

        def run_model() -> Any:
            for name, value in saved.items():
                setattr(context, name, value)
            return self._model_forward(**inputs)

        self._run_model = run_model
        if self.phase == "warmup":
            return run_model()
        return self._capture_and_time()

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
        self.result["num_tokens_after_padding"] = input_batch.num_tokens_after_padding
        self.result["batch_order"] = input_batch.req_ids

    def _write_input_ids(
        self, input_batch: InputBatch, input_ids: torch.Tensor | None
    ) -> None:
        """Decode/verify inputs otherwise come from sampler state."""
        if input_ids is None:
            return
        ids = np.concatenate(
            [self.batch.query_token_ids[req_id] for req_id in input_batch.req_ids]
        )
        input_ids[: len(ids)].copy_(torch.from_numpy(ids))

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
        self._events = (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
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
        self.result["latency_us"] = self._time(graph)
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
