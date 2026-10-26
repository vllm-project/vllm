# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline forwards with identical microbatches and no per-step host barrier."""

import gzip
import json
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm.benchmarks.forward.capture import ForwardBench, _compare, write_input_ids
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
    get_world_group,
    graph_capture,
)
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.sequence import IntermediateTensors
from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher


class ForwardPipelineExecutor(ExecutorWithExternalLauncher):
    """Load one stage per torchrun process; PipelineBench drives execution.

    Serving's external launcher requires sampling broadcasts across PP stages.
    The forward benchmark has no sampling or independent schedulers to sync.
    """

    supports_pp = True

    def execute_model(self, *args, **kwargs):
        raise RuntimeError("ForwardPipelineExecutor requires PipelineBench")


def _tensors(output: Any) -> dict[str, torch.Tensor]:
    if isinstance(output, IntermediateTensors):
        return output.tensors
    if isinstance(output, tuple):
        output = output[0]
    return {"hidden_states": output}


class PipelineClock:
    """Correlate local GPU events with rank zero's monotonic clock.

    CPU ping-pong bounds clock-offset error without assuming synchronized node
    clocks. An idle-GPU event is bracketed on the host to correlate GPU time.
    Both bounds are retained alongside the reported latency.
    """

    def __init__(self, trials: int = 20):
        world = get_world_group()
        group = world.cpu_group
        offsets = [(0.0, 0.0)] * world.world_size
        message = torch.zeros(1, dtype=torch.int64)
        for peer in range(1, world.world_size):
            candidates = []
            for _ in range(trials):
                if world.rank == 0:
                    begin = time.perf_counter_ns()
                    dist.send(message, peer, group=group)
                    dist.recv(message, peer, group=group)
                    end = time.perf_counter_ns()
                    candidates.append(((begin + end) / 2 - message.item(), end - begin))
                elif world.rank == peer:
                    dist.recv(message, 0, group=group)
                    message[0] = time.perf_counter_ns()
                    dist.send(message, 0, group=group)
            if world.rank == 0:
                offset, rtt = min(candidates, key=lambda x: x[1])
                offsets[peer] = (offset / 1000, rtt / 2000)
        values = [offsets]
        dist.broadcast_object_list(values, src=0, group=group)
        offset_us, uncertainty_us = values[0][world.rank]
        torch.accelerator.synchronize()
        anchors = []
        for _ in range(trials):
            event = torch.cuda.Event(enable_timing=True)
            begin = time.perf_counter_ns()
            event.record()
            event.synchronize()
            end = time.perf_counter_ns()
            anchors.append((end - begin, (begin + end) / 2000, event))
        width, midpoint, self.anchor = min(anchors, key=lambda x: x[0])
        self.origin_us = midpoint + offset_us
        self.uncertainty_us = uncertainty_us + width / 2000

    def timestamp(self, event: torch.cuda.Event, end: "PipelineClock") -> float:
        fraction = self.anchor.elapsed_time(event) / self.anchor.elapsed_time(
            end.anchor
        )
        return self.origin_us + fraction * (end.origin_us - self.origin_us)


class PipelineBench(ForwardBench):
    """Capture each stage, then stream matching microbatches through the stages."""

    def run(self) -> dict[str, Any]:
        runner, model = self.runner, self.runner.model
        pp = get_pp_group()
        prepare = runner.prepare_inputs
        self._model_forward = model.forward

        def prepare_and_check(*args, **kwargs):
            batch = prepare(*args, **kwargs)
            self._input_batch = batch
            self._check_batch(
                runner.pcp_manager.global_batch if runner.pcp_manager else batch
            )
            self.result.update(
                num_local_tokens=batch.num_tokens,
                num_tokens_after_padding=batch.num_tokens_after_padding,
                batch_order=batch.req_ids,
                pp_rank=pp.rank_in_group,
            )
            return batch

        runner.prepare_inputs = prepare_and_check
        model.forward = self._prepare_stage
        try:
            runner.execute_model(
                self.batch.scheduler_output,
                intermediate_tensors=runner.intermediate_tensors,
            )
        finally:
            del runner.prepare_inputs, model.forward
            runner.execute_model_state = None
        self._capture_stage()
        self._stream()
        self._stream(profile=True)
        return self.result

    def _prepare_stage(self, **inputs):
        pp = get_pp_group()
        write_input_ids(self.batch, self._input_batch, inputs["input_ids"])
        self.input_tensors = inputs["intermediate_tensors"]
        if not pp.is_first_rank:
            received = pp.recv_tensor_dict(all_gather_group=get_tp_group())
            assert received is not None
            for key, tensor in self.input_tensors.tensors.items():
                tensor.copy_(received[key])
        context = get_forward_context()
        saved = dict(vars(context))

        def forward():
            for name, value in saved.items():
                setattr(context, name, value)
            with override_forward_context(context):
                return self._model_forward(**inputs)

        self._run_model = forward
        self.snapshots = [(view, view.clone()) for view in self.state_views]
        self.input_snapshots = (
            {k: v.clone() for k, v in self.input_tensors.tensors.items()}
            if self.input_tensors is not None
            else {}
        )
        output = forward()
        self.reference = {k: v.clone() for k, v in _tensors(output).items()}
        if not pp.is_last_rank:
            pp.send_tensor_dict(_tensors(output), all_gather_group=get_tp_group())
        return output

    def _restore(self):
        for view, snapshot in self.snapshots:
            view.copy_(snapshot)

    def _capture_stage(self):
        with graph_capture(self.device) as capture_context:
            for key, tensor in self.input_snapshots.items():
                self.input_tensors.tensors[key].copy_(tensor)
            self._restore()
            self._run_model()
            for key, tensor in self.input_snapshots.items():
                self.input_tensors.tensors[key].copy_(tensor)
            capture_context.stream.synchronize()
            self.reset_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.reset_graph, stream=capture_context.stream):
                self._restore()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=capture_context.stream):
                self.output = self._run_model()
        for key, tensor in self.input_snapshots.items():
            self.input_tensors.tensors[key].copy_(tensor)
        self.reset_graph.replay()
        self.graph.replay()
        torch.accelerator.synchronize()
        comparisons = {
            key: _compare(value, self.reference[key])
            for key, value in _tensors(self.output).items()
        }
        self.result.update(
            graph_vs_eager=max(comparisons.values(), key=lambda x: x["relative_rms"]),
            intermediate_graph_vs_eager=comparisons,
            state_reset_bytes=sum(
                v.numel() * v.element_size() for v in self.state_views
            ),
        )

    def _stream(self, profile: bool = False):
        pp, world = get_pp_group(), get_world_group()
        tp = get_tp_group()
        received = []
        if not pp.is_first_rank:
            for key, tensor in self.input_tensors.tensors.items():
                if not tensor.numel():
                    continue
                sharded = pp._should_use_all_gather(key, tensor.numel(), tp, None)
                buffer = (
                    torch.empty_like(tensor.reshape(tp.world_size, -1)[0])
                    if sharded
                    else tensor
                )
                received.append((tensor, buffer, sharded))
        sent = []
        if not pp.is_last_rank:
            for key, tensor in _tensors(self.output).items():
                if tensor.numel():
                    if pp._should_use_all_gather(key, tensor.numel(), tp, None):
                        tensor = tensor.reshape(tp.world_size, -1)[tp.rank_in_group]
                    sent.append(tensor)
        clock = None if profile else PipelineClock()
        # Warmup and drain are excluded by microbatch ID, not by local stage time.
        margin = 4 * pp.world_size
        count = 3 if profile else self.iters * self.samples
        starts, ends = [], []
        dist.barrier(group=world.cpu_group)
        torch_profile = profile and self.profile_prefix is not None
        profiler = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            if torch_profile
            else nullcontext()
        )
        if profile and not torch_profile:
            torch.cuda.profiler.start()
        scope = (
            torch.cuda.nvtx.range(f"forward/{self.name}") if profile else nullcontext()
        )
        with profiler as prof, scope:
            for _ in range(count + 2 * margin):
                self.reset_graph.replay()
                for tensor, buffer, sharded in received:
                    dist.irecv(
                        buffer,
                        pp.ranks[pp.rank_in_group - 1],
                        group=pp.device_group,
                    ).wait()
                    if sharded:
                        tensor.copy_(tp.all_gather(buffer, dim=0).reshape(tensor.shape))
                if not profile:
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                self.graph.replay()
                if not profile:
                    end.record()
                    starts.append(start)
                    ends.append(end)
                for tensor in sent:
                    dist.isend(
                        tensor,
                        pp.ranks[pp.rank_in_group + 1],
                        group=pp.device_group,
                    ).wait()
            torch.accelerator.synchronize()
        if profile and not torch_profile:
            torch.cuda.profiler.stop()
            return
        if profile:
            from vllm.benchmarks.forward.profile import summarize

            assert prof is not None
            trace = f"{self.profile_prefix}.trace.json.gz"
            prof.export_chrome_trace(trace)
            with gzip.open(trace, "rt") as f:
                events = json.load(f)["traceEvents"]
            summary = {
                "trace": trace,
                **summarize(events, count + 2 * margin, len(self.state_views)),
            }
            summary_path = f"{self.profile_prefix}.kernels.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=1)
            self.result["profile"] = {
                "summary": summary_path,
                "scope": "pipeline_stage_including_transfers",
                "span_us": summary["span_us"],
                "busy_us": summary["busy_us"],
                "activities": summary["activities"],
                "top_kernels": summary["kernels"][:10],
            }
            return
        self.result["pipeline_vs_eager"] = {
            key: _compare(value, self.reference[key])
            for key, value in _tensors(self.output).items()
        }
        self.result["finite"] = all(
            bool(t.isfinite().all()) for t in _tensors(self.output).values()
        )
        end_clock = PipelineClock()
        assert clock is not None
        local = {
            "stage": pp.rank_in_group,
            "dp_rank": self.runner.dp_rank,
            "starts_us": [clock.timestamp(e, end_clock) for e in starts],
            "ends_us": [clock.timestamp(e, end_clock) for e in ends],
            "clock_uncertainty_us": max(clock.uncertainty_us, end_clock.uncertainty_us),
        }
        gathered: list[Any] = [None] * world.world_size
        dist.all_gather_object(gathered, local, group=world.cpu_group)
        latencies, uncertainties = [], []
        for dp_rank in {r["dp_rank"] for r in gathered}:
            replica = [r for r in gathered if r["dp_rank"] == dp_rank]
            first = [r for r in replica if r["stage"] == 0]
            last = [r for r in replica if r["stage"] == pp.world_size - 1]
            latencies.append(
                np.max([r["ends_us"] for r in last], axis=0)
                - np.min([r["starts_us"] for r in first], axis=0)
            )
            uncertainties.append(
                max(r["clock_uncertainty_us"] for r in first)
                + max(r["clock_uncertainty_us"] for r in last)
            )
        latency = np.max(latencies, axis=0)[margin:-margin]
        medians = np.median(latency.reshape(self.samples, self.iters), axis=1)
        self.result.update(
            pipeline_timeline=local,
            pipeline_warmup_microbatches=margin,
            clock_uncertainty_us=max(uncertainties),
            latency_us={
                "timer": "pipeline_correlated_gpu_events_rank_max",
                "median": float(np.median(latency)),
                "mean": float(latency.mean()),
                "min": float(latency.min()),
                "p90": float(np.percentile(latency, 90)),
                "std": float(latency.std()),
                "count": count,
                "samples_us": latency.tolist(),
                "sample_spread_pct": float(100 * np.ptp(medians) / np.median(latency)),
            },
        )
