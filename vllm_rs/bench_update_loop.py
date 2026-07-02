"""Microbench: scheduler.update_from_output() per-request loop body.

Reproduces the hot loop at vllm/v1/core/sched/scheduler.py:1345-1479 in
isolation, so we can measure what Python overhead is actually incurred
per iteration at 256 reqs × hundreds of steps — and then compare against
a Rust-ported prefix + Python-action variant.

No vLLM required. All types reimplemented locally with the same attributes
as the real classes so per-iteration attribute-access cost is representative.

Run:
    python vllm_rs/bench_update_loop.py [--num-reqs 256] [--spec-frac 0.5] [--iters 500]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vllm_rs  # noqa: E402


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()


@dataclass(slots=True)
class SamplingParams:
    min_tokens: int = 0
    max_tokens: int = 1024
    eos_token_id: int | None = None
    stop_token_ids: tuple = ()
    repetition_detection: object | None = None
    logprobs: int | None = None


@dataclass(slots=True)
class Request:
    request_id: str
    status: RequestStatus = RequestStatus.RUNNING
    num_computed_tokens: int = 100
    num_output_placeholders: int = 0
    num_prompt_tokens: int = 100
    num_tokens: int = 100
    max_tokens: int = 512
    num_cached_tokens: int = 0
    num_external_computed_tokens: int = 0
    num_nans_in_logits: int = 0
    client_index: int = 0
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    pooling_params: object | None = None
    structured_output_request: object | None = None
    trace_headers: object | None = None
    stop_reason: object | None = None
    output_token_ids: list[int] = field(default_factory=list)
    all_token_ids: list[int] = field(default_factory=list)
    lora_request: object | None = None
    mm_features: list = field(default_factory=list)
    _events: list = field(default_factory=list)

    def is_finished(self) -> bool:
        return self.status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        )

    def append_output_token_ids(self, tok: int):
        self.output_token_ids.append(tok)
        self.num_tokens += 1

    def get_finished_reason(self):
        return None

    def take_events(self):
        if not self._events:
            return None
        evs = self._events
        self._events = []
        return evs


@dataclass(slots=True)
class EngineCoreOutput:
    # mirrors vllm.v1.engine.EngineCoreOutput field list (msgspec.Struct in real code)
    request_id: str
    new_token_ids: list[int]
    finish_reason: object | None
    new_logprobs: object | None
    new_prompt_logprobs_tensors: object | None
    pooling_output: object | None
    stop_reason: object | None
    events: object | None
    kv_transfer_params: object | None
    trace_headers: object | None
    num_cached_tokens: int
    num_external_computed_tokens: int
    routed_experts: object | None
    num_nans_in_logits: int


@dataclass(slots=True)
class SchedulerOutput:
    scheduled_spec_decode_tokens: dict[str, list[int]]
    num_invalid_spec_tokens: int = 0


@dataclass(slots=True)
class ModelRunnerOutput:
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]


def make_workload(num_reqs: int, spec_frac: float, seed: int = 0):
    import random
    rng = random.Random(seed)
    requests: dict[str, Request] = {}
    num_scheduled_tokens: dict[str, int] = {}
    scheduled_spec: dict[str, list[int]] = {}
    req_id_to_index: dict[str, int] = {}
    sampled_token_ids: list[list[int]] = []
    for i in range(num_reqs):
        rid = f"req-{i}"
        requests[rid] = Request(
            request_id=rid,
            num_computed_tokens=rng.randint(50, 200),
            num_output_placeholders=0,
            num_prompt_tokens=rng.randint(50, 200),
            num_tokens=rng.randint(50, 200),
            max_tokens=rng.randint(256, 1024),
            sampling_params=SamplingParams(
                eos_token_id=2,
                stop_token_ids=(5, 7),
            ),
        )
        num_scheduled_tokens[rid] = 1
        req_id_to_index[rid] = i
        if rng.random() < spec_frac:
            scheduled_spec[rid] = [rng.randint(100, 999) for _ in range(3)]
            # 1 + (accepted) generated tokens (accepted < num_draft)
            accepted = rng.randint(1, 3)
            sampled_token_ids.append([rng.randint(100, 999) for _ in range(accepted + 1)])
        else:
            sampled_token_ids.append([rng.randint(100, 999)])
    return requests, num_scheduled_tokens, scheduled_spec, req_id_to_index, sampled_token_ids


# ---------- Python reference: faithful reimplementation of the hot loop ----------

def py_update_loop(requests, num_scheduled_tokens, scheduled_spec, req_id_to_index,
                   sampled_token_ids, failed_kv_load_req_ids, max_model_len=4096):
    outputs: dict[int, list[EngineCoreOutput]] = {}
    stopped_running: set[Request] = set()
    spec_stats_accum = {"num_draft": 0, "num_accepted": 0, "num_invalid": 0}
    num_invalid_spec_tokens = 0

    for req_id, n in num_scheduled_tokens.items():
        assert n > 0
        if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
            continue
        request = requests.get(req_id)
        if request is None or request.is_finished():
            continue

        req_index = req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

        scheduled_spec_token_ids = scheduled_spec.get(req_id)
        if scheduled_spec_token_ids and generated_token_ids:
            num_draft_tokens = len(scheduled_spec_token_ids)
            num_accepted = len(generated_token_ids) - 1
            num_rejected = num_draft_tokens - num_accepted
            if request.num_computed_tokens > 0:
                request.num_computed_tokens -= num_rejected
            if request.num_output_placeholders > 0:
                request.num_output_placeholders -= num_rejected
            spec_stats_accum["num_draft"] += num_draft_tokens
            spec_stats_accum["num_accepted"] += num_accepted
            spec_stats_accum["num_invalid"] += num_invalid_spec_tokens

        stopped = False
        new_logprobs = None
        new_token_ids = generated_token_ids
        pooler_output = None
        kv_transfer_params = None
        status_before = request.status

        # Check-stop approximation: length cap
        if new_token_ids:
            # inline simple stop check
            for tok in new_token_ids:
                request.append_output_token_ids(tok)
                if request.num_tokens >= max_model_len or (
                    request.num_tokens - request.num_prompt_tokens
                ) >= request.max_tokens:
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                    stopped = True
                    break
                if tok == request.sampling_params.eos_token_id:
                    request.status = RequestStatus.FINISHED_STOPPED
                    stopped = True
                    break

        finish_reason = None
        if stopped:
            finish_reason = "length" if request.status == RequestStatus.FINISHED_LENGTH_CAPPED else "stop"
            if status_before == RequestStatus.RUNNING:
                stopped_running.add(request)

        # Logprobs slice (skipped for simplicity — zero-cost if logprobs disabled)

        if new_token_ids or stopped:
            outputs.setdefault(request.client_index, []).append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=new_token_ids,
                    finish_reason=finish_reason,
                    new_logprobs=new_logprobs,
                    new_prompt_logprobs_tensors=None,
                    pooling_output=pooler_output,
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    kv_transfer_params=kv_transfer_params,
                    trace_headers=request.trace_headers,
                    num_cached_tokens=request.num_cached_tokens,
                    num_external_computed_tokens=request.num_external_computed_tokens,
                    routed_experts=None,
                    num_nans_in_logits=request.num_nans_in_logits,
                )
            )
    return outputs, stopped_running, spec_stats_accum


# ---------- Rust-assisted variant (prefix done in Rust, actions in Python) ----------

def rs_assisted_update_loop(requests, num_scheduled_tokens, scheduled_spec,
                             req_id_to_index, sampled_token_ids,
                             failed_kv_load_req_ids, max_model_len=4096):
    """Rust preamble does: early-skip (failed/finished), req_index lookup,
    spec-decode accepted/rejected counters + mutations, and returns a flat
    list of (req_id, req_index, generated_token_ids, spec_accepted_count)
    for the remaining work. Python loop handles stop check + output
    construction — the bits that need Python dataclasses / method calls.
    """
    prelim = vllm_rs.scheduler_update_preamble(
        num_scheduled_tokens,
        failed_kv_load_req_ids,
        requests,
        scheduled_spec,
        sampled_token_ids,
        req_id_to_index,
    )
    # prelim is a list of tuples: (req_id, req_index, generated_token_ids_py_ref, accepted, rejected)
    outputs: dict[int, list[EngineCoreOutput]] = {}
    stopped_running: set[Request] = set()
    for req_id, req_index, generated_token_ids, accepted, rejected in prelim:
        request = requests[req_id]
        # Preamble already mutated num_computed_tokens/num_output_placeholders when
        # spec rejections happened, using attribute access from Rust — we rely on that.

        stopped = False
        new_token_ids = generated_token_ids
        pooler_output = None
        kv_transfer_params = None
        status_before = request.status

        if new_token_ids:
            for tok in new_token_ids:
                request.append_output_token_ids(tok)
                if request.num_tokens >= max_model_len or (
                    request.num_tokens - request.num_prompt_tokens
                ) >= request.max_tokens:
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                    stopped = True
                    break
                if tok == request.sampling_params.eos_token_id:
                    request.status = RequestStatus.FINISHED_STOPPED
                    stopped = True
                    break

        finish_reason = None
        if stopped:
            finish_reason = "length" if request.status == RequestStatus.FINISHED_LENGTH_CAPPED else "stop"
            if status_before == RequestStatus.RUNNING:
                stopped_running.add(request)

        if new_token_ids or stopped:
            outputs.setdefault(request.client_index, []).append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=new_token_ids,
                    finish_reason=finish_reason,
                    new_logprobs=None,
                    new_prompt_logprobs_tensors=None,
                    pooling_output=pooler_output,
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    kv_transfer_params=kv_transfer_params,
                    trace_headers=request.trace_headers,
                    num_cached_tokens=request.num_cached_tokens,
                    num_external_computed_tokens=request.num_external_computed_tokens,
                    routed_experts=None,
                    num_nans_in_logits=request.num_nans_in_logits,
                )
            )
    return outputs, stopped_running, None


def run_bench(num_reqs: int, spec_frac: float, iters: int):
    # Build one workload and copy-reset it per iteration so each call sees
    # the same state.
    requests0, nst0, spec0, r2i0, sti0 = make_workload(num_reqs, spec_frac)

    def reset():
        # Cheap deep-copy of the requests dict (only resets mutable fields)
        requests = {
            rid: Request(
                request_id=rid,
                num_computed_tokens=r.num_computed_tokens,
                num_output_placeholders=r.num_output_placeholders,
                num_prompt_tokens=r.num_prompt_tokens,
                num_tokens=r.num_tokens,
                max_tokens=r.max_tokens,
                sampling_params=r.sampling_params,
            )
            for rid, r in requests0.items()
        }
        return requests

    def time_it(fn, kind: str):
        # Warmup
        for _ in range(3):
            fn()
        samples = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                fn()
            samples.append((time.perf_counter_ns() - t0) / iters)
        return statistics.median(samples) / 1e3, kind

    def py_call():
        reqs = reset()
        py_update_loop(reqs, nst0, spec0, r2i0, sti0, None)

    def rs_call():
        reqs = reset()
        rs_assisted_update_loop(reqs, nst0, spec0, r2i0, sti0, None)

    py_us, _ = time_it(py_call, "py")
    rs_us, _ = time_it(rs_call, "rs")
    return py_us, rs_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-reqs", type=int, default=256)
    ap.add_argument("--spec-frac", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    print(f"{'num_reqs':>8} {'spec_frac':>10} {'py (us/loop)':>14} {'rs (us/loop)':>14} {'speedup':>10}")
    for n in (32, 64, 128, 256, 512):
        py_us, rs_us = run_bench(n, args.spec_frac, args.iters)
        print(f"{n:>8} {args.spec_frac:>10.2f} {py_us:>14.2f} {rs_us:>14.2f} {py_us/rs_us:>9.2f}x")


if __name__ == "__main__":
    main()
