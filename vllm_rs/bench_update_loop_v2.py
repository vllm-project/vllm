"""Microbench v2: scheduler.update_from_output loop with Request as a Rust pyclass.

Three variants compared:
  py:     pure Python Request + pure Python loop body
  rs-py:  Rust Request pyclass + pure Python loop body (tests whether
          swapping only the data class hurts the Python code)
  rs-rs:  Rust Request pyclass + Rust loop body (the full port)

Run: python vllm_rs/bench_update_loop_v2.py
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vllm_rs


MAX_MODEL_LEN = 4096


class PyRequestStatus(Enum):
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
    extra_args: dict | None = None
    skip_reading_prefix_cache: bool | None = None


@dataclass(slots=True)
class PyRequest:
    """Reference Python Request (replaces the full vllm Request for the
    purposes of this microbench; exposes the attributes the hot loop reads)."""

    request_id: str
    status: PyRequestStatus = PyRequestStatus.RUNNING
    num_computed_tokens: int = 100
    num_output_placeholders: int = 0
    num_prompt_tokens: int = 100
    max_tokens: int = 512
    num_cached_tokens: int = 0
    num_external_computed_tokens: int = 0
    num_nans_in_logits: int = 0
    client_index: int = 0
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    stop_reason: object | None = None
    output_token_ids: list[int] = field(default_factory=list)
    all_token_ids: list[int] = field(default_factory=list)
    _events: list = field(default_factory=list)

    def is_finished(self) -> bool:
        return self.status in (
            PyRequestStatus.FINISHED_STOPPED,
            PyRequestStatus.FINISHED_LENGTH_CAPPED,
        )

    def append_output_token_ids(self, tok: int):
        self.output_token_ids.append(tok)
        self.all_token_ids.append(tok)

    @property
    def num_tokens(self):
        return len(self.all_token_ids)

    @property
    def num_output_tokens(self):
        return len(self.output_token_ids)

    def take_events(self):
        if not self._events:
            return None
        evs = self._events
        self._events = []
        return evs


def build_py_requests(num_reqs: int, spec_frac: float, seed: int = 0):
    rng = random.Random(seed)
    reqs = {}
    nst = {}
    spec = {}
    r2i = {}
    sti = []
    for i in range(num_reqs):
        rid = f"req-{i}"
        reqs[rid] = PyRequest(
            request_id=rid,
            num_computed_tokens=rng.randint(50, 200),
            num_prompt_tokens=rng.randint(50, 200),
            max_tokens=rng.randint(256, 1024),
            sampling_params=SamplingParams(eos_token_id=2, stop_token_ids=(5, 7)),
        )
        nst[rid] = 1
        r2i[rid] = i
        if rng.random() < spec_frac:
            spec[rid] = [rng.randint(100, 999) for _ in range(3)]
            accepted = rng.randint(1, 3)
            sti.append([rng.randint(100, 999) for _ in range(accepted + 1)])
        else:
            sti.append([rng.randint(100, 999)])
    return reqs, nst, spec, r2i, sti


def build_rs_requests(num_reqs: int, spec_frac: float, seed: int = 0):
    """Same workload, but each Request is a vllm_rs.Request pyclass."""
    rng = random.Random(seed)
    reqs = {}
    nst = {}
    spec = {}
    r2i = {}
    sti = []
    for i in range(num_reqs):
        rid = f"req-{i}"
        sp = SamplingParams(eos_token_id=2, stop_token_ids=(5, 7), max_tokens=rng.randint(256, 1024))
        r = vllm_rs.Request(
            request_id=rid,
            prompt_token_ids=list(range(rng.randint(50, 200))),
            sampling_params=sp,
            pooling_params=None,
        )
        r.status = vllm_rs.RequestStatus.RUNNING
        r.num_computed_tokens = rng.randint(50, 200)
        r.num_prompt_tokens = rng.randint(50, 200)
        reqs[rid] = r
        nst[rid] = 1
        r2i[rid] = i
        if rng.random() < spec_frac:
            spec[rid] = [rng.randint(100, 999) for _ in range(3)]
            accepted = rng.randint(1, 3)
            sti.append([rng.randint(100, 999) for _ in range(accepted + 1)])
        else:
            sti.append([rng.randint(100, 999)])
    return reqs, nst, spec, r2i, sti


def py_update_loop(reqs, nst, spec, r2i, sti, failed=None):
    outputs = {}
    stopped_running = set()
    for req_id, _ in nst.items():
        if failed and req_id in failed:
            continue
        r = reqs.get(req_id)
        if r is None or r.is_finished():
            continue
        idx = r2i[req_id]
        generated = sti[idx] if sti else []
        scheduled_spec = spec.get(req_id)
        if scheduled_spec and generated:
            nd = len(scheduled_spec)
            na = len(generated) - 1
            nr = nd - na
            if r.num_computed_tokens > 0:
                r.num_computed_tokens -= nr
            if r.num_output_placeholders > 0:
                r.num_output_placeholders -= nr
        stopped = False
        status_before = r.status
        if generated:
            for tok in generated:
                r.append_output_token_ids(tok)
                if r.num_tokens >= MAX_MODEL_LEN or r.num_output_tokens >= r.max_tokens:
                    # Python ref uses enum, rust uses its own — keep uniform
                    if isinstance(r, PyRequest):
                        r.status = PyRequestStatus.FINISHED_LENGTH_CAPPED
                    else:
                        r.status = vllm_rs.RequestStatus.FINISHED_LENGTH_CAPPED
                    stopped = True
                    break
                if tok == r.sampling_params.eos_token_id:
                    if isinstance(r, PyRequest):
                        r.status = PyRequestStatus.FINISHED_STOPPED
                    else:
                        r.status = vllm_rs.RequestStatus.FINISHED_STOPPED
                    stopped = True
                    break
        finish_reason = None
        if stopped:
            if isinstance(r, PyRequest):
                finish_reason = "length" if r.status == PyRequestStatus.FINISHED_LENGTH_CAPPED else "stop"
                was_running = status_before == PyRequestStatus.RUNNING
            else:
                finish_reason = "length" if r.status == vllm_rs.RequestStatus.FINISHED_LENGTH_CAPPED else "stop"
                was_running = status_before == vllm_rs.RequestStatus.RUNNING
            if was_running:
                stopped_running.add(req_id)
        if generated or stopped:
            outputs.setdefault(r.client_index, []).append(
                (req_id, generated, finish_reason, r.stop_reason, r.num_cached_tokens,
                 r.num_external_computed_tokens, r.num_nans_in_logits, r.take_events())
            )
    return outputs, stopped_running


def rs_update_loop(reqs, nst, spec, r2i, sti, failed=None):
    rows = vllm_rs.scheduler_update_loop_rs_request(
        nst, failed, reqs, spec, sti, r2i, MAX_MODEL_LEN,
    )
    # Post-process: build outputs dict keyed by client_index (matching py ref).
    outputs = {}
    stopped_running = set()
    for row in rows:
        (req_id, new_tokens, finish_reason, client_index, events,
         stop_reason, num_cached, num_ext, num_nans, stopped, was_running) = row
        outputs.setdefault(client_index, []).append(
            (req_id, new_tokens, finish_reason, stop_reason, num_cached,
             num_ext, num_nans, events)
        )
        if stopped and was_running:
            stopped_running.add(req_id)
    return outputs, stopped_running


def time_it(fn, iters: int):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter_ns() - t0) / iters)
    return statistics.median(samples) / 1e3


def reset_py(orig_reqs, orig_sti):
    # Build fresh Py requests with same state as orig.
    out = {}
    for rid, r in orig_reqs.items():
        out[rid] = PyRequest(
            request_id=rid,
            num_computed_tokens=r.num_computed_tokens,
            num_prompt_tokens=r.num_prompt_tokens,
            max_tokens=r.max_tokens,
            sampling_params=r.sampling_params,
        )
    return out, [list(x) for x in orig_sti]


def reset_rs(orig_reqs, orig_sti):
    out = {}
    for rid, r in orig_reqs.items():
        sp = SamplingParams(eos_token_id=2, stop_token_ids=(5, 7), max_tokens=r.max_tokens)
        new = vllm_rs.Request(
            request_id=rid, prompt_token_ids=[0] * r.num_prompt_tokens,
            sampling_params=sp, pooling_params=None,
        )
        new.status = vllm_rs.RequestStatus.RUNNING
        new.num_computed_tokens = r.num_computed_tokens
        out[rid] = new
    return out, [list(x) for x in orig_sti]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--spec-frac", type=float, default=0.5)
    args = ap.parse_args()

    print(f"{'n_reqs':>8} {'py (us)':>10} {'rs-py (us)':>12} {'rs-rs (us)':>12} {'py/rs-rs':>10}")
    for n in (32, 64, 128, 256, 512):
        # Build canonical workload once. Bump caps so requests never stop —
        # each call is idempotent-ish (just appends 1 more token per req).
        py_reqs0, nst, spec, r2i, sti = build_py_requests(n, args.spec_frac)
        rs_reqs0, nst2, spec2, r2i2, sti2 = build_rs_requests(n, args.spec_frac)
        # Disable stop conditions so the loop is idempotent
        for r in py_reqs0.values():
            r.max_tokens = 10_000_000
            r.sampling_params = SamplingParams(eos_token_id=-1, stop_token_ids=())
        # For rust: rebuild with huge max + non-matching eos
        for rid, r in rs_reqs0.items():
            r.max_tokens = 10_000_000
            r.sampling_params = SamplingParams(eos_token_id=-1, stop_token_ids=())
        # Also replace the sampled-token-ids with non-eos tokens (they already are)
        # so nothing matches eos=-1.

        # Variant 1: pure Python  — no reset; in-place idempotent calls
        def v_py():
            py_update_loop(py_reqs0, nst, spec, r2i, sti)
        # Variant 2: Rust Request + Python loop
        def v_rs_py():
            py_update_loop(rs_reqs0, nst2, spec2, r2i2, sti2)
        # Variant 3: Rust Request + Rust loop
        def v_rs_rs():
            rs_update_loop(rs_reqs0, nst2, spec2, r2i2, sti2)

        t_py = time_it(v_py, args.iters)
        t_rs_py = time_it(v_rs_py, args.iters)
        t_rs_rs = time_it(v_rs_rs, args.iters)
        print(f"{n:>8} {t_py:>10.2f} {t_rs_py:>12.2f} {t_rs_rs:>12.2f} {t_py/t_rs_rs:>9.2f}x")


if __name__ == "__main__":
    main()
