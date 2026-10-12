# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the simulated-execution timing providers (no engine)."""

import json
import time

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.simulated.timing import (
    BUILTIN_TIMING_PROVIDERS,
    ConstantTimingProvider,
    RealClock,
    ReplayTimingProvider,
    SimRequestStep,
    SimulatedAsyncOutput,
    StepCost,
    StepDescriptor,
    StepLogWriter,
    TableTimingProvider,
    TimingProvider,
    ZeroTimingProvider,
    get_timing_provider,
)


def _req(req_id: str, scheduled: int, computed: int, prompt: int) -> SimRequestStep:
    return SimRequestStep(
        req_id=req_id,
        num_scheduled_tokens=scheduled,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_cached_tokens=0,
        num_spec_tokens=0,
        is_prefill=computed + scheduled <= prompt,
    )


def _step(index: int, *reqs: SimRequestStep) -> StepDescriptor:
    prefill = [r for r in reqs if r.is_prefill]
    decode = [r for r in reqs if not r.is_prefill]
    return StepDescriptor(
        step_index=index,
        requests=tuple(reqs),
        num_prefill_tokens=sum(r.num_scheduled_tokens for r in prefill),
        num_decode_tokens=sum(r.num_scheduled_tokens for r in decode),
        num_prefill_reqs=len(prefill),
        num_decode_reqs=len(decode),
        num_common_prefix_blocks=0,
        num_kv_tokens=sum(r.num_computed_tokens + r.num_scheduled_tokens for r in reqs),
    )


class _Cfg:
    """Providers only receive the config object; built-ins ignore it."""


def test_builtin_registry_and_protocol():
    assert set(BUILTIN_TIMING_PROVIDERS) == {"zero", "constant", "table", "replay"}
    for name in ("zero", "constant"):
        provider = get_timing_provider(name, {}, _Cfg())  # type: ignore[arg-type]
        assert isinstance(provider, TimingProvider)
    with pytest.raises(ValueError, match="Unknown timing provider"):
        get_timing_provider("nope", {}, _Cfg())  # type: ignore[arg-type]


def test_zero_provider_rejects_args():
    with pytest.raises(ValueError):
        ZeroTimingProvider(_Cfg(), foo=1)  # type: ignore[arg-type]
    assert ZeroTimingProvider(_Cfg()).estimate(_step(0)).device_s == 0.0  # type: ignore[arg-type]


def test_constant_provider_linear_terms():
    p = ConstantTimingProvider(
        _Cfg(),  # type: ignore[arg-type]
        fixed_us=1000,
        prefill_us_per_token=10,
        decode_us_per_req=50,
        decode_us_per_kv_token=0.1,
        host_us=200,
    )
    # one prefill chunk of 100 tokens + two decodes with 1000 and 2000 KV tokens
    step = _step(
        0,
        _req("p", 100, 0, 300),
        _req("d1", 1, 999, 10),
        _req("d2", 1, 1999, 10),
    )
    cost = p.estimate(step)
    expected_us = 1000 + 10 * 100 + 50 * 2 + 0.1 * (1000 + 2000)
    assert cost.device_s == pytest.approx(expected_us * 1e-6)
    assert cost.host_s == pytest.approx(200e-6)
    assert cost.total_s == pytest.approx((expected_us + 200) * 1e-6)
    assert cost.breakdown == {"prefill_us": 1000.0, "decode_us": 400.0}


def test_replay_exact_match_then_knn_fallback(tmp_path):
    log = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(log))
    steps = [
        _step(0, _req("a", 128, 0, 128)),
        _step(1, _req("a", 1, 128, 128)),
        _step(2, _req("a", 1, 129, 128), _req("b", 1, 7, 7)),
    ]
    costs = [
        StepCost(device_s=0.010),
        StepCost(device_s=0.002),
        StepCost(device_s=0.003),
    ]
    for s, c in zip(steps, costs):
        writer.write(s, c, t_start=0.0, t_deadline=c.device_s, finished_req_ids=[])
    writer.close()
    assert len(log.read_text().splitlines()) == 3

    replay = ReplayTimingProvider(_Cfg(), path=str(log))  # type: ignore[arg-type]
    # Same batch composition under different request ids still matches.
    same_shape = _step(9, _req("x", 1, 7, 7), _req("y", 1, 129, 128))
    assert replay.estimate(same_shape).device_s == pytest.approx(0.003)
    assert replay.estimate(steps[0]).device_s == pytest.approx(0.010)
    # Unseen composition falls back to the nearest neighbours.
    miss = replay.estimate(_step(10, _req("z", 1, 128, 128), _req("w", 1, 128, 128)))
    assert 0.002 <= miss.device_s <= 0.010
    m = replay.metrics()
    assert m["replay_exact_match_steps"] == 2
    assert m["replay_miss_steps"] == 1

    zero_fallback = ReplayTimingProvider(
        _Cfg(),  # type: ignore[arg-type]
        path=str(log),
        fallback="zero",
        miss_fallback_s=0.5,
    )
    assert zero_fallback.estimate(same_shape).device_s == pytest.approx(0.003)
    assert zero_fallback.estimate(_step(11, _req("q", 64, 0, 64))).device_s == 0.5


def test_table_provider_accepts_hand_written_records(tmp_path):
    table = tmp_path / "table.json"
    table.write_text(
        json.dumps(
            [
                {"num_reqs": 1, "num_tokens": 1, "sum_past_kv": 100, "device_ms": 2.0},
                {"num_reqs": 8, "num_tokens": 8, "sum_past_kv": 800, "device_ms": 3.0},
                {"num_reqs": 1, "num_tokens": 512, "sum_past_kv": 0, "device_ms": 20.0},
            ]
        )
    )
    p = TableTimingProvider(_Cfg(), path=str(table), k=1)  # type: ignore[arg-type]
    decode1 = _step(0, _req("a", 1, 100, 50))
    assert p.estimate(decode1).device_s == pytest.approx(2e-3)
    prefill = _step(1, _req("b", 512, 0, 512))
    assert p.estimate(prefill).device_s == pytest.approx(20e-3)


def test_simulated_async_output_waits_until_deadline():
    clock = RealClock()
    out = ModelRunnerOutput(
        req_ids=["r"], req_id_to_index={"r": 0}, sampled_token_ids=[[1]]
    )
    deadline = clock.now() + 0.05
    async_out = SimulatedAsyncOutput(out, deadline, clock)
    t0 = time.monotonic()
    assert async_out.get_output() is out
    assert time.monotonic() - t0 >= 0.045


def test_replay_key_is_order_independent():
    a = _step(0, _req("a", 1, 10, 5), _req("b", 4, 20, 30))
    b = _step(1, _req("b", 4, 20, 30), _req("a", 1, 10, 5))
    assert a.replay_key() == b.replay_key()


def test_replay_critical_path_uses_host_time_when_host_bound(tmp_path):
    log = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(log))
    prefill = _step(0, _req("a", 512, 0, 512))
    decode = _step(1, _req("a", 1, 512, 512))
    # Host-bound prefill, device-bound decode.
    writer.write(prefill, StepCost(device_s=0.006, host_s=0.008), 0.0, 0.006, [])
    writer.write(decode, StepCost(device_s=0.002, host_s=0.001), 0.0, 0.002, [])
    writer.close()
    by_max = ReplayTimingProvider(_Cfg(), path=str(log))  # type: ignore[arg-type]
    assert by_max.estimate(prefill).device_s == pytest.approx(0.008)
    assert by_max.estimate(decode).device_s == pytest.approx(0.002)
    by_device = ReplayTimingProvider(
        _Cfg(),  # type: ignore[arg-type]
        path=str(log),
        critical_path="device",
    )
    assert by_device.estimate(prefill).device_s == pytest.approx(0.006)
