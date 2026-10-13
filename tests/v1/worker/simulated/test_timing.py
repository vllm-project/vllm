# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the simulated-execution timing providers (no engine)."""

import json

import pytest

from tests.v1.worker.simulated.utils import StubConfig, sim_req, sim_step
from vllm.v1.worker.simulated.timing import (
    HostTimeModel,
    LinearTimingProvider,
    ReplayTimingProvider,
    SimRequestStep,
    StepCost,
    StepLogWriter,
    TableTimingProvider,
    acceptance_rates_from_log,
    get_timing_provider,
    knn_lookup_counts,
)


def test_unknown_timing_provider():
    with pytest.raises(ValueError, match="Unknown timing provider"):
        get_timing_provider("nope", {}, StubConfig())  # type: ignore[arg-type]


def test_linear_provider_terms():
    p = LinearTimingProvider(
        StubConfig(),  # type: ignore[arg-type]
        fixed_us=1000,
        prefill_us_per_token=10,
        decode_us_per_req=50,
        decode_us_per_kv_token=0.1,
        host_us=200,
    )
    # one prefill chunk of 100 tokens + two decodes with 1000 and 2000 KV tokens
    step = sim_step(
        0,
        sim_req("p", 100, 0, 300),
        sim_req("d1", 1, 999, 10),
        sim_req("d2", 1, 1999, 10),
    )
    cost = p.estimate(step)
    expected_us = 1000 + 10 * 100 + 50 * 2 + 0.1 * (1000 + 2000)
    assert cost.device_s == pytest.approx(expected_us * 1e-6)
    assert cost.host_s == pytest.approx(200e-6)
    assert cost.breakdown == {"prefill_us": 1000.0, "decode_us": 400.0}


def test_replay_exact_match_then_knn_fallback(tmp_path):
    log = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(log))
    steps = [
        sim_step(0, sim_req("a", 128, 0, 128)),
        sim_step(1, sim_req("a", 1, 128, 128)),
        sim_step(2, sim_req("a", 1, 129, 128), sim_req("b", 1, 7, 7)),
        sim_step(3, sim_req("c", 128, 0, 128)),
    ]
    costs = [
        StepCost(device_s=0.010),
        StepCost(device_s=0.002),
        StepCost(device_s=0.003),
        StepCost(device_s=0.014),
    ]
    for s, c in zip(steps, costs):
        writer.write(s, c, t_start=0.0, t_deadline=c.device_s, finished_req_ids=[])
    writer.close()
    assert len(log.read_text().splitlines()) == 4

    replay = ReplayTimingProvider(StubConfig(), path=str(log))  # type: ignore[arg-type]
    # Same batch composition under different request ids still matches.
    same_shape = sim_step(9, sim_req("x", 1, 7, 7), sim_req("y", 1, 129, 128))
    assert replay.estimate(same_shape).device_s == pytest.approx(0.003)
    # A composition recorded twice replays its mean.
    assert replay.estimate(steps[0]).device_s == pytest.approx(0.012)
    # Unseen composition falls back to the nearest neighbours.
    miss = replay.estimate(
        sim_step(10, sim_req("z", 1, 128, 128), sim_req("w", 1, 128, 128))
    )
    assert 0.002 <= miss.device_s <= 0.014
    assert miss.breakdown == {"replay_exact": 0.0}

    zero_fallback = ReplayTimingProvider(
        StubConfig(),  # type: ignore[arg-type]
        path=str(log),
        fallback="zero",
        miss_fallback_s=0.5,
    )
    assert zero_fallback.estimate(same_shape).device_s == pytest.approx(0.003)
    assert zero_fallback.estimate(sim_step(11, sim_req("q", 64, 0, 64))).device_s == 0.5


def _write_table(path, rows) -> None:
    """A hand-written table: (num_reqs, num_tokens, sum_past_kv, device_s)."""
    keys = ("num_reqs", "num_tokens", "sum_past_kv", "device_s")
    path.write_text("".join(json.dumps(dict(zip(keys, r))) + "\n" for r in rows))


def test_table_provider_accepts_hand_written_records(tmp_path):
    table = tmp_path / "table.jsonl"
    rows = [(1, 1, 100, 2e-3), (8, 8, 800, 3e-3), (1, 512, 0, 20e-3)]
    _write_table(table, rows)
    p = TableTimingProvider(StubConfig(), path=str(table), k=1)  # type: ignore[arg-type]
    decode1 = sim_step(0, sim_req("a", 1, 100, 50))
    assert p.estimate(decode1).device_s == pytest.approx(2e-3)
    prefill = sim_step(1, sim_req("b", 512, 0, 512))
    assert p.estimate(prefill).device_s == pytest.approx(20e-3)


def test_table_counts_lookups_beyond_recorded_range(tmp_path):
    table = tmp_path / "table.jsonl"
    _write_table(table, [(1, 1, 100, 2e-3), (4, 4, 400, 3e-3)])
    p = TableTimingProvider(StubConfig(), path=str(table), k=1)  # type: ignore[arg-type]
    before = knn_lookup_counts()
    p.estimate(sim_step(0, *(sim_req(str(i), 1, 100, 50) for i in range(2))))
    # A batch larger than any recorded gets the largest recorded step's time.
    big = p.estimate(sim_step(1, *(sim_req(str(i), 1, 100, 50) for i in range(16))))
    assert big.device_s == pytest.approx(3e-3)
    lookups, outside = (a - b for a, b in zip(knn_lookup_counts(), before))
    assert (lookups, outside) == (2, 1)


def test_replay_and_table_return_recorded_host_time(tmp_path):
    log = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(log))
    prefill = sim_step(0, sim_req("a", 512, 0, 512))
    decode = sim_step(1, sim_req("a", 1, 512, 512))
    # Host-bound prefill, device-bound decode.
    writer.write(prefill, StepCost(device_s=0.006, host_s=0.008), 0.0, 0.008, [])
    writer.write(decode, StepCost(device_s=0.002, host_s=0.001), 0.0, 0.002, [])
    writer.close()
    replay = ReplayTimingProvider(StubConfig(), path=str(log))  # type: ignore[arg-type]
    cost = replay.estimate(prefill)
    assert (cost.device_s, cost.host_s) == pytest.approx((0.006, 0.008))
    table = TableTimingProvider(StubConfig(), path=str(log), k=1)  # type: ignore[arg-type]
    cost = table.estimate(sim_step(2, sim_req("b", 1, 500, 500)))
    assert (cost.device_s, cost.host_s) == pytest.approx((0.002, 0.001))


def test_build_step_descriptor_from_scheduler_output():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.worker.simulated.timing import (
        ParallelSummary,
        build_step_descriptor,
    )

    prompt_lens: dict[str, int] = {}
    parallel = ParallelSummary()

    # Step 0: a new 300-token prompt with 96 cached tokens (chunk of 100) and
    # a new 8-token prompt in one shot.
    so = SchedulerOutput.make_empty()
    so.scheduled_new_reqs = [
        NewRequestData(
            req_id="a",
            prompt_token_ids=list(range(300)),
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=4),
            pooling_params=None,
            block_ids=([0, 1, 2],),
            num_computed_tokens=96,
            lora_request=None,
            prefill_token_ids=list(range(300)),
        ),
        NewRequestData(
            req_id="b",
            prompt_token_ids=list(range(8)),
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=4),
            pooling_params=None,
            block_ids=([3],),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=list(range(8)),
        ),
    ]
    so.num_scheduled_tokens = {"a": 100, "b": 8}
    so.total_num_scheduled_tokens = 108
    so.num_common_prefix_blocks = [0]
    d0 = build_step_descriptor(so, prompt_lens, 0, parallel)
    assert [r.req_id for r in d0.requests] == ["a", "b"]
    assert d0.num_prefill_tokens == 108 and d0.num_prefill_reqs == 2
    assert d0.num_decode_reqs == 0
    a, b = d0.requests
    assert (a.num_computed_tokens, a.num_cached_tokens, a.is_prefill) == (96, 96, True)
    assert b.num_prompt_tokens == 8 and b.is_prefill
    assert prompt_lens == {"a": 300, "b": 8}

    # Step 1: "a" continues its prefill (104 more), "b" decodes one token.
    so1 = SchedulerOutput.make_empty()
    so1.scheduled_cached_reqs = CachedRequestData(
        req_ids=["a", "b"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None, None],
        num_computed_tokens=[196, 8],
        num_output_tokens=[0, 1],
    )
    so1.num_scheduled_tokens = {"a": 104, "b": 1}
    so1.total_num_scheduled_tokens = 105
    d1 = build_step_descriptor(so1, prompt_lens, 1, parallel)
    a, b = d1.requests
    assert a.is_prefill and a.num_computed_tokens == 196 and a.num_cached_tokens == 0
    assert not b.is_prefill and b.num_scheduled_tokens == 1
    assert d1.num_prefill_tokens == 104 and d1.num_decode_tokens == 1
    assert d1.num_kv_tokens == (196 + 104) + (8 + 1)

    # Step 2: "b" finished; the builder forgets it.
    so2 = SchedulerOutput.make_empty()
    so2.finished_req_ids = {"b"}
    so2.scheduled_cached_reqs = CachedRequestData(
        req_ids=["a"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[300],
        num_output_tokens=[1],
    )
    so2.num_scheduled_tokens = {"a": 1}
    so2.total_num_scheduled_tokens = 1
    d2 = build_step_descriptor(so2, prompt_lens, 2, parallel)
    assert "b" not in prompt_lens
    assert d2.is_decode_only and d2.requests[0].num_computed_tokens == 300


def test_acceptance_rates_from_log(tmp_path):
    def spec_req(req_id: str, drafts: int) -> SimRequestStep:
        return SimRequestStep(req_id, 1 + drafts, 100, 50, 0, drafts, False)

    log = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(log))
    # Four requests with 3 drafts accept 3, 2, 0, 1; one with 1 draft accepts 1.
    step = sim_step(0, *(spec_req(r, 3) for r in "abcd"), spec_req("e", 1))
    writer.write(step, StepCost(device_s=0.01, num_accepted=(3, 2, 0, 1, 1)), 0, 0, [])
    writer.write(
        sim_step(1, sim_req("f", 1, 100, 50)), StepCost(device_s=0.01), 0, 0, []
    )
    writer.close()
    # Position 1: 4 of 5 accepted. Position 2: 2 of the 3 that reached it.
    # Position 3: 1 of the 2 that reached it.
    assert acceptance_rates_from_log(str(log)) == pytest.approx([4 / 5, 2 / 3, 1 / 2])


def test_pipeline_stages_overlap_consecutive_steps():
    from types import SimpleNamespace

    from vllm.v1.worker.simulated.model_runner import SimulatedModelRunner

    runner = SimpleNamespace(_stage_free_at=[0.0, 0.0])
    run = SimulatedModelRunner._run_pipeline
    # 4 ms through two balanced stages: 2 ms each.
    assert run(runner, 0.0, 4e-3) == pytest.approx(4e-3)
    # A second step enters stage 0 as the first moves to stage 1.
    assert run(runner, 0.0, 4e-3) == pytest.approx(6e-3)
    assert runner._stage_free_at == pytest.approx([4e-3, 6e-3])


def _host_log(path, host_us) -> str:
    """A step log of prefill and decode steps whose host time is
    `host_us(num_reqs, prefill_tokens)`."""
    writer = StepLogWriter(str(path))
    for i, (num_decode, prefill) in enumerate([(1, 0), (4, 0), (2, 512), (8, 128)]):
        reqs = [sim_req(f"d{j}", 1, 100, 50) for j in range(num_decode)]
        if prefill:
            reqs.append(sim_req("p", prefill, 0, prefill))
        host_s = host_us(len(reqs), prefill) * 1e-6
        writer.write(sim_step(i, *reqs), StepCost(1e-3, host_s), 0.0, 1e-3, [])
    writer.close()
    return str(path)


def test_host_time_model_fitted_from_log(tmp_path):
    path = _host_log(
        tmp_path / "a.jsonl", lambda reqs, prefill: 1000 + 30 * reqs + 2 * prefill
    )
    model = HostTimeModel.from_log(path)
    assert model.fixed_us == pytest.approx(1000)
    assert model.per_req_us == pytest.approx(30)
    assert model.per_prefill_token_us == pytest.approx(2)

    provider = LinearTimingProvider(StubConfig(), fixed_us=500, host_fit_log=path)
    step = sim_step(0, sim_req("a", 1, 10, 5), sim_req("b", 64, 0, 64))
    assert provider.estimate(step).host_s == pytest.approx((1000 + 60 + 128) * 1e-6)

    # A term that would fit negative is dropped and the rest refitted.
    path = _host_log(
        tmp_path / "b.jsonl", lambda reqs, prefill: 1000 - 10 * reqs + 2 * prefill
    )
    model = HostTimeModel.from_log(path)
    assert model.per_req_us == 0.0
    assert model.fixed_us > 0 and model.per_prefill_token_us > 0


def test_host_time_model_leaves_undetermined_terms_at_zero(tmp_path):
    # Every step schedules 4 requests: per-request time is indistinguishable
    # from fixed time and stays in the fixed term.
    path = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(path))
    for i, prefill in enumerate([0, 0, 256, 64]):
        reqs = [sim_req(f"d{j}", 1, 100, 50) for j in range(4 - bool(prefill))]
        if prefill:
            reqs.append(sim_req("p", prefill, 0, prefill))
        host_s = (1100 + 15 * prefill) * 1e-6
        writer.write(sim_step(i, *reqs), StepCost(1e-3, host_s), 0.0, 1e-3, [])
    writer.close()
    model = HostTimeModel.from_log(str(path))
    assert model == HostTimeModel(pytest.approx(1100), 0.0, pytest.approx(15))
