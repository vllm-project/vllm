# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the simulated-execution timing providers (no engine)."""

import json
import time
from types import SimpleNamespace

import pytest

from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.simulated.kv_transfer import (
    SimulatedOffloadingWorker,
    TransferRate,
    transfer_rates_from_log,
)
from vllm.v1.worker.simulated.timing import (
    BUILTIN_TIMING_PROVIDERS,
    ConstantTimingProvider,
    FastForwardClock,
    HostTimeModel,
    RealClock,
    ReplayTimingProvider,
    SimRequestStep,
    SimulatedAsyncOutput,
    StepCost,
    StepDescriptor,
    StepLogWriter,
    TableTimingProvider,
    TimingProvider,
    VirtualClock,
    ZeroTimingProvider,
    acceptance_rates_from_log,
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


def test_replay_and_table_return_recorded_host_time(tmp_path):
    log = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(log))
    prefill = _step(0, _req("a", 512, 0, 512))
    decode = _step(1, _req("a", 1, 512, 512))
    # Host-bound prefill, device-bound decode.
    writer.write(prefill, StepCost(device_s=0.006, host_s=0.008), 0.0, 0.008, [])
    writer.write(decode, StepCost(device_s=0.002, host_s=0.001), 0.0, 0.002, [])
    writer.close()
    replay = ReplayTimingProvider(_Cfg(), path=str(log))  # type: ignore[arg-type]
    cost = replay.estimate(prefill)
    assert (cost.device_s, cost.host_s) == pytest.approx((0.006, 0.008))
    table = TableTimingProvider(_Cfg(), path=str(log), k=1)  # type: ignore[arg-type]
    cost = table.estimate(_step(2, _req("b", 1, 500, 500)))
    assert (cost.device_s, cost.host_s) == pytest.approx((0.002, 0.001))


def test_clocks():
    virtual = VirtualClock()
    virtual.sleep_until(1.5)
    virtual.sleep_until(1.0)
    assert virtual.now() == 1.5

    ff = FastForwardClock()
    t0 = time.monotonic()
    target = ff.now() + 10.0
    ff.sleep_until(target)
    assert time.monotonic() - t0 < 1.0
    assert ff.now() >= target


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
    step = _step(0, *(spec_req(r, 3) for r in "abcd"), spec_req("e", 1))
    writer.write(step, StepCost(device_s=0.01, num_accepted=(3, 2, 0, 1, 1)), 0, 0, [])
    writer.write(_step(1, _req("f", 1, 100, 50)), StepCost(device_s=0.01), 0, 0, [])
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


def _gpu_spec(*group_sizes: int) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(
        list(range(sum(group_sizes))), group_sizes, [0] * len(group_sizes)
    )


def test_kv_transfers_queue_per_direction_behind_device_work():
    clock = VirtualClock()
    worker = SimulatedOffloadingWorker(
        clock,
        load=TransferRate(latency_s=0.0, bytes_per_s=100.0),
        store=TransferRate(latency_s=0.25, bytes_per_s=50.0),
        bytes_per_block=[10, 30],
        device_free_at=lambda: 0.5,
    )
    # 2 blocks of group 0 and 1 of group 1: 50 bytes.
    worker.submit_load(1, None, _gpu_spec(2, 1))  # 0.5 -> 1.0
    worker.submit_load(2, None, _gpu_spec(1, 0))  # 1.0 -> 1.1
    worker.submit_store(3, _gpu_spec(5, 0), None)  # 0.5 -> 1.75
    assert worker.next_completion() == pytest.approx(1.0)

    clock.sleep_until(1.05)
    [done] = worker.get_finished()
    assert (done.job_id, done.transfer_size) == (1, 50)
    assert done.transfer_time == pytest.approx(0.5)

    worker.wait({3})
    assert clock.now() == pytest.approx(1.75)
    assert sorted(r.job_id for r in worker.get_finished()) == [2, 3]
    assert worker.next_completion() is None


def test_kv_transfer_rates_fitted_from_log(tmp_path):
    path = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(path))
    writer.write(_step(0, _req("a", 1, 8, 8)), StepCost(device_s=1e-3), 0.0, 1e-3, [])
    load = TransferRate(latency_s=20e-6, bytes_per_s=40e9)
    for count, num_bytes in [(1, 1 << 20), (3, 3 << 22), (2, 5 << 24)]:
        seconds = count * load.latency_s + num_bytes / load.bytes_per_s
        writer.write_transfers(
            {"load": {"count": count, "bytes": num_bytes, "seconds": seconds}}, t=0.0
        )
    # Durations that fall with size: no latency fits, only the mean bandwidth.
    for num_bytes, seconds in [(1 << 20, 2e-3), (4 << 20, 1e-3)]:
        writer.write_transfers(
            {"store": {"count": 1, "bytes": num_bytes, "seconds": seconds}}, t=0.0
        )
    writer.close()

    fitted_load, fitted_store = transfer_rates_from_log(str(path))
    assert fitted_load.latency_s == pytest.approx(load.latency_s)
    assert fitted_load.bytes_per_s == pytest.approx(load.bytes_per_s)
    assert fitted_store.latency_s == 0.0
    assert fitted_store.bytes_per_s == pytest.approx((5 << 20) / 3e-3)
    # The step providers skip the transfer records.
    assert len(ReplayTimingProvider(_Cfg(), path=str(path)).exact) == 1


def _host_log(path, host_us) -> str:
    """A step log of prefill and decode steps whose host time is
    `host_us(num_reqs, prefill_tokens)`."""
    writer = StepLogWriter(str(path))
    for i, (num_decode, prefill) in enumerate([(1, 0), (4, 0), (2, 512), (8, 128)]):
        reqs = [_req(f"d{j}", 1, 100, 50) for j in range(num_decode)]
        if prefill:
            reqs.append(_req("p", prefill, 0, prefill))
        host_s = host_us(len(reqs), prefill) * 1e-6
        writer.write(_step(i, *reqs), StepCost(1e-3, host_s), 0.0, 1e-3, [])
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

    provider = ConstantTimingProvider(_Cfg(), fixed_us=500, host_fit_log=path)
    step = _step(0, _req("a", 1, 10, 5), _req("b", 64, 0, 64))
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
        reqs = [_req(f"d{j}", 1, 100, 50) for j in range(4 - bool(prefill))]
        if prefill:
            reqs.append(_req("p", prefill, 0, prefill))
        host_s = (1100 + 15 * prefill) * 1e-6
        writer.write(_step(i, *reqs), StepCost(1e-3, host_s), 0.0, 1e-3, [])
    writer.close()
    model = HostTimeModel.from_log(str(path))
    assert model == HostTimeModel(pytest.approx(1100), 0.0, pytest.approx(15))


def test_kv_transfer_rates_without_stores(tmp_path):
    from vllm.v1.worker.simulated.kv_transfer import kv_transfer_rates

    load, store = kv_transfer_rates({"load_gbps": 2.0, "load_latency_us": 5.0})
    assert (load.bytes_per_s, load.latency_s, store) == (2e9, pytest.approx(5e-6), None)
    # A disaggregated decode instance's recording has reads only.
    path = tmp_path / "steps.jsonl"
    writer = StepLogWriter(str(path))
    for num_bytes in (1 << 20, 4 << 20):
        writer.write_transfers(
            {"load": {"count": 1, "bytes": num_bytes, "seconds": num_bytes / 1e10}},
            t=0.0,
        )
    writer.close()
    load, store = transfer_rates_from_log(str(path))
    assert load.bytes_per_s == pytest.approx(1e10) and store is None


def _nixl_worker(engine_id: str, clock: VirtualClock):
    from vllm.v1.worker.simulated.kv_nixl import SimulatedNixlWorker

    group = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(page_size_bytes=10), layer_names=["layer"]
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[group], transfer_group_ids=(0,))
    worker = SimpleNamespace(kv_transfer_config=object())
    nixl = SimulatedNixlWorker(worker, engine_id, kv_cache_config)
    nixl.configure(clock, TransferRate(latency_s=0.0, bytes_per_s=10.0))
    return nixl


def test_simulated_nixl_reads_notify_the_prefill_instance():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
        RemoteMeta,
        ReqMeta,
    )

    clock = VirtualClock()
    prefill = _nixl_worker("sim-p", clock)
    decode = _nixl_worker("sim-d", clock)
    sent = NixlConnectorMetadata()
    sent.reqs_in_batch = {"p1", "p2"}
    prefill.start_load_kv(sent)

    def recv(blocks, remote_req, awaiting=True):
        return ReqMeta(
            local_block_ids=(blocks,),
            local_physical_block_ids=(blocks,),
            tp_size=1,
            awaiting_kvs=awaiting,
            remote=RemoteMeta(blocks, "", 0, "sim-p", remote_req),
        )

    meta = NixlConnectorMetadata()
    # 20 and 10 bytes at 10 B/s, serial on the link; r3 only notifies, about
    # a request the prefill instance no longer tracks.
    meta.reqs_to_recv = {
        "r1": recv([0, 1], "p1"),
        "r2": recv([2], "p2"),
        "r3": recv([], "p3", awaiting=False),
    }
    decode.start_load_kv(meta)
    assert decode.next_completion() == pytest.approx(2.0)

    assert decode.get_finished() == (set(), set())
    assert prefill.get_finished() == (set(), set())
    clock.sleep_until(2.5)
    assert decode.get_finished() == (set(), {"r1"})
    assert prefill.get_finished() == ({"p1"}, set())
    clock.sleep_until(3.0)
    assert decode.get_finished() == (set(), {"r2"})
    assert prefill.get_finished() == ({"p2"}, set())
    assert decode.next_completion() is None and prefill.next_completion() is None
