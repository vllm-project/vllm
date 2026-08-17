# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-notification buffer and cross-rank gather.

Bare `EngineCore` instances: the helpers only touch `_pending_notifications`,
so no model is needed.
"""

import queue
import time
import uuid
from types import SimpleNamespace

import msgspec
import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest, core
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.executor.abstract import Executor
from vllm.v1.notifications import (
    CustomNotification,
    publish_worker_notification,
    take_worker_notifications,
)

from ...utils import create_new_process_for_each_test


@pytest.fixture(autouse=True)
def _drain_worker_buffer():
    """Process-local buffer; don't leak events between tests."""
    take_worker_notifications()
    yield
    take_worker_notifications()


def _bare_engine_core() -> EngineCore:
    engine_core = EngineCore.__new__(EngineCore)
    engine_core._pending_notifications = []
    return engine_core


def test_notifications_accumulate_additively():
    """Additive: no coalescing, or counter increments would get lost."""
    engine_core = _bare_engine_core()

    first = CustomNotification(key="my_plugin", payload={"n": 1})
    second = CustomNotification(key="my_plugin", payload={"n": 2})
    engine_core._publish_notifications([first])
    engine_core._publish_notifications([second])

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)

    assert outputs[0].engine_notifications == [first, second]


def test_flush_clears_buffer_between_steps():
    """A flush drains the buffer; later events are independent."""
    engine_core = _bare_engine_core()

    engine_core._publish_notifications([CustomNotification(key="my_plugin")])
    first_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(first_outputs)

    second_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(second_outputs)
    assert second_outputs == {}

    later = CustomNotification(key="my_plugin", payload={"n": 3})
    engine_core._publish_notifications([later])
    third_outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(third_outputs)
    assert third_outputs[0].engine_notifications == [later]


def test_flush_reuses_existing_outputs():
    """Notifications attach to an existing per-engine outputs entry."""
    engine_core = _bare_engine_core()

    event = CustomNotification(key="my_plugin")
    engine_core._publish_notifications([event])

    existing = EngineCoreOutputs(engine_index=2)
    outputs = {2: existing}
    engine_core._flush_notifications(outputs)

    assert outputs == {2: existing}
    assert existing.engine_notifications == [event]


def test_flush_noop_when_empty():
    """Nothing is added to outputs when the buffer is empty."""
    engine_core = _bare_engine_core()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs == {}


def test_worker_publish_take_roundtrip():
    """The out-of-tree producer path."""
    event = CustomNotification(key="my_plugin", payload={"bytes": 1})
    publish_worker_notification(event)

    assert take_worker_notifications() == [event]
    # None rather than [], so the quiet path allocates nothing.
    assert take_worker_notifications() is None


def test_worker_notifications_survive_until_the_first_drain():
    """Load-time producers publish before any gather runs."""
    first = CustomNotification(key="my_plugin", payload={"n": 1})
    second = CustomNotification(key="my_plugin", payload={"n": 2})
    publish_worker_notification(first)
    publish_worker_notification(second)

    assert take_worker_notifications() == [first, second]


def _gathering_engine_core(per_rank) -> EngineCore:
    """EngineCore whose executor answers take_notifications with per_rank."""
    engine_core = _bare_engine_core()
    executor = SimpleNamespace(collective_rpc=lambda method: per_rank)
    engine_core.model_executor = executor
    return engine_core


def test_gather_keeps_every_rank():
    """A rank-0-only reply would drop per-rank producers entirely."""
    rank0 = CustomNotification(key="my_plugin", payload={"rank": 0})
    rank1 = CustomNotification(key="my_plugin", payload={"rank": 1})
    rank2 = CustomNotification(key="my_plugin", payload={"rank": 2})
    engine_core = _gathering_engine_core([[rank0], [rank1], [rank2]])

    engine_core.gather_worker_notifications()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs[0].engine_notifications == [rank0, rank1, rank2]


def test_gather_publishes_nothing_when_all_ranks_are_quiet():
    engine_core = _gathering_engine_core([[], [], []])

    engine_core.gather_worker_notifications()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs == {}


def test_gather_tolerates_ranks_that_cannot_report():
    """A worker without the method replies None."""
    event = CustomNotification(key="my_plugin", payload={"rank": 1})
    engine_core = _gathering_engine_core([None, [event]])

    engine_core.gather_worker_notifications()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs[0].engine_notifications == [event]


def test_gather_never_raises_into_the_engine():
    """A metrics channel must not take down an engine operation."""
    engine_core = _bare_engine_core()

    def boom(method):
        raise RuntimeError("executor is gone")

    engine_core.model_executor = SimpleNamespace(collective_rpc=boom)
    engine_core.gather_worker_notifications()

    outputs: dict[int, EngineCoreOutputs] = {}
    engine_core._flush_notifications(outputs)
    assert outputs == {}


def test_engine_core_outputs_round_trip_over_the_wire():
    """array_like, so the field is positional and always present.

    omit_defaults does not trim trailing fields, and non-Python frontends
    reject an array longer than they know.
    """
    event = CustomNotification(key="my_plugin", payload={"count": 5})
    encoded = msgspec.msgpack.encode(
        EngineCoreOutputs(engine_index=1, engine_notifications=[event])
    )

    as_array = msgspec.msgpack.decode(encoded)
    assert isinstance(as_array, list)
    assert as_array[-1] == [
        {"type": "custom", "key": "my_plugin", "payload": {"count": 5}}
    ]

    decoded = msgspec.msgpack.decode(encoded, type=EngineCoreOutputs)
    assert decoded.engine_notifications == [event]


def test_proc_broadcasts_to_every_frontend():
    """Every API server gets the event, not just one."""
    proc = EngineCoreProc.__new__(EngineCoreProc)
    proc.addresses = SimpleNamespace(outputs=["a", "b", "c"])
    proc.output_queue = queue.Queue()

    event = CustomNotification(key="my_plugin")
    proc._publish_notifications([event])

    delivered = [proc.output_queue.get_nowait() for _ in range(3)]
    assert [client_index for client_index, _ in delivered] == [0, 1, 2]
    assert all(out.engine_notifications == [event] for _, out in delivered)
    assert proc.output_queue.empty()


def test_dp_busy_loop_gathers_on_startup():
    """The DP override of run_busy_loop shipped without either gather call
    once; pin the startup gather so it cannot regress. Pre-loop placement is
    the point: the ordering test below cannot tell pre-loop from
    first-iteration, and a first-iteration gather would be an unthrottled rpc
    per iteration."""
    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    gathered = []
    proc.gather_worker_notifications = lambda: gathered.append("startup")
    proc._handle_shutdown = lambda: False

    with pytest.raises(SystemExit):
        proc.run_busy_loop()

    assert gathered == ["startup"]


def test_dp_busy_loop_polls_inside_each_iteration():
    """The interval poll sits between input processing and the step, matching
    the non-DP loop."""
    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    calls = []
    proc.gather_worker_notifications = lambda: calls.append("startup")
    proc._handle_shutdown = lambda: True
    proc._process_input_queue = lambda: calls.append("input")
    proc._maybe_publish_request_counts = lambda: None

    def poll():
        calls.append("poll")
        # The rest of the iteration needs a scheduler and an executor; the
        # wiring is proven by reaching here, so stop the loop.
        raise SystemExit

    proc._maybe_gather_worker_notifications = poll

    with pytest.raises(SystemExit):
        proc.run_busy_loop()

    assert calls == ["startup", "input", "poll"]


def test_poll_is_off_by_default(monkeypatch):
    """Interval 0 must mean no gather rpc at all, however overdue."""
    monkeypatch.setattr(core.envs, "VLLM_WORKER_NOTIFICATION_POLL_INTERVAL", 0.0)
    proc = EngineCoreProc.__new__(EngineCoreProc)
    proc._last_notification_gather = float("-inf")
    gathered = []
    proc.gather_worker_notifications = lambda: gathered.append(1)

    proc._maybe_gather_worker_notifications()

    assert gathered == []


def test_poll_interval_gates_the_rpc(monkeypatch):
    """One gather when due, then nothing until the interval passes again."""
    monkeypatch.setattr(core.envs, "VLLM_WORKER_NOTIFICATION_POLL_INTERVAL", 60.0)
    proc = EngineCoreProc.__new__(EngineCoreProc)
    proc._last_notification_gather = time.monotonic() - 120.0
    gathered = []
    proc.gather_worker_notifications = lambda: gathered.append(1)

    proc._maybe_gather_worker_notifications()
    assert len(gathered) == 1

    proc._maybe_gather_worker_notifications()
    assert len(gathered) == 1, "gathered again inside the interval"


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Requires a CUDA-alike platform to run a real engine.",
)
@create_new_process_for_each_test("spawn")
def test_custom_notification_reaches_frontend(monkeypatch: pytest.MonkeyPatch):
    """Full path with a real engine: a plugin-style publish inside the worker
    process, the interval poll gather, the msgpack wire, and delivery on the
    frontend client, interleaved with normal generation traffic."""
    monkeypatch.setenv("VLLM_WORKER_NOTIFICATION_POLL_INTERVAL", "0.001")
    # The publish callable rides collective_rpc as a cloudpickle payload.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    engine_args = EngineArgs(model="Qwen/Qwen3-0.6B", enforce_eager=True)
    vllm_config = engine_args.create_engine_config(UsageContext.UNKNOWN_CONTEXT)
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        client = EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
        )

    try:
        # What a LoRA-hotswap plugin would emit after landing an adapter,
        # so a router can steer matching requests to this worker.
        adapter_loaded = {
            "event": "adapter_loaded",
            "adapter": "shakespeare-insults-v2",
            "rank": 0,
            "load_ms": 128,
        }

        def publish(worker):
            from vllm.v1.notifications import (
                CustomNotification,
                publish_worker_notification,
            )

            publish_worker_notification(
                CustomNotification(key="lora_hotswap", payload=adapter_loaded)
            )
            return "published"

        assert client.collective_rpc(publish) == ["published"]

        request_id = f"request-{uuid.uuid4()}"
        client.add_request(
            EngineCoreRequest(
                request_id=request_id,
                external_req_id=request_id,
                prompt_token_ids=list(range(10, 30)),
                mm_features=None,
                sampling_params=SamplingParams(max_tokens=5),
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )
        )

        notifications: list[CustomNotification] = []
        finished = False
        while not (notifications and finished):
            outputs = client.get_output()
            notifications.extend(outputs.engine_notifications or ())
            finished |= any(out.finished for out in outputs.outputs)

        assert notifications == [
            CustomNotification(key="lora_hotswap", payload=adapter_loaded)
        ]
    finally:
        client.shutdown()
