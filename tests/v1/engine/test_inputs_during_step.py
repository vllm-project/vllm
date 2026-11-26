# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EngineCoreProc handling inputs while waiting for a step's output."""

import queue
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any
from unittest import mock
from uuid import uuid4

import pytest
import zmq

from tests.v1.kv_connector.unit.utils import MockKVConnector
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine import core as engine_core
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.abstract import Executor

ADD = EngineCoreRequestType.ADD
ABORT = EngineCoreRequestType.ABORT
UTILITY = EngineCoreRequestType.UTILITY
START_DP_WAVE = EngineCoreRequestType.START_DP_WAVE
EXECUTOR_FAILED = EngineCoreRequestType.EXECUTOR_FAILED
WAKEUP = EngineCoreRequestType.WAKEUP


class PollableFuture(Future):
    """Becomes done on the `ready_after`-th poll; on_poll is called first."""

    def __init__(
        self,
        ready_after: int,
        on_poll: Callable[[int], None] | None = None,
    ):
        super().__init__()
        self.ready_after = ready_after
        self.on_poll = on_poll
        self.polls = 0
        self.waits = 0

    def poll(self) -> bool:
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        if self.polls >= self.ready_after and not self.done():
            self.set_result("output")
        return self.done()

    def wait_briefly(self, pending, deadline) -> None:
        self.waits += 1

    def result(self, timeout=None):
        # Stands in for blocking until the output arrives.
        if not self.done():
            self.set_result("output")
        return super().result(timeout)


def make_engine(handle: bool = True) -> tuple[EngineCoreProc, list[Any]]:
    """An EngineCoreProc with just the state used by input handling."""
    engine = object.__new__(EngineCoreProc)
    engine.input_queue = queue.Queue()
    engine._deferred_inputs = deque()
    engine.has_work = lambda: True
    handled: list[Any] = []
    if handle:

        def record(request_type: EngineCoreRequestType, request: Any) -> None:
            handled.append((request_type, request))

        engine._handle_client_request = record
    return engine, handled


def test_adds_and_aborts_are_handled_while_waiting():
    engine, handled = make_engine()

    def on_poll(n: int) -> None:
        if n == 2:
            engine.input_queue.put((ADD, "a"))
        elif n == 4:
            engine.input_queue.put((ABORT, ["a"]))
            engine.input_queue.put((WAKEUP, None))

    future = PollableFuture(ready_after=10, on_poll=on_poll)
    assert engine._wait_for_output(future) == "output"
    assert future.polls == 10
    assert handled == [(ADD, "a"), (ABORT, ["a"]), (WAKEUP, None)]
    assert not engine._deferred_inputs


@pytest.mark.parametrize("barrier", [UTILITY, START_DP_WAVE])
def test_other_inputs_and_those_after_them_wait_for_step(barrier):
    engine, handled = make_engine()
    inputs = [(ADD, "a"), (ABORT, ["x"]), (barrier, "u"), (ADD, "b"), (ABORT, ["a"])]
    for req in inputs:
        engine.input_queue.put(req)

    future = PollableFuture(ready_after=1000)
    assert engine._wait_for_output(future) == "output"
    # Stopped polling at the first input that can't be handled during the step.
    assert future.polls < 10
    assert handled == inputs[:2]
    assert list(engine._deferred_inputs) == [inputs[2]]

    # After the step, the remaining inputs are handled in their original order.
    engine._process_input_queue()
    assert handled == inputs
    assert not engine._deferred_inputs
    assert engine.input_queue.empty()


def test_executor_failure_raises_while_waiting():
    engine, _ = make_engine(handle=False)
    engine.input_queue.put((EXECUTOR_FAILED, b""))
    with pytest.raises(RuntimeError, match="Executor failed"):
        engine._wait_for_output(PollableFuture(ready_after=1000))


def test_future_without_poll_blocks_for_result():
    engine, handled = make_engine()
    engine.input_queue.put((ADD, "a"))
    future: Future = Future()
    future.set_result("output")
    assert engine._wait_for_output(future) == "output"
    assert handled == []
    assert engine.input_queue.qsize() == 1


def test_blocks_for_output_after_spin_period():
    """Once the spin period has passed, pending inputs are still handled, then
    the wait blocks for the output rather than busy-looping."""
    engine, handled = make_engine()
    engine.input_queue.put((ADD, "a"))
    future = PollableFuture(ready_after=1000)
    with mock.patch.object(engine_core, "_INPUT_SPIN_DURATION_S", 0.0):
        assert engine._wait_for_output(future) == "output"
    assert future.waits == 0
    assert future.polls == 2
    assert handled == [(ADD, "a")]


def test_stop_on_add_returns_at_first_add():
    """With a free batch-queue slot, the wait returns as soon as a request is added
    (aborts don't stop it), leaving the step's output to be collected later."""
    engine, handled = make_engine()

    def on_poll(n: int) -> None:
        if n == 2:
            engine.input_queue.put((ABORT, ["x"]))
        elif n == 4:
            engine.input_queue.put((ADD, "a"))
            engine.input_queue.put((ADD, "b"))

    future = PollableFuture(ready_after=1000, on_poll=on_poll)
    assert engine._handle_inputs_while_waiting(future, stop_on_add=True)
    assert not future.done()
    assert handled == [(ABORT, ["x"]), (ADD, "a")]
    assert engine.input_queue.qsize() == 1

    # A request arriving after the spin period still stops the wait.
    engine, handled = make_engine()

    def on_late_poll(n: int) -> None:
        if n == 5:
            engine.input_queue.put((ADD, "c"))

    future = PollableFuture(ready_after=1000, on_poll=on_late_poll)
    with mock.patch.object(engine_core, "_INPUT_SPIN_DURATION_S", 0.0):
        assert engine._handle_inputs_while_waiting(future, stop_on_add=True)
    assert future.waits == 0
    assert handled == [(ADD, "c")]

    # Output ready before any request arrives.
    engine, handled = make_engine()
    assert not engine._handle_inputs_while_waiting(
        PollableFuture(ready_after=3), stop_on_add=True
    )


@pytest.mark.parametrize("notify_before_get", [True, False])
def test_connector_wakeup_closes_idle_queue_race(notify_before_get):
    """A notification both before and during a blocking get wakes the engine."""
    from concurrent.futures import ThreadPoolExecutor

    from vllm.v1.engine.core import _ConnectorWakeup

    inputs: queue.Queue = queue.Queue()
    wakeup = _ConnectorWakeup(inputs)
    with ThreadPoolExecutor(1) as executor:
        if notify_before_get:
            wakeup.notify()
        waiting = executor.submit(inputs.get, True, 5)
        wakeup.notify()
        assert waiting.result(timeout=5) == (
            EngineCoreRequestType.WAKEUP,
            None,
        )
        wakeup.notify()
        assert inputs.empty()
        wakeup.consume()
        wakeup.notify()
        assert inputs.get_nowait()[0] == EngineCoreRequestType.WAKEUP


class WakeupConnector(MockKVConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.endpoint = vllm_config.kv_transfer_config.kv_connector_extra_config[
            "endpoint"
        ]
        self.pending = threading.Event()
        self.stepped = threading.Event()
        self.stop = threading.Event()
        self.callback = None
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def set_engine_wakeup(self, callback):
        self.callback = callback

    def serve(self):
        with zmq.Context() as ctx, ctx.socket(zmq.REP) as socket:
            socket.bind(self.endpoint)
            while not self.stop.is_set():
                if socket.poll(100):
                    socket.recv()
                    self.pending.set()
                    if self.callback is not None:
                        self.callback()
                    if self.stepped.wait(5):
                        socket.send(b"stepped")

    def has_pending_push_work(self):
        return self.pending.is_set()

    def build_connector_meta(self, output):
        self.pending.clear()
        self.stepped.set()
        return super().build_connector_meta(output)

    def shutdown(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)


class WakeupExecutor(Executor):
    def _init_executor(self):
        pass

    def check_health(self):
        pass

    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block=False
    ):
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

        result = {
            "get_kv_cache_layer_configs": [{}],
            "get_supported_kv_cache_layouts": [["LBNHC"]],
            "get_supported_tasks": [("generate",)],
            "get_kv_connector_handshake_metadata": [None],
            "compile_or_warm_up_model": [],
            "execute_model": [EMPTY_MODEL_RUNNER_OUTPUT],
        }.get(method, [None])
        if non_block:
            future: Future[list[Any]] = Future()
            future.set_result(result)
            return future
        return result

    def execute_model(self, scheduler_output, non_block=False):
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

        if non_block:
            future: Future[Any] = Future()
            future.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
            return future
        return EMPTY_MODEL_RUNNER_OUTPUT


@pytest.fixture
def wakeup_config(tmp_path):
    from transformers import OPTConfig

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    OPTConfig(
        hidden_size=64,
        num_attention_heads=2,
        ffn_dim=128,
        num_hidden_layers=1,
        vocab_size=128,
        max_position_embeddings=64,
    ).save_pretrained(tmp_path)
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, dtype="float32"
        ),
        device_config=DeviceConfig("cpu"),
        cache_config=CacheConfig(
            block_size=16, enable_prefix_caching=False, enable_extensible_kv_cache=False
        ),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_num_seqs=2,
            max_model_len=64,
            max_num_batched_tokens=64,
            enable_chunked_prefill=False,
            async_scheduling=False,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="WakeupConnector",
            kv_connector_module_path=__name__,
            kv_role="kv_both",
            kv_connector_extra_config={
                "matched_tokens": 0,
                "is_async": False,
                "endpoint": f"ipc://@vllm-wakeup-{uuid4().hex}",
            },
        ),
    )


def test_idle_engine_proc_steps_for_connector_and_shuts_down(wakeup_config):
    import zmq

    from vllm.v1.engine.core_client import EngineCoreClient, SyncMPClient

    client = EngineCoreClient.make_client(
        multiprocess_mode=True,
        asyncio_mode=False,
        vllm_config=wakeup_config,
        executor_class=WakeupExecutor,
        collect_stats=False,
    )
    assert isinstance(client, SyncMPClient)
    try:
        client.call_utility("pause_scheduler", "wait", False, False)
        with zmq.Context() as ctx, ctx.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(
                wakeup_config.kv_transfer_config.kv_connector_extra_config["endpoint"]
            )
            socket.send(b"work")
            assert socket.poll(10000), "Idle engine did not run connector maintenance"
            assert socket.recv() == b"stepped"
    finally:
        client.shutdown()
