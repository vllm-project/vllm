# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise producer identity through real managers, Scheduler and ZMQ.

The contract is an additive source field with unchanged event order and counts.
Independent managers emitting identical hashes reproduce loss of attribution.
No model execution or physical KV transfer is needed for this control-plane path.
"""

import time
import uuid

import msgspec
import pytest
import torch
import zmq
from transformers import GPT2Config

from tests.v1.core.utils import create_requests
from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.config import KVEventsConfig
from vllm.distributed.kv_events import AllBlocksCleared, KVEventBatch, ZmqEventPublisher
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test


def offload():
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": 65536, "block_size": 16},
    }


def multi(children):
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"connectors": children},
    }


@pytest.fixture
def scheduler_factory(tmp_path):
    GPT2Config(
        n_embd=32,
        n_head=1,
        n_layer=1,
        n_positions=64,
        vocab_size=258,
        bos_token_id=256,
        eos_token_id=256,
        architectures=["GPT2LMHeadModel"],
    ).save_pretrained(tmp_path)
    schedulers = []

    def create(connector, enabled=True):
        config = create_vllm_config(
            model=str(tmp_path), max_model_len=64, max_num_seqs=4, **connector
        )
        config.cache_config.num_gpu_blocks = 32
        config.scheduler_config.async_scheduling = False
        config.model_config.skip_tokenizer_init = True
        name = uuid.uuid4().hex
        config.kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            endpoint=f"inproc://{name}",
            replay_endpoint=f"inproc://{name}-replay",
            topic="kv",
            buffer_steps=64,
        )
        # Set after construction so the regression also runs on the base revision.
        config.kv_events_config.enable_source_attribution = enabled
        cache = KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16
                    ),
                )
            ],
        )
        result = Scheduler(
            vllm_config=config,
            kv_cache_config=cache,
            structured_output_manager=StructuredOutputManager(config),
            block_size=16,
            log_stats=False,
        )
        schedulers.append(result)
        return result

    yield create
    for scheduler in schedulers:
        scheduler.shutdown()


class Capture:
    def __init__(self, publisher):
        config = publisher.get_publisher_config()
        context = zmq.Context.instance()
        self.sub = context.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"kv")
        self.sub.connect(config.endpoint)
        self.replay = context.socket(zmq.DEALER)
        self.replay.connect(config.replay_endpoint)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            publisher.publish(KVEventBatch(ts=-1.0, events=[]))
            if self.sub.poll(20):
                self.sub.recv_multipart()
                break
        else:
            self.close()
            raise AssertionError("live subscription readiness timed out")

    def receive(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not self.sub.poll(max(1, int((deadline - time.monotonic()) * 1000))):
                continue
            topic, seq, payload = self.sub.recv_multipart()
            batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
            if batch.ts != -1:
                assert topic == b"kv"
                return seq, payload, batch
        raise AssertionError("live event timed out")

    def replay_all(self):
        self.replay.send_multipart([b"", (0).to_bytes(8, "big")])
        frames: list[tuple[bytes, bytes]] = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not self.replay.poll(max(1, int((deadline - time.monotonic()) * 1000))):
                continue
            delimiter, topic, seq, payload = self.replay.recv_multipart()
            assert delimiter == b""
            if seq == ZmqEventPublisher.END_SEQ:
                return frames
            batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
            if batch.ts != -1:
                assert topic == b"kv"
                frames.append((seq, payload))
        raise AssertionError("replay end marker missing")

    def close(self):
        self.sub.close(linger=0)
        self.replay.close(linger=0)


def store(child, value=b"0123456789abcdef"):
    manager = child.scheduler.manager
    key = make_offload_key(value, 0)
    context = ReqContext(req_id="completion")
    result = manager.prepare_store([key], context)
    assert result is not None and result.keys_to_store == [key]
    manager.complete_store([key], context)


def tick(scheduler, output=None):
    scheduler.update_from_output(
        scheduler.schedule(),
        ModelRunnerOutput(req_ids=[], req_id_to_index={}, kv_connector_output=output),
    )


def source_id(event):
    source = getattr(event, "source", None)
    return source.source_id if source is not None else None


@pytest.mark.parametrize("enabled", [False, True])
def test_identical_children_retain_source_through_publication_and_replay(
    scheduler_factory, enabled
):
    s = scheduler_factory(multi([offload(), offload()]), enabled)
    capture = Capture(s.kv_event_publisher)
    try:
        first, second = s.connector.sub_connectors
        frames = []
        for child in (second, first):
            store(child)
            tick(s)
            frames.append(capture.receive())
        events = [frame[2].events[0] for frame in frames]
        assert [source_id(e) for e in events] == (
            ["connector/1", "connector/0"] if enabled else [None, None]
        )
        # Compare every payload field after removing only the added metadata.
        projected = [msgspec.to_builtins(event) for event in events]
        for event in projected:
            event.pop("source", None)
        assert projected[0] == projected[1]
        assert capture.replay_all() == [frame[:2] for frame in frames]
    finally:
        capture.close()


@pytest.mark.parametrize("nested", [False, True])
def test_single_and_nested_configured_occurrences(scheduler_factory, nested):
    config = multi([multi([offload(), offload()]), offload()]) if nested else offload()
    s = scheduler_factory(config)
    capture = Capture(s.kv_event_publisher)
    try:
        if nested:
            inner, outer = s.connector.sub_connectors
            store(inner.sub_connectors[1])
            store(outer)
            expected = ["connector/0/1", "connector/1"]
        else:
            store(s.connector)
            expected = ["connector"]
        tick(s)
        events = capture.receive()[2].events
        assert [source_id(e) for e in events] == expected
        assert all(e.source.connector_name == "OffloadingConnector" for e in events)
    finally:
        capture.close()


def test_native_and_connector_order_and_clear_origin(scheduler_factory):
    s = scheduler_factory(multi([offload(), offload()]))
    capture = Capture(s.kv_event_publisher)
    try:
        request = create_requests(1, num_tokens=16, max_tokens=1)[0]
        s.add_request(request)
        scheduled = s.schedule()
        for child in s.connector.sub_connectors:
            store(child)
        s.update_from_output(
            scheduled,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[42]],
            ),
        )
        frame = capture.receive()
        assert [source_id(e) for e in frame[2].events] == [
            "cache_manager",
            "connector/0",
            "connector/1",
        ]
        assert [e.medium for e in frame[2].events] == ["GPU", "CPU", "CPU"]
        assert s.reset_prefix_cache(reset_connector=False)
        tick(s)
        cleared = capture.receive()[2].events
        assert len(cleared) == 1 and isinstance(cleared[0], AllBlocksCleared)
        assert source_id(cleared[0]) == "cache_manager"
        # A native reset does not reset either offload manager.
        key = make_offload_key(b"0123456789abcdef", 0)
        assert all(
            c.scheduler.manager.lookup(key, ReqContext(req_id="lookup"))
            is LookupResult.HIT
            for c in s.connector.sub_connectors
        )
    finally:
        capture.close()


def test_remove_keeps_other_connector_residency_and_event_order(scheduler_factory):
    import mmap

    from vllm.distributed.kv_events import BlockRemoved, BlockStored

    child_config = offload()
    child_config["kv_connector_extra_config"]["cpu_bytes_to_use"] = mmap.PAGESIZE
    s = scheduler_factory(multi([child_config, child_config]))
    capture = Capture(s.kv_event_publisher)
    try:
        first, second = s.connector.sub_connectors
        store(first)
        store(second)
        tick(s)
        initial = capture.receive()[2].events
        assert [source_id(e) for e in initial] == ["connector/0", "connector/1"]
        store(first, b"fedcba9876543210")
        tick(s)
        events = capture.receive()[2].events
        assert [type(e) for e in events] == [BlockRemoved, BlockStored]
        assert [source_id(e) for e in events] == ["connector/0", "connector/0"]
        assert events[0].block_hashes == initial[0].block_hashes
        key = make_offload_key(b"0123456789abcdef", 0)
        assert (
            first.scheduler.manager.lookup(key, ReqContext(req_id="first"))
            is LookupResult.MISS
        )
        assert (
            second.scheduler.manager.lookup(key, ReqContext(req_id="second"))
            is LookupResult.HIT
        )
    finally:
        capture.close()


def test_restart_at_same_endpoint_does_not_promise_wire_identity():
    endpoint = "inproc://source-restart"
    payloads = []
    for _ in range(2):
        publisher = ZmqEventPublisher(0, endpoint, endpoint + "-replay", topic="kv")
        capture = Capture(publisher)
        try:
            from vllm.distributed.kv_events import EventSource

            publisher.publish(
                KVEventBatch(
                    ts=1.0,
                    events=[AllBlocksCleared(EventSource("cache_manager", "native"))],
                )
            )
            frame = capture.receive()
            payloads.append(frame[1])
            assert capture.replay_all()[-1] == frame[:2]
        finally:
            capture.close()
            publisher.shutdown()
    assert payloads[0] == payloads[1]


@pytest.mark.parametrize("enabled", [False, True])
def test_publisher_discovery_reports_source_setting(scheduler_factory, enabled):
    s = scheduler_factory(offload(), enabled)
    assert s.get_kv_event_publisher_config().enable_source_attribution is enabled
