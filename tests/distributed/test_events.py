# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time

import msgspec
import pytest

from vllm.distributed.kv_events import (
    EventBatch,
    EventPublisherFactory,
    NullEventPublisher,
    ZmqEventPublisher,
)

DP_RANK = 0


class EventSample(
    msgspec.Struct,
    tag=True,  # type: ignore
    array_like=True,  # type: ignore
):
    """Test event for publisher testing"""

    id: int
    value: str


class SampleBatch(EventBatch):
    """Test event batch for publisher testing"""

    events: list[EventSample]


def create_test_events(count: int) -> SampleBatch:
    """Create a batch of test events"""
    events = [EventSample(id=i, value=f"test-{i}") for i in range(count)]
    return SampleBatch(ts=time.time(), events=events)


def test_basic_publishing(publisher, subscriber):
    """Test basic event publishing works"""

    test_batch = create_test_events(5)
    publisher.publish(test_batch)

    result = subscriber.receive_one(timeout=1000)
    assert result is not None, "No message received"

    seq, received = result
    assert seq == 0, "Sequence number mismatch"
    assert received.ts == pytest.approx(test_batch.ts, abs=0.1), "Timestamp mismatch"
    assert len(received.events) == len(test_batch.events), "Number of events mismatch"

    for i, event in enumerate(received.events):
        assert event.id == i, "Event id mismatch"
        assert event.value == f"test-{i}", "Event value mismatch"


def test_multiple_events(publisher, subscriber):
    """Test publishing and receiving multiple event batches"""
    for _ in range(10):
        batch = create_test_events(2)
        publisher.publish(batch)

    received = []
    for _ in range(10):
        data = subscriber.receive_one(timeout=100)
        if data:
            received.append(data)

    assert len(received) == 10, "Number of messages mismatch"
    seqs = [seq for seq, _ in received]
    assert seqs == list(range(10)), "Sequence numbers mismatch"


def test_replay_mechanism(publisher, subscriber):
    """Test the replay mechanism works correctly"""
    for _ in range(19):
        batch = create_test_events(1)
        publisher.publish(batch)

    # Drain live events to ensure publisher has buffered them.
    for _ in range(19):
        assert subscriber.receive_one(timeout=1000) is not None

    subscriber.request_replay(10)

    replayed = subscriber.receive_replay()

    assert len(replayed) == 9, (
        f"Expected 9 replayed messages (seq 10-18), got {len(replayed)}"
    )
    seqs = [seq for seq, _ in replayed]
    assert seqs == list(range(10, 19)), "Replayed sequences should be 10-18"


def test_replay_includes_topic(publisher, subscriber, publisher_config):
    """Test that replay responses include the topic, matching PUB format"""
    for _ in range(5):
        publisher.publish(create_test_events(1))

    # Drain live events to ensure publisher has processed them.
    for _ in range(5):
        assert subscriber.receive_one(timeout=1000) is not None

    subscriber.request_replay(0)

    # receive_replay unpacks (topic, seq, payload) and asserts
    # topic == publisher topic for each message.
    replayed = subscriber.receive_replay()
    assert len(replayed) == 5, f"Expected 5 replayed messages, got {len(replayed)}"
    seqs = [seq for seq, _ in replayed]
    assert seqs == list(range(5)), "Replayed sequences should be 0-4"


def test_buffer_limit(publisher, subscriber, publisher_config):
    """Test buffer limit behavior"""
    buffer_size = publisher_config.buffer_steps

    # Publish more events than the buffer can hold
    for i in range(buffer_size + 10):
        batch = create_test_events(1)
        publisher.publish(batch)

    time.sleep(0.5)  # Need publisher to process above requests
    subscriber.request_replay(0)

    replayed = subscriber.receive_replay()

    assert len(replayed) == buffer_size, (
        f"Expected {buffer_size} replayed messages, got {len(replayed)}"
    )

    seqs = [seq for seq, _ in replayed]
    assert seqs == list(range(10, buffer_size + 10)), (
        "Should replay seq 11 through buffer_size+10"
    )


def test_topic_filtering(publisher_config):
    """
    Test that a subscriber only receives messages matching its topic filter
    """
    publisher_config.replay_endpoint = None

    publisher_config.topic = "foo"
    pub = EventPublisherFactory.create(publisher_config, DP_RANK)

    from .conftest import MockSubscriber

    sub_foo = MockSubscriber(publisher_config.endpoint, None, "foo")
    sub_bar = MockSubscriber(publisher_config.endpoint, None, "bar")

    try:
        time.sleep(0.1)

        for _ in range(3):
            pub.publish(create_test_events(1))

        foo_received = [sub_foo.receive_one(timeout=200) for _ in range(3)]
        assert all(msg is not None for msg in foo_received), (
            "Subscriber with matching topic should receive messages"
        )

        bar_received = [sub_bar.receive_one(timeout=200) for _ in range(3)]
        assert all(msg is None for msg in bar_received), (
            "Subscriber with non-matching topic should receive no messages"
        )
    finally:
        pub.shutdown()
        sub_foo.close()
        sub_bar.close()


def test_high_volume(publisher, subscriber):
    """Test publishing and receiving a high volume of events"""
    num_batches = 10_000
    events_per_batch = 100

    # Publish events in a separate thread to not block
    def publish_events():
        for i in range(num_batches):
            batch = create_test_events(events_per_batch)
            publisher.publish(batch)
            # Small delay to avoid overwhelming
            if i % 100 == 0:
                time.sleep(0.01)

    received: list[tuple[int, SampleBatch]] = []

    publisher_thread = threading.Thread(target=publish_events)
    publisher_thread.start()

    start_time = time.time()
    while len(received) < num_batches:
        if time.time() - start_time > 10:  # Timeout after 10 seconds
            break

        result = subscriber.receive_one(timeout=100)
        if result:
            received.append(result)

    publisher_thread.join()

    assert len(received) >= num_batches * 0.9, "We should have received most messages"

    seqs = [seq for seq, _ in received]
    assert sorted(seqs) == seqs, "Sequence numbers should be in order"


def test_null_publisher():
    """Test that NullEventPublisher can be used without errors"""
    publisher = NullEventPublisher(DP_RANK)

    # This should not raise any errors
    batch = create_test_events(5)
    publisher.publish(batch)
    publisher.shutdown()


def test_data_parallel_rank_tagging(publisher_config):
    """Test that events are properly tagged with their data parallel rank"""

    publisher_config.topic = "foo"
    pub_0 = EventPublisherFactory.create(publisher_config, DP_RANK)
    pub_1 = EventPublisherFactory.create(publisher_config, DP_RANK + 1)

    # Hardcode the expected endpoints based on port offsetting behavior
    # Both ranks get offsets according to _offset_endpoint_port function
    base_endpoint = publisher_config.endpoint
    if "tcp://" in base_endpoint:
        # For TCP endpoints: tcp://localhost:5557 -> tcp://localhost:5557, tcp://localhost:5558
        expected_endpoint_0 = base_endpoint  # rank 0 gets port + 0 = same port
        expected_endpoint_1 = base_endpoint.replace(
            ":5557", ":5558"
        )  # rank 1 gets port + 1
    else:
        # For inproc endpoints: inproc://test -> inproc://test_dp0, inproc://test_dp1
        expected_endpoint_0 = base_endpoint  # rank 0 gets base
        expected_endpoint_1 = base_endpoint + "_dp1"  # rank 1 gets _dp1

    from .conftest import MockSubscriber

    sub_0 = MockSubscriber(expected_endpoint_0, None, publisher_config.topic)
    sub_1 = MockSubscriber(expected_endpoint_1, None, publisher_config.topic)

    try:
        time.sleep(0.1)  # Let publishers start up

        # Publish events from different ranks
        batch_0 = create_test_events(2)
        batch_1 = create_test_events(3)

        pub_0.publish(batch_0)
        pub_1.publish(batch_1)

        # Receive events from rank 0
        result_0 = sub_0.receive_one(timeout=200)
        assert result_0 is not None, "No message received from rank 0"
        seq_0, received_0 = result_0

        # Receive events from rank 1
        result_1 = sub_1.receive_one(timeout=200)
        assert result_1 is not None, "No message received from rank 1"
        seq_1, received_1 = result_1

        # Verify DP rank tagging
        assert received_0.data_parallel_rank == 0, (
            f"Expected DP rank 0, got {received_0.data_parallel_rank}"
        )
        assert received_1.data_parallel_rank == 1, (
            f"Expected DP rank 1, got {received_1.data_parallel_rank}"
        )

        # Verify event content is correct
        assert len(received_0.events) == 2, "Wrong number of events from rank 0"
        assert len(received_1.events) == 3, "Wrong number of events from rank 1"

    finally:
        pub_0.shutdown()
        pub_1.shutdown()
        sub_0.close()
        sub_1.close()


def test_event_publisher_factory(random_port):
    """Test event publisher factory creation behavior under different configurations"""
    from vllm.config.kv_events import KVEventsConfig
    from vllm.distributed.kv_events import ZmqEventPublisher

    # test config is None
    publisher = EventPublisherFactory.create(None, DP_RANK)
    assert isinstance(publisher, NullEventPublisher)
    publisher.shutdown()

    # test disable kv cache events
    config = KVEventsConfig(
        enable_kv_cache_events=False,
        publisher="zmq",  # Even if zmq is specified, should return NullEventPublisher
        endpoint="tcp://localhost:5557",
    )
    publisher = EventPublisherFactory.create(config, DP_RANK)
    assert isinstance(publisher, NullEventPublisher)
    publisher.shutdown()

    # test zmq publisher
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{random_port}",
        replay_endpoint=f"tcp://*:{random_port + 100}",
    )
    publisher = EventPublisherFactory.create(config, DP_RANK + 1)
    assert isinstance(publisher, ZmqEventPublisher)
    resolved_config = publisher.get_publisher_config()
    assert resolved_config.endpoint == f"tcp://*:{random_port + 1}"
    assert resolved_config.replay_endpoint == f"tcp://*:{random_port + 101}"
    assert config.endpoint == f"tcp://*:{random_port}"
    publisher.shutdown()

    # test unknown publisher
    with pytest.raises(ValueError, match="Input should be"):
        KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="unknown_publisher",
            endpoint="tcp://localhost:5557",
        )

    # test publisher not specified
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        # publisher not specified, should default to "zmq"
        endpoint="tcp://localhost:5557",
    )
    publisher = EventPublisherFactory.create(config, DP_RANK)
    assert isinstance(publisher, ZmqEventPublisher)
    publisher.shutdown()


def test_offset_endpoint_port_ephemeral_tcp():
    """A configured port of 0 (ephemeral) is never offset: every DP rank
    must bind independently and let the OS pick a port."""
    assert ZmqEventPublisher.offset_endpoint_port("tcp://*:0", 0) == "tcp://*:0"
    assert ZmqEventPublisher.offset_endpoint_port("tcp://*:0", 1) == "tcp://*:0"
    assert ZmqEventPublisher.offset_endpoint_port("tcp://*:0", 3) == "tcp://*:0"


def test_offset_endpoint_port_explicit_tcp_backward_compat():
    """Explicit non-zero ports keep the base_port + dp_rank behavior."""
    assert ZmqEventPublisher.offset_endpoint_port("tcp://*:5557", 0) == "tcp://*:5557"
    assert ZmqEventPublisher.offset_endpoint_port("tcp://*:5557", 1) == "tcp://*:5558"
    assert ZmqEventPublisher.offset_endpoint_port("tcp://*:5557", 3) == "tcp://*:5560"


def test_ephemeral_publisher_resolves_real_port():
    """A publisher bound to tcp://*:0 reports the actual bound port on a
    real, dialable host (not the wildcard "*" it was bound with), and the
    config passed in by the caller is left untouched."""
    from vllm.config.kv_events import KVEventsConfig
    from vllm.utils.network_utils import get_ip, split_host_port

    config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://*:0",
    )
    publisher = EventPublisherFactory.create(config, DP_RANK)
    try:
        assert isinstance(publisher, ZmqEventPublisher)
        resolved = publisher.get_publisher_config()
        assert resolved.endpoint.startswith("tcp://")
        assert "*" not in resolved.endpoint

        host, port = split_host_port(resolved.endpoint.removeprefix("tcp://"))
        assert host == get_ip()
        assert port != 0

        # Original input config must not be mutated.
        assert config.endpoint == "tcp://*:0"
    finally:
        publisher.shutdown()


def test_multiple_ephemeral_publishers_get_distinct_ports():
    """Two DP ranks both requesting tcp://*:0 must resolve to different
    real, dialable hosts:ports, since ephemeral ports are no longer
    rank-offset."""
    from vllm.config.kv_events import KVEventsConfig
    from vllm.utils.network_utils import get_ip, split_host_port

    config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://*:0",
    )
    pub_0 = EventPublisherFactory.create(config, DP_RANK)
    pub_1 = EventPublisherFactory.create(config, DP_RANK + 1)
    try:
        assert isinstance(pub_0, ZmqEventPublisher)
        assert isinstance(pub_1, ZmqEventPublisher)
        endpoint_0 = pub_0.get_publisher_config().endpoint
        endpoint_1 = pub_1.get_publisher_config().endpoint
        assert endpoint_0 != "tcp://*:0"
        assert endpoint_1 != "tcp://*:0"
        assert endpoint_0 != endpoint_1

        host_0, _ = split_host_port(endpoint_0.removeprefix("tcp://"))
        host_1, _ = split_host_port(endpoint_1.removeprefix("tcp://"))
        assert host_0 == host_1 == get_ip()
    finally:
        pub_0.shutdown()
        pub_1.shutdown()


def test_ephemeral_replay_endpoint_resolves_real_port():
    """replay_endpoint="tcp://*:0" also resolves to a real, dialable host
    and non-zero port, without mutating the caller's original config."""
    from vllm.config.kv_events import KVEventsConfig
    from vllm.utils.network_utils import get_ip, split_host_port

    config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint="tcp://*:0",
        replay_endpoint="tcp://*:0",
    )
    publisher = EventPublisherFactory.create(config, DP_RANK)
    try:
        assert isinstance(publisher, ZmqEventPublisher)
        resolved = publisher.get_publisher_config()
        assert resolved.replay_endpoint is not None
        assert resolved.replay_endpoint.startswith("tcp://")
        assert "*" not in resolved.replay_endpoint

        host, port = split_host_port(resolved.replay_endpoint.removeprefix("tcp://"))
        assert host == get_ip()
        assert port != 0

        # Original input config must not be mutated.
        assert config.replay_endpoint == "tcp://*:0"
    finally:
        publisher.shutdown()


def test_ephemeral_publisher_binds_for_0_0_0_0():
    """A 0.0.0.0 wildcard endpoint must bind and resolve its ephemeral port."""
    from vllm.config.kv_events import KVEventsConfig
    from vllm.utils.network_utils import get_ip, split_host_port

    wildcard_endpoint = "tcp://0.0.0.0:0"
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=wildcard_endpoint,
    )
    publisher = EventPublisherFactory.create(config, DP_RANK)
    try:
        assert isinstance(publisher, ZmqEventPublisher)
        resolved = publisher.get_publisher_config()
        assert resolved.endpoint != wildcard_endpoint

        host, port = split_host_port(resolved.endpoint.removeprefix("tcp://"))
        assert host == get_ip()
        assert port != 0
    finally:
        publisher.shutdown()
