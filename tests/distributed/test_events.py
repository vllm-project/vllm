# SPDX-License-Identifier: Apache-2.0
import threading
import time

import msgspec
import pytest

from vllm.distributed.kv_events import (EventBatch, EventPublisherFactory,
                                        NullEventPublisher)


class EventSample(
        msgspec.Struct,
        tag=True,  # type: ignore
        array_like=True  # type: ignore
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
    assert received.ts == pytest.approx(test_batch.ts,
                                        abs=0.1), ("Timestamp mismatch")
    assert len(received.events) == len(
        test_batch.events), ("Number of events mismatch")

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

    time.sleep(0.5)  # Need publisher to process above requests
    subscriber.request_replay(10)

    batch = create_test_events(1)
    publisher.publish(batch)  # 20th message

    replayed = subscriber.receive_replay()

    assert len(replayed) > 0, "No replayed messages received"
    seqs = [seq for seq, _ in replayed]
    assert all(seq >= 10 for seq in seqs), "Replayed messages not in order"
    assert seqs == list(range(min(seqs),
                              max(seqs) +
                              1)), ("Replayed messages not consecutive")


def test_buffer_limit(publisher, subscriber, publisher_config):
    """Test buffer limit behavior"""
    buffer_size = publisher_config.buffer_steps

    # Publish more events than the buffer can hold
    for i in range(buffer_size + 10):
        batch = create_test_events(1)
        publisher.publish(batch)

    time.sleep(0.5)  # Need publisher to process above requests
    subscriber.request_replay(0)

    batch = create_test_events(1)
    publisher.publish(batch)

    replayed = subscriber.receive_replay()

    assert len(replayed) <= buffer_size, "Can't replay more than buffer size"

    oldest_seq = min(seq for seq, _ in replayed)
    assert oldest_seq >= 10, "The oldest sequence should be at least 10"


def test_topic_filtering(publisher_config):
    """
    Test that a subscriber only receives messages matching its topic filter
    """
    publisher_config.replay_endpoint = None

    cfg = publisher_config.model_copy()
    cfg.topic = "foo"
    pub = EventPublisherFactory.create(cfg)

    from .conftest import MockSubscriber
    sub_foo = MockSubscriber(cfg.endpoint, None, "foo")
    sub_bar = MockSubscriber(cfg.endpoint, None, "bar")

    try:
        time.sleep(0.1)

        for _ in range(3):
            pub.publish(create_test_events(1))

        foo_received = [sub_foo.receive_one(timeout=200) for _ in range(3)]
        assert all(msg is not None for msg in foo_received), (
            "Subscriber with matching topic should receive messages")

        bar_received = [sub_bar.receive_one(timeout=200) for _ in range(3)]
        assert all(msg is None for msg in bar_received), (
            "Subscriber with non-matching topic should receive no messages")
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

    assert len(received) >= num_batches * 0.9, (
        "We should have received most messages")

    seqs = [seq for seq, _ in received]
    assert sorted(seqs) == seqs, "Sequence numbers should be in order"


def test_null_publisher():
    """Test that NullEventPublisher can be used without errors"""
    publisher = NullEventPublisher()

    # This should not raise any errors
    batch = create_test_events(5)
    publisher.publish(batch)
    publisher.shutdown()
