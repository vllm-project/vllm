# SPDX-License-Identifier: Apache-2.0
import random
from typing import Optional, Union

import msgspec
import msgspec.msgpack
import pytest
import zmq

from vllm.config import KVEventsConfig
from vllm.distributed.kv_events import EventPublisherFactory

from .test_events import SampleBatch


@pytest.fixture
def random_port():
    """Generate a random port number for testing"""
    return random.randint(10000, 60000)


@pytest.fixture
def publisher_config(random_port, request):
    """Create a publisher config with inproc transport"""
    how = request.param if hasattr(request, "param") else "inproc"

    if how == "inproc":
        endpoint = f"inproc://test-{random_port}"
        replay_endpoint = endpoint + "-replay"
    else:
        endpoint = f"tcp://*:{random_port}"
        replay_endpoint = f"tcp://*:{random_port + 1}"

    return KVEventsConfig(enable_kv_cache_events=True,
                          publisher="zmq",
                          endpoint=endpoint,
                          replay_endpoint=replay_endpoint,
                          buffer_steps=100,
                          hwm=1000,
                          topic="test")


@pytest.fixture
def publisher(publisher_config):
    """Create and return a publisher instance"""
    pub = EventPublisherFactory.create(publisher_config)
    yield pub
    pub.shutdown()


@pytest.fixture
def subscriber(publisher_config):
    """Create and return a subscriber for testing"""
    endpoint = publisher_config.endpoint
    replay_endpoint = publisher_config.replay_endpoint

    if endpoint.startswith("tcp://*"):
        endpoint = endpoint.replace("*", "127.0.0.1")
    if replay_endpoint and replay_endpoint.startswith("tcp://*"):
        replay_endpoint = replay_endpoint.replace("*", "127.0.0.1")

    sub = MockSubscriber(endpoint, replay_endpoint, publisher_config.topic)
    yield sub
    sub.close()


class MockSubscriber:
    """Helper class to receive and verify published events"""

    def __init__(self,
                 pub_endpoint: str,
                 replay_endpoint: Optional[str] = None,
                 topic: str = "",
                 decode_type=SampleBatch):
        self.ctx = zmq.Context.instance()

        # Set up subscriber socket
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
        self.sub.connect(pub_endpoint)

        # Set up replay socket if provided
        self.replay = None
        if replay_endpoint:
            self.replay = self.ctx.socket(zmq.REQ)
            self.replay.connect(replay_endpoint)

        self.topic = topic
        self.topic_bytes = topic.encode('utf-8')
        self.received_msgs: list[tuple[int, SampleBatch]] = []
        self.last_seq = -1
        self.decoder = msgspec.msgpack.Decoder(type=decode_type)

    def receive_one(self,
                    timeout=1000) -> Union[tuple[int, SampleBatch], None]:
        """Receive a single message with timeout"""
        if not self.sub.poll(timeout):
            return None

        topic_bytes, seq_bytes, payload = self.sub.recv_multipart()
        assert topic_bytes == self.topic_bytes

        seq = int.from_bytes(seq_bytes, "big")
        data = self.decoder.decode(payload)
        self.last_seq = seq
        self.received_msgs.append((seq, data))
        return seq, data

    def request_replay(self, start_seq: int) -> None:
        """Request replay of messages starting from start_seq"""
        if not self.replay:
            raise ValueError("Replay socket not initialized")

        self.replay.send(start_seq.to_bytes(8, "big"))

    def receive_replay(self) -> list[tuple[int, SampleBatch]]:
        """Receive replayed messages"""
        if not self.replay:
            raise ValueError("Replay socket not initialized")

        replayed: list[tuple[int, SampleBatch]] = []
        while True:
            try:
                if not self.replay.poll(1000):
                    break

                frames = self.replay.recv_multipart()
                if not frames or not frames[-1]:
                    # End of replay marker
                    break

                seq_bytes, payload = frames
                seq = int.from_bytes(seq_bytes, "big")
                data = self.decoder.decode(payload)
                replayed.append((seq, data))
            except zmq.ZMQError as _:
                break

        return replayed

    def close(self):
        """Clean up resources"""
        self.sub.close()
        if self.replay:
            self.replay.close()
