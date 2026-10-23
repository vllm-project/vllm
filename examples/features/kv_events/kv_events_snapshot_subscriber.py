# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consume KV snapshots and resynchronize after gaps or publisher restarts."""

import argparse
import time

import msgspec
import zmq


class ResyncRequired(RuntimeError):
    pass


class _Identity(msgspec.Struct, array_like=True):
    """The trailing publisher identity of a live batch; events stay encoded."""

    ts: float
    events: msgspec.Raw
    data_parallel_rank: int | None = None
    publisher_id: bytes | None = None


_identity = msgspec.msgpack.Decoder(_Identity)


class SnapshotClient:
    """One publisher's snapshot/live transport; use from a single thread.

    ``ready`` describes transport continuity. Callers must build and install a
    private index before advertising it to their own request-serving path.
    """

    MAX_BUFFER_BYTES = 64 * 1024 * 1024

    def __init__(self, live_endpoint, snapshot_endpoint, topic=b""):
        self.context = zmq.Context.instance()
        self.sub = self.context.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, topic)
        self.sub.connect(live_endpoint)
        self.endpoint = snapshot_endpoint
        self.ready = False
        self.publisher_id = None
        self.next_seq = None
        self.last_receive = time.monotonic()

    def _read(self):
        frames = self.sub.recv_multipart()
        self.last_receive = time.monotonic()
        if len(frames) != 3 or len(frames[1]) != 8:
            self.ready = False
            raise ResyncRequired("malformed live message")
        _, sequence, payload = frames
        publisher_id = _identity.decode(payload).publisher_id
        if publisher_id is None:
            self.ready = False
            raise ResyncRequired("publisher does not serve snapshots")
        return publisher_id, int.from_bytes(sequence, "big"), payload

    def bootstrap(self, timeout=10):
        """Return snapshot chunks plus a contiguous buffered suffix.

        Build a private index from these payloads and only then expose it. On
        any exception the source stays unready. Retries use a fresh REQ socket.
        """
        self.ready = False
        deadline = time.monotonic() + timeout
        # Seeing a live message establishes subscription delivery. An idle
        # publisher emits a heartbeat, so this does not require inference.
        if not self.sub.poll(max(0, int((deadline - time.monotonic()) * 1000))):
            raise ResyncRequired("live subscription timed out")
        buffered = [self._read()]
        size = len(buffered[0][2])
        if size > self.MAX_BUFFER_BYTES:
            raise ResyncRequired("bootstrap buffer exhausted")
        with self.context.socket(zmq.REQ) as req:
            req.setsockopt(zmq.LINGER, 0)
            req.connect(self.endpoint)
            req.send(b"snapshot")
            poller = zmq.Poller()
            poller.register(req, zmq.POLLIN)
            poller.register(self.sub, zmq.POLLIN)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ResyncRequired("snapshot timed out")
                available = dict(poller.poll(max(1, int(remaining * 1000))))
                if self.sub in available:
                    message = self._read()
                    size += len(message[2])
                    if size > self.MAX_BUFFER_BYTES:
                        raise ResyncRequired("bootstrap buffer exhausted")
                    buffered.append(message)
                if req in available:
                    frames = req.recv_multipart()
                    break
        if len(frames) < 2 or len(frames[0]) != 8 or len(frames[1]) != 16:
            raise ResyncRequired("invalid snapshot reply")
        seq = int.from_bytes(frames[0], "big", signed=True)
        if seq < -1:
            raise ResyncRequired("snapshot unavailable")
        publisher_id = frames[1]
        if buffered[-1][0] != publisher_id:
            raise ResyncRequired("publisher restarted during bootstrap")
        next_seq = seq + 1
        payloads = frames[2:]
        for epoch, number, payload in buffered:
            if epoch != publisher_id or number <= seq:
                continue
            if number != next_seq:
                raise ResyncRequired("gap during bootstrap")
            payloads.append(payload)
            next_seq += 1
        self.publisher_id = publisher_id
        self.next_seq = next_seq
        self.ready = True
        return seq, payloads

    def poll(self, timeout_ms=1000):
        if not self.ready:
            raise ResyncRequired("source has not bootstrapped")
        if not self.sub.poll(timeout_ms):
            if time.monotonic() - self.last_receive > 5:
                self.ready = False
                raise ResyncRequired("publisher heartbeat timed out")
            return None
        epoch, seq, payload = self._read()
        if epoch != self.publisher_id or seq > self.next_seq:
            self.ready = False
            raise ResyncRequired("publisher restart or live sequence gap")
        if seq < self.next_seq:
            return None
        self.next_seq += 1
        return payload

    def close(self):
        self.ready = False
        self.sub.close(linger=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://localhost:5557")
    parser.add_argument("--snapshot-endpoint", default="tcp://localhost:5559")
    parser.add_argument("--topic", default="kv-events")
    args = parser.parse_args()
    client = SnapshotClient(args.endpoint, args.snapshot_endpoint, args.topic.encode())
    try:
        while True:
            try:
                if not client.ready:
                    sequence, chunks = client.bootstrap()
                    print(f"Snapshot through sequence {sequence}:", flush=True)
                    for chunk in chunks:
                        print(msgspec.msgpack.decode(chunk), flush=True)
                payload = client.poll()
                if payload is not None:
                    print(msgspec.msgpack.decode(payload), flush=True)
            except ResyncRequired as error:
                print(f"Source unavailable; resynchronizing: {error}", flush=True)
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
