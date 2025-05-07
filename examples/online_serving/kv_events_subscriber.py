# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Union

import msgspec
import zmq
from msgspec.msgpack import Decoder


#
# Types copied from vllm.distributed.kv_events
#
class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True,
                 gc=False):
    ts: float
    events: list[Any]


class KVCacheEvent(msgspec.Struct,
                   array_like=True,
                   omit_defaults=True,
                   gc=False,
                   tag=True):
    """Base class for all KV cache-related events"""


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]


def process_event(event_batch):
    print(f"Received event batch at {event_batch.ts}:")
    for event in event_batch.events:
        print(f"  - {event}")


def main():
    decoder = Decoder(type=KVEventBatch)
    last_seq = -1

    context = zmq.Context()

    # Set up the main subscription socket
    sub = context.socket(zmq.SUB)
    sub.connect("tcp://localhost:5557")
    topic = "kv-events"
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    # Initialize replay socket
    replay = context.socket(zmq.REQ)
    replay.connect("tcp://localhost:5558")
    poller = zmq.Poller()
    poller.register(replay, zmq.POLLIN)

    print("Listening for KV cache events on topic:", topic)

    while True:
        try:
            if sub.poll(50):
                _, seq_bytes, payload = sub.recv_multipart()
                seq = int.from_bytes(seq_bytes, "big")

                if last_seq >= 0 and seq > last_seq + 1:
                    missed = seq - last_seq - 1
                    print(f"Missed {missed} messages"
                          f" (last: {last_seq}, current: {seq})")

                    replay.send((last_seq + 1).to_bytes(8, "big"))

                    while poller.poll(timeout=200):
                        seq_bytes, replay_payload = replay.recv_multipart()
                        if not replay_payload:
                            # End of replay marker is sent as an empty frame
                            # for the payload
                            break

                        replay_seq = int.from_bytes(seq_bytes, "big")

                        if replay_seq > last_seq:
                            event_batch = decoder.decode(replay_payload)
                            process_event(event_batch)
                            last_seq = replay_seq
                            if replay_seq >= seq - 1:
                                break

                event_batch = decoder.decode(payload)
                process_event(event_batch)

            # ... do other periodic work or check for shutdown ...

        except KeyboardInterrupt:
            print("Interrupted")
            break
        except Exception as e:
            print("Error decoding message:", e)


if __name__ == "__main__":
    main()
