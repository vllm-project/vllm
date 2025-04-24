# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Union

import msgspec
import zmq


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
    num_toks_per_block: list[int]
    lora_id: Optional[int]


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]


decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5557")
topic = "kv-events"
socket.setsockopt_string(zmq.SUBSCRIBE, topic)

print("Listening for KV cache events on topic:", topic)

while True:
    try:
        _, seq_bytes, payload = socket.recv_multipart()
        seq = int.from_bytes(seq_bytes, "big")
        event_batch = decoder.decode(payload)
        print(f"Received event batch at {event_batch.ts}:")
        for event in event_batch.events:
            print(f"  - {event}")
    except KeyboardInterrupt:
        print("Interrupted")
        break
    except Exception as e:
        print("Error decoding message:", e)
