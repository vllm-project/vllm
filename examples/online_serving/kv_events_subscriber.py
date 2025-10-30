# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import msgspec
import zmq
from msgspec.msgpack import Decoder

from vllm.v1.core.kv_cache_utils import ExternalBlockHash


#
# Types copied from vllm.distributed.kv_events
#
class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    ts: float
    events: list[Any]


class KVCacheEvent(
    msgspec.Struct, array_like=True, omit_defaults=True, gc=False, tag=True
):
    """Base class for all KV cache-related events"""


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: list[BlockStored | BlockRemoved | AllBlocksCleared]


def process_event(event_batch):
    def _format_hash(bh):
        if bh is None:
            return None
        if isinstance(bh, bytes):
            return f"{bh.hex()[:16]}..."
        if isinstance(bh, int):
            return f"int:{bh}"
        return str(bh)

    print(f"Received event batch at {event_batch.ts}:")
    for event in event_batch.events:
        if isinstance(event, BlockStored):
            event_type = "BlockStored"
            formatted_hashes = [_format_hash(bh) for bh in event.block_hashes]
            parent_hash = _format_hash(event.parent_block_hash)

            print(f"  - {event_type}:")
            print(f"    block_hashes: {formatted_hashes}")
            print(f"    parent_block_hash: {parent_hash}")
            token_preview = event.token_ids[:5]
            if len(event.token_ids) > 5:
                token_preview = f"{token_preview}..."
            print(f"    token_ids: {token_preview}")
            print(f"    block_size: {event.block_size}")
            print(f"    lora_id: {event.lora_id}")
            print(f"    medium: {event.medium}")

        elif isinstance(event, BlockRemoved):
            event_type = "BlockRemoved"
            formatted_hashes = [_format_hash(bh) for bh in event.block_hashes]

            print(f"  - {event_type}:")
            print(f"    block_hashes: {formatted_hashes}")
            print(f"    medium: {event.medium}")

        elif isinstance(event, AllBlocksCleared):
            event_type = "AllBlocksCleared"
            print(f"  - {event_type}: All KV cache blocks cleared")

        else:
            event_type = event.__class__.__name__
            # For unknown types, try safe attribute access
            try:
                attrs = {}
                for field_name in event.__struct_fields__:
                    value = getattr(event, field_name)
                    attrs[field_name] = value
                print(f"  - {event_type}: {attrs}")
            except Exception:
                print(f"  - {event_type}: {event}")


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
                    print(
                        f"Missed {missed} messages (last: {last_seq}, current: {seq})"
                    )

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
