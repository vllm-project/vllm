# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the KV cache event wire format that the Rust frontend decodes.

The Rust frontend decodes the batches `ZmqEventPublisher` publishes (see
`rust/src/engine-core-client/src/protocol/kv_events.rs`) and re-exports them
over gRPC. Its unit tests decode the payloads in `VECTORS` and check the
decoder against the fields listed there; the tests here check that the
engine's event structs still have those fields and still encode to those
payloads, so a change to the events fails here instead of silently splitting
the two apart.

If the change is intended, give the new field or event a value in `_batches`,
regenerate the vectors by running this file directly, then update the Rust
decoder and the gRPC schema (`rust/proto/kv_events.proto`) until the Rust
tests pass:

    .venv/bin/python tests/v1/core/test_kv_event_vectors.py
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, get_args, get_type_hints
from unittest.mock import patch

import msgspec
import pytest
import torch

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
)
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    generate_block_hash_extra_keys,
    maybe_convert_block_hash,
    to_event_extra_keys,
)
from vllm.v1.kv_offload.base import Locality
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

REPO_ROOT = Path(__file__).resolve().parents[3]
VECTORS = (
    REPO_ROOT
    / "rust"
    / "src"
    / "engine-core-client"
    / "src"
    / "tests"
    / "kv_event_vectors.json"
)

BLOCK_SIZE = 2


def _request(**kwargs: Any) -> Request:
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="0", sampling_params=sampling_params, pooling_params=None, **kwargs
    )


def _published_extra_keys() -> list[tuple[Any, ...] | None] | None:
    """The extra keys the block pool publishes for two blocks of a request with
    every source of extra keys, then for a block of a request with none."""
    request = _request(
        prompt_token_ids=None,
        prompt_embeds=torch.zeros((2 * BLOCK_SIZE, 2)),
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="image-hash",
                # Starts in the first block and ends in the second.
                mm_position=PlaceholderRange(offset=1, length=2),
            )
        ],
        lora_request=LoRARequest(
            lora_name="adapter", lora_int_id=1, lora_path="/models/adapter"
        ),
        cache_salt="salt",
    )
    first, mm_idx = generate_block_hash_extra_keys(request, 0, BLOCK_SIZE, 0)
    second, _ = generate_block_hash_extra_keys(
        request, BLOCK_SIZE, 2 * BLOCK_SIZE, mm_idx
    )
    plain, _ = generate_block_hash_extra_keys(
        _request(prompt_token_ids=[1, 2]), 0, BLOCK_SIZE, 0
    )
    return to_event_extra_keys([first, second, plain])


def _published_hash(digest: bytes, *, as_int: bool) -> ExternalBlockHash:
    """A block hash as the engine publishes it in either hash mode."""
    mode = {"VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": "1" if as_int else "0"}
    with patch.dict(os.environ, mode):
        return maybe_convert_block_hash(BlockHash(digest))


def _batches() -> dict[str, KVEventBatch]:
    def digest(n: int) -> ExternalBlockHash:
        return _published_hash(bytes([n]) * 32, as_int=False)

    def integer(n: int) -> ExternalBlockHash:
        return _published_hash(bytes([n]) * 32, as_int=True)

    return {
        # Every event type, with every field set to a value other than its
        # default; `test_every_field_batch_is_complete` keeps it that way.
        "every_field": KVEventBatch(
            ts=1700000000.5,
            events=[
                BlockStored(
                    block_hashes=[digest(1), digest(2), digest(3)],
                    parent_block_hash=digest(0),
                    token_ids=[11, 12, 21, 22, 31, 32],
                    block_size=BLOCK_SIZE,
                    lora_id=7,
                    medium="GPU",
                    lora_name="adapter",
                    extra_keys=_published_extra_keys(),
                    group_idx=1,
                    kv_cache_spec_kind="sliding_window",
                    kv_cache_spec_sliding_window=4096,
                    locality=Locality.LOCAL.value,
                    ownership="kvcr",
                    session_id="session-1",
                    lookahead_token_ids=[[21], [31], [41]],
                ),
                BlockRemoved(
                    block_hashes=[digest(1)],
                    medium="CPU",
                    group_idx=1,
                    locality=Locality.REMOTE.value,
                    ownership="kvcr",
                ),
                AllBlocksCleared(),
            ],
            data_parallel_rank=1,
        ),
        # Only the fields without defaults, which are sent even when None.
        "required_fields_only": KVEventBatch(
            ts=0.0,
            events=[
                BlockStored(
                    block_hashes=[],
                    parent_block_hash=None,
                    token_ids=[],
                    block_size=0,
                    lora_id=None,
                    medium=None,
                    lora_name=None,
                ),
                BlockRemoved(block_hashes=[], medium=None),
            ],
        ),
        # `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` publishes the low 64 bits of a
        # hash. The LMCache connector forwards LMCache's own hashes, which can
        # be Python's signed `hash()`.
        "integer_block_hashes": KVEventBatch(
            ts=0.0,
            events=[
                BlockStored(
                    block_hashes=[integer(1), integer(255), -2],
                    parent_block_hash=integer(0),
                    token_ids=[1, 2, 3],
                    block_size=1,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                ),
                BlockRemoved(block_hashes=[integer(255), -2], medium="GPU"),
            ],
            data_parallel_rank=0,
        ),
    }


def _fields(struct: type[msgspec.Struct]) -> tuple[msgspec.structs.FieldInfo, ...]:
    return msgspec.structs.fields(struct)


def _type_name(field_type: Any) -> str:
    return field_type.__name__ if get_args(field_type) == () else repr(field_type)


def _batch_event_types() -> tuple[type[msgspec.Struct], ...]:
    (events,) = get_args(get_type_hints(KVEventBatch)["events"])
    return get_args(events)


def _engine_vectors() -> dict[str, Any]:
    return {
        # In wire order: a batch is encoded as an array.
        "batch_fields": [field.encode_name for field in _fields(KVEventBatch)],
        "event_fields": {
            event_type.__struct_config__.tag: {
                field.encode_name: {
                    "type": _type_name(field.type),
                    "required": field.required,
                }
                for field in _fields(event_type)
            }
            for event_type in _batch_event_types()
        },
        "batches": [
            {"name": name, "payload": msgspec.msgpack.encode(batch).hex()}
            for name, batch in _batches().items()
        ],
    }


def test_engine_matches_frontend_kv_event_vectors():
    assert VECTORS.is_file(), f"missing {VECTORS}"
    assert json.loads(VECTORS.read_text()) == _engine_vectors(), (
        "The engine's KV cache events no longer match the vectors that the "
        "Rust frontend's decoder is tested against. Give a new field or event "
        "a value in `_batches`, regenerate the vectors with "
        "`.venv/bin/python tests/v1/core/test_kv_event_vectors.py`, then "
        "update rust/src/engine-core-client/src/protocol/kv_events.rs and "
        "rust/proto/kv_events.proto until the Rust tests pass."
    )


def test_every_field_batch_is_complete():
    """The Rust decoder is only checked on values that a vector carries."""
    batch = _batches()["every_field"]
    assert batch.data_parallel_rank is not None
    assert {type(event) for event in batch.events} == set(_batch_event_types()), (
        "add an event of the new type to the `every_field` batch"
    )
    for event in batch.events:
        # Fields left at their default are omitted from the encoding.
        encoded = set(msgspec.to_builtins(event)) - {"type"}
        fields = {field.encode_name for field in _fields(type(event))}
        assert encoded == fields, (
            f"set {sorted(fields - encoded)} of {type(event).__name__} to a "
            "non-default value in the `every_field` batch"
        )


def _event_subclasses(event_type: type) -> Iterator[type]:
    for subclass in event_type.__subclasses__():
        yield subclass
        yield from _event_subclasses(subclass)


def test_batches_declare_every_event_type():
    """The Rust decoder fails on an event type it does not know, and learns
    them from the `KVEventBatch.events` union."""
    published = {
        event_type
        for event_type in _event_subclasses(KVCacheEvent)
        if event_type.__module__.startswith("vllm.")
    }
    assert published == set(_batch_event_types())


def test_optional_event_fields_default_to_none():
    """The Rust decoder reads a field the engine omits as None."""
    for event_type in _batch_event_types():
        for field in _fields(event_type):
            assert field.required or field.default is None, (
                f"{event_type.__name__}.{field.name} defaults to "
                f"{field.default!r}: decode an omitted "
                f"`{field.encode_name}` as that value in "
                "rust/src/engine-core-client/src/protocol/kv_events.rs"
            )


if __name__ == "__main__":
    VECTORS.write_text(json.dumps(_engine_vectors(), indent=2) + "\n")
    print(f"wrote {VECTORS}")
