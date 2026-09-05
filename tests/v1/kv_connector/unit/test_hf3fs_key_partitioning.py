# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every dimension that must partition the KV cache keyspace has to reach the
HF3FS external cache keys.

The engine computes one canonical block hash per full block, folding in multimodal
identity, LoRA identity, ``cache_salt`` and prompt-embeds content, and it builds that
hasher whenever a KV connector is configured -- see ``EngineCore.__init__``. An external
connector therefore never has to re-derive the key. A key derived from the token ids
alone collides across every one of those dimensions, and the external store then serves
one request's blocks to another.
"""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.common import (
    HF3FSRequestMetadata,
    RequestSchedulingState,
    external_block_keys,
)
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

@pytest.fixture(autouse=True, scope="module")
def _none_hash():
    """Seed NONE_HASH once for the module. Re-seeding between requests would give
    every request its own parent hash and quietly defeat the comparisons below."""
    init_none_hash(sha256)


BLOCK_SIZE = 16
# Two full blocks. Multimodal placeholder tokens are identical whatever the image is,
# which is exactly why the image hash has to be in the key.
TOKENS = list(range(1000, 1000 + BLOCK_SIZE * 2))


def _make_request(
    request_id: str,
    token_ids: list[int] | None = None,
    mm_hash: str | None = None,
    cache_salt: str | None = None,
    lora_request: LoRARequest | None = None,
) -> Request:
    token_ids = TOKENS if token_ids is None else token_ids
    mm_features = None
    if mm_hash is not None:
        mm_features = [
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=PlaceholderRange(offset=0, length=len(token_ids)),
                identifier=mm_hash,
                modality="image",
            )
        ]
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(token_ids),
        mm_features=mm_features,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=lora_request,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _keys(request, hash_block_size: int = BLOCK_SIZE, block_size: int = BLOCK_SIZE):
    """Keys at the connector's own block size, the way the connector asks for them."""
    return external_block_keys(request, hash_block_size, block_size)


MUST_PARTITION = {
    "multimodal_content": (
        {"mm_hash": "hash-of-image-A"},
        {"mm_hash": "hash-of-image-B"},
    ),
    "lora_identity": (
        {"lora_request": LoRARequest("adapter-A", 1, "/tmp/adapter-a")},
        {"lora_request": LoRARequest("adapter-B", 2, "/tmp/adapter-b")},
    ),
    "cache_salt": ({"cache_salt": "tenant-1"}, {"cache_salt": "tenant-2"}),
    "prompt_tokens": ({}, {"token_ids": [t + 1 for t in TOKENS]}),
}


@pytest.mark.parametrize("dimension", sorted(MUST_PARTITION))
def test_external_keys_partition_on(dimension: str):
    """Two requests differing only in one such dimension must not share keys."""
    kwargs_a, kwargs_b = MUST_PARTITION[dimension]
    keys_a = _keys(_make_request("a", **kwargs_a))
    keys_b = _keys(_make_request("b", **kwargs_b))

    assert keys_a and keys_b, "expected one key per full block"
    assert len(keys_a) == len(keys_b) == len(TOKENS) // BLOCK_SIZE
    assert keys_a != keys_b, (
        f"{dimension} does not reach the external cache key: both requests map to "
        f"{keys_a[0]}, so the external store would serve one request's KV blocks "
        f"to the other"
    )


def test_external_keys_are_shared_when_nothing_differs():
    """The control: without a distinguishing dimension, sharing is the point."""
    keys_a = _keys(_make_request("a"))
    keys_b = _keys(_make_request("b"))
    assert keys_a == keys_b


def test_external_keys_are_prefix_chained():
    """A differing second block must not disturb the first block's key, or every
    request would miss on a prefix it legitimately shares."""
    long_tokens = TOKENS + list(range(9000, 9000 + BLOCK_SIZE))
    shared = _keys(_make_request("a"))
    extended = _keys(_make_request("b", token_ids=long_tokens))
    assert extended[: len(shared)] == shared
    assert len(extended) == len(shared) + 1


@pytest.mark.parametrize("dimension", sorted(MUST_PARTITION))
def test_request_metadata_carries_partitioned_keys(dimension: str):
    """The keys the worker actually uses come from the connector metadata, so the
    partitioning has to survive that hop as well."""
    kwargs_a, kwargs_b = MUST_PARTITION[dimension]

    def build(request_id: str, **kwargs) -> HF3FSRequestMetadata:
        request = _make_request(request_id, **kwargs)
        state = RequestSchedulingState(request_id=request_id, request=request)
        state.token_ids = list(request.prompt_token_ids)
        state.allocated_block_ids = list(range(len(TOKENS) // BLOCK_SIZE))
        metadata = HF3FSRequestMetadata.from_scheduling_state(
            state, BLOCK_SIZE, BLOCK_SIZE
        )
        assert metadata is not None
        return metadata

    meta_a = build("a", **kwargs_a)
    meta_b = build("b", **kwargs_b)

    assert len(meta_a.block_keys) == len(meta_a.block_ids)
    assert meta_a.block_keys != meta_b.block_keys, (
        f"{dimension} is dropped between the scheduler and the worker"
    )


def test_a_coarser_block_size_takes_the_strided_view():
    """Block ids are indexed at the scheduler block size while ``block_hashes`` are
    computed at the hash block size; for hybrid models and dcp > 1 those differ, and
    pairing them one-to-one would misalign every key with its block.

    Each hash is chained over its whole prefix, so the coarse key for a block is the
    hash at that block's last fine boundary -- not the first, and not a re-hash.
    """
    request = _make_request("a", token_ids=list(range(2000, 2000 + BLOCK_SIZE * 4)))
    fine = _keys(request, BLOCK_SIZE, BLOCK_SIZE)
    coarse = _keys(request, BLOCK_SIZE, BLOCK_SIZE * 2)

    assert len(fine) == 4
    assert len(coarse) == 2
    assert coarse == [fine[1], fine[3]]


def test_the_partitioning_survives_a_coarser_block_size():
    """The dimensions must still separate once the view is strided."""
    kwargs_a, kwargs_b = MUST_PARTITION["multimodal_content"]
    long_a = _make_request("a", token_ids=list(range(2000, 2000 + BLOCK_SIZE * 4)),
                           **kwargs_a)
    long_b = _make_request("b", token_ids=list(range(2000, 2000 + BLOCK_SIZE * 4)),
                           **kwargs_b)
    assert _keys(long_a, BLOCK_SIZE, BLOCK_SIZE * 2) != _keys(
        long_b, BLOCK_SIZE, BLOCK_SIZE * 2
    )
