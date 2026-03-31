# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_offload.hashing import HybridChunkBlockHashList, RequestBlockHashList
from vllm.v1.request import Request


def make_request(num_tokens: int, block_size: int = 16) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="r0",
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def test_request_block_hash_list_matches_request_hashes_when_sizes_match():
    request = make_request(64, block_size=16)
    direct_hashes = list(RequestBlockHashList(request, 16, sha256))

    assert direct_hashes == request.block_hashes


def test_request_block_hash_list_supports_arbitrary_block_sizes():
    request = make_request(65536, block_size=1056)
    direct_hashes = RequestBlockHashList(request, 16384, sha256)

    assert len(direct_hashes) == 4
    assert direct_hashes[0] != direct_hashes[1]


def test_hybrid_chunk_block_hash_list_uses_per_group_granularity():
    request = make_request(65536, block_size=1056)
    hash_list = HybridChunkBlockHashList(
        request,
        group_block_sizes=(16384, 16384, 16384, 1056),
        logical_chunk_size=16384,
        hash_function=sha256,
    )

    assert len(hash_list) == 4
    assert hash_list[0] != hash_list[1]


def test_hybrid_chunk_block_hash_list_caches_chunk_hashes():
    """Accessing the same index twice should return the cached value."""
    request = make_request(65536, block_size=1056)
    hash_list = HybridChunkBlockHashList(
        request,
        group_block_sizes=(16384, 1056),
        logical_chunk_size=16384,
        hash_function=sha256,
    )

    # Cache starts empty
    assert len(hash_list._chunk_hashes) == 0

    # Access index 0: should populate the cache
    h0 = hash_list[0]
    assert len(hash_list._chunk_hashes) == 1
    assert hash_list._chunk_hashes[0] == h0

    # Access index 1: cache grows
    h1 = hash_list[1]
    assert len(hash_list._chunk_hashes) == 2

    # Re-access index 0: served from cache, identical value
    assert hash_list[0] == h0

    # Re-access index 1: served from cache
    assert hash_list[1] == h1

    # Cache does not grow on repeated access
    assert len(hash_list._chunk_hashes) == 2


def test_hybrid_chunk_block_hash_list_skips_leading_unhashable_chunks():
    request = make_request(100000, block_size=1056)
    hash_list = HybridChunkBlockHashList(
        request,
        group_block_sizes=(50000, 16384, 1056),
        logical_chunk_size=16384,
        hash_function=sha256,
    )

    assert hash_list.first_hashable_chunk_idx == 3
    assert len(hash_list) == 3
