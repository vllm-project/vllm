# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-engine tier of the KV-cache key-partitioning conformance suite.

Invariant (RFC #53194): if two requests differ in any dimension that must
partition the KV-cache keyspace, they must not reuse each other's blocks;
if they agree, they must. Every negative arm has a positive control, and
assertions are on what ``KVCacheManager`` hands back plus block-hash
equality, never on how the dimension is encoded into the hash.
"""

import pytest

from vllm.lora.request import LoRARequest
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.request import Request

from .dimensions import (
    BLOCK_SIZE,
    DIMENSIONS,
    NUM_FULL_BLOCKS,
    Dimension,
    make_kv_cache_manager,
    make_request,
    negative_params,
)

FULL_PREFIX = NUM_FULL_BLOCKS * BLOCK_SIZE


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def warm(manager: KVCacheManager, request: Request) -> None:
    """Run ``request`` through lookup and allocation so its blocks get cached."""
    computed, num_computed, _ = manager.get_computed_blocks(request)
    assert num_computed == 0, "warm request must start cold"
    blocks = manager.allocate_slots(
        request, request.num_prompt_tokens, num_computed, computed
    )
    assert blocks is not None
    request.num_computed_tokens = request.num_prompt_tokens


def lookup(manager: KVCacheManager, request: Request) -> int:
    """Number of prompt tokens the prefix cache would serve for ``request``."""
    _, num_computed, _ = manager.get_computed_blocks(request)
    return num_computed


@pytest.mark.parametrize("dim", DIMENSIONS, ids=lambda d: d.name)
def test_same_value_reuses_blocks(dim: Dimension):
    manager = make_kv_cache_manager()
    first = make_request("first", **dim.build("x"))
    second = make_request("second", **dim.build("x"))
    assert first.block_hashes == second.block_hashes
    warm(manager, first)
    assert lookup(manager, second) == FULL_PREFIX


@pytest.mark.parametrize("dim", negative_params())
def test_different_value_never_reuses_blocks(dim: Dimension):
    manager = make_kv_cache_manager()
    first = make_request("first", **dim.build("x"))
    other = make_request("other", **dim.build("y"))
    assert all(a != b for a, b in zip(first.block_hashes, other.block_hashes))
    warm(manager, first)
    assert lookup(manager, other) == 0


@pytest.mark.parametrize("dim", DIMENSIONS, ids=lambda d: d.name)
def test_unset_value_never_reuses_set_value(dim: Dimension):
    manager = make_kv_cache_manager()
    plain = make_request("plain")
    tagged = make_request("tagged", **dim.build("x"))
    warm(manager, plain)
    assert lookup(manager, tagged) == 0
    warm(manager, tagged)
    assert lookup(manager, make_request("plain2")) == FULL_PREFIX


@pytest.mark.xfail(
    strict=True,
    reason="#44701: cache_salt and lora_name are hashed without domain "
    "separation, so equal strings collide. Fixed by #51899.",
)
def test_cross_dimension_equal_strings_never_reuse_blocks():
    manager = make_kv_cache_manager()
    salted = make_request("salted", cache_salt="shared")
    lora = make_request(
        "lora",
        lora_request=LoRARequest(
            lora_name="shared", lora_int_id=7, lora_path="/nonexistent"
        ),
    )
    warm(manager, salted)
    assert lookup(manager, lora) == 0
    assert salted.block_hashes[0] != lora.block_hashes[0]
