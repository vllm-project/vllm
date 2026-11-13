# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prefix graph of recovery units and the keys of frontier checkpoints."""

import pytest

from tests.v1.core.test_prefix_caching import make_request
from vllm.lora.request import LoRARequest
from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_checkpoint.keys import frontier_key
from vllm.v1.core.hybrid_checkpoint.units import CheckpointGraph
from vllm.v1.core.kv_cache_utils import init_none_hash

pytestmark = pytest.mark.cpu_test

BLOCK = 16


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def test_removed_unit_hands_its_children_to_its_parent():
    graph = CheckpointGraph(BLOCK)
    for key, num_tokens in (("a", 16), ("b", 32), ("c", 40)):
        graph.inserted(key, num_tokens)
    graph.admitted("r1", "a")
    graph.link("r1", "b")
    graph.link("r1", "c")
    graph.admitted("r2", "b")
    graph.leaves.add("b")
    assert graph.units["c"].parent == "b"
    assert graph.units["c"].frontier
    assert not graph.units["b"].frontier

    graph.removed("b")

    assert graph.units["c"].parent == "a"
    assert graph.units["a"].children == {"c"}
    assert graph.last["r1"] == "c"
    assert graph.resumed["r2"] is None
    assert "b" not in graph.leaves


def test_branch_point_is_a_second_child_or_a_second_hit():
    graph = CheckpointGraph(BLOCK)
    for key, num_tokens in (("root", 16), ("x", 32), ("y", 48)):
        graph.inserted(key, num_tokens)
    graph.admitted("r1", "root")
    graph.link("r1", "x")
    graph.touched("root")
    assert not graph.units["root"].is_branch_point

    graph.admitted("r2", "root")
    graph.link("r2", "y")
    assert graph.units["root"].is_branch_point

    graph.touched("x")
    graph.touched("x")
    assert graph.units["x"].is_branch_point


def test_frontier_key_identifies_tokens_length_and_extra_keys():
    tokens = list(range(40))
    request = make_request("a", tokens, BLOCK, sha256)
    key = frontier_key(sha256, request, 37, BLOCK)

    same = make_request("b", tokens, BLOCK, sha256)
    assert frontier_key(sha256, same, 37, BLOCK) == key
    assert frontier_key(sha256, request, 36, BLOCK) != key
    changed = make_request("c", [*tokens[:36], 999, *tokens[37:]], BLOCK, sha256)
    assert frontier_key(sha256, changed, 37, BLOCK) != key
    assert frontier_key(sha256, changed, 37, BLOCK, last_token=36) == key
    salted = make_request("d", tokens, BLOCK, sha256, cache_salt="salt")
    assert frontier_key(sha256, salted, 37, BLOCK) != key
    adapter = LoRARequest("adapter", 1, "/adapter")
    lora = make_request("e", tokens, BLOCK, sha256, lora_request=adapter)
    assert frontier_key(sha256, lora, 37, BLOCK) != key
    # On a block boundary the key still differs from the block hash there.
    assert frontier_key(sha256, request, 32, BLOCK) != request.block_hashes[1]
