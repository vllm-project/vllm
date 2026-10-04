# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-affinity tie-breaking in the internal DP load balancer.

#47420 made tied load-balancing decisions rotate across engines to fix a
fairness bias, which also destroyed prefix-cache locality: consecutive turns
of one conversation rotate across DP ranks, and every turn misses the prefix
the previous turn left in KV cache. Measured on DeepSeek-V4-Flash (DP=4, a
multi-turn trace designed for 81.7% prefix reuse), the hit rate fell from
81.6% (v0.23.0, pre-rotation) to 8-53%.

The fix: remember which engine served a prompt's first-block prefix and
prefer that engine when scores tie. Load-aware decisions and the fairness
rotation for unrelated requests are unchanged. These tests drive
``get_core_engine_for_request`` directly with synthetic state; no engines or
GPUs involved.
"""

from collections import Counter, OrderedDict
from types import SimpleNamespace

from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import PREFIX_AFFINITY_LRU_SIZE, DPLBAsyncMPClient

BLOCK_SIZE = 16


def _make_client(num_engines: int = 4, client_count: int = 1, client_index: int = 0):
    """A DPLBAsyncMPClient with just the state the LB decision reads."""
    client = object.__new__(DPLBAsyncMPClient)
    client.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE)
    )
    client.core_engines = [bytes([i]) for i in range(num_engines)]
    client.lb_engines = [[0, 0, 0.0] for _ in range(num_engines)]
    client.client_count = client_count
    client.client_index = client_index
    client.eng_start_index = (num_engines * client_index) // client_count
    client.engine_inflight = Counter()
    client.reqs_in_flight = {}
    client.prefix_affinity = OrderedDict()
    return client


def _request(
    request_id: str,
    prompt_token_ids: list[int] | None,
    cache_salt: str | None = None,
    data_parallel_rank: int | None = None,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=cache_salt,
        data_parallel_rank=data_parallel_rank,
    )


def _finish(client: DPLBAsyncMPClient, request_id: str) -> None:
    """Mark a request finished and refresh idle coordinator stats."""
    engine = client.reqs_in_flight.pop(request_id)
    client.engine_inflight[engine] -= 1
    client.lb_engines = [[0, 0, 0.0] for _ in client.core_engines]


def _engine_index(client: DPLBAsyncMPClient, engine: bytes) -> int:
    return client.core_engines.index(engine)


def test_same_prefix_sticks_across_sequential_turns():
    """Turns of one conversation keep landing on the same engine on ties."""
    client = _make_client()
    prompt = list(range(BLOCK_SIZE * 2))

    first = client.get_core_engine_for_request(_request("t0", prompt))
    _finish(client, "t0")

    # Each later turn extends the conversation; the first block is unchanged.
    for turn in range(1, 6):
        grown = prompt + list(range(BLOCK_SIZE * 2, BLOCK_SIZE * 2 + 8 * turn))
        chosen = client.get_core_engine_for_request(_request(f"t{turn}", grown))
        assert chosen is first, f"turn {turn} left its prefix's engine"
        _finish(client, f"t{turn}")


def test_unrelated_tied_requests_still_rotate():
    """#47420's fairness is preserved for requests with distinct prefixes."""
    client = _make_client()
    chosen = []
    for i in range(4):
        prompt = [1000 * (i + 1) + t for t in range(BLOCK_SIZE)]
        engine = client.get_core_engine_for_request(_request(f"r{i}", prompt))
        chosen.append(engine)
        _finish(client, f"r{i}")
    assert len(set(chosen)) == len(client.core_engines), (
        "tied unrelated requests should spread across all engines"
    )


def test_load_beats_affinity():
    """A loaded affinity engine loses to an idle one; affinity re-learns."""
    client = _make_client()
    prompt = list(range(BLOCK_SIZE))

    first = client.get_core_engine_for_request(_request("a0", prompt))
    _finish(client, "a0")
    first_index = _engine_index(client, first)

    # Pile synthetic load onto the affinity engine.
    client.lb_engines[first_index] = [5, 5, 0.0]

    second = client.get_core_engine_for_request(_request("a1", prompt))
    assert second is not first, "load-aware routing must beat affinity"
    # The mapping now points at the engine that actually holds the newer
    # prefix instance.
    key = client._prefix_affinity_key(_request("probe", prompt))
    assert client.prefix_affinity[key] == _engine_index(client, second)


def test_sub_block_prompt_has_no_affinity():
    """Prompts shorter than one KV block cannot hit the cache: no key."""
    client = _make_client()
    short = list(range(BLOCK_SIZE - 1))
    assert client._prefix_affinity_key(_request("s0", short)) is None

    client.get_core_engine_for_request(_request("s0", short))
    assert not client.prefix_affinity


def test_cache_salt_separates_affinity():
    """Same tokens, different cache_salt: distinct keys (salted requests
    cannot share cache entries)."""
    client = _make_client()
    prompt = list(range(BLOCK_SIZE))
    key_a = client._prefix_affinity_key(_request("x", prompt, cache_salt="a"))
    key_b = client._prefix_affinity_key(_request("x", prompt, cache_salt="b"))
    assert key_a != key_b


def test_explicit_data_parallel_rank_bypasses_affinity():
    """A pinned data_parallel_rank is honored verbatim."""
    client = _make_client()
    prompt = list(range(BLOCK_SIZE))
    client.prefix_affinity[client._prefix_affinity_key(_request("p", prompt))] = 0

    chosen = client.get_core_engine_for_request(
        _request("p", prompt, data_parallel_rank=2)
    )
    assert chosen is client.core_engines[2]


def test_stale_affinity_after_scale_in_is_ignored():
    """An affinity index beyond the current engine count is discarded."""
    client = _make_client(num_engines=2)
    prompt = list(range(BLOCK_SIZE))
    key = client._prefix_affinity_key(_request("z", prompt))
    client.prefix_affinity[key] = 7  # recorded before a scale-in

    chosen = client.get_core_engine_for_request(_request("z", prompt))
    assert chosen in client.core_engines
    assert client.prefix_affinity[key] < 2


def test_affinity_lru_is_bounded():
    client = _make_client()
    for i in range(PREFIX_AFFINITY_LRU_SIZE + 10):
        prompt = [i * BLOCK_SIZE + t for t in range(BLOCK_SIZE)]
        client.get_core_engine_for_request(_request(f"lru{i}", prompt))
        _finish(client, f"lru{i}")
    assert len(client.prefix_affinity) == PREFIX_AFFINITY_LRU_SIZE
