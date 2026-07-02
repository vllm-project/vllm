# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import msgspec
import pytest

from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreReadyResponse, EngineCoreRequest
from vllm.v1.engine.core_client import AsyncMPClient, DPLBAsyncMPClient
from vllm.v1.engine.dp_prefix_cache_router import DPPrefixCacheRouter
from vllm.v1.request import Request


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


def _make_engine_request(
    token_ids: list[int],
    *,
    request_id: str = "request",
    cache_salt: str | None = None,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=token_ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=cache_salt,
        data_parallel_rank=None,
    )


def _make_router(
    *,
    n_ranks: int = 2,
    block_size: int = 4,
    branch_block_size: int | None = None,
    shallow_depth: int = 1,
    deep_depth: int = 2,
    warm_threshold: int = 1,
    max_prefixes: int = 100_000,
) -> DPPrefixCacheRouter:
    parallel_config = ParallelConfig(
        data_parallel_size=n_ranks,
        data_parallel_prefix_cache_lb=True,
        data_parallel_prefix_cache_lb_shallow_depth=shallow_depth,
        data_parallel_prefix_cache_lb_deep_depth=deep_depth,
        data_parallel_prefix_cache_lb_warm_threshold=warm_threshold,
        data_parallel_prefix_cache_lb_max_prefixes=max_prefixes,
    )
    return DPPrefixCacheRouter(
        parallel_config,
        hash_block_size=block_size,
        branch_block_size=branch_block_size,
        hash_algo="sha256",
        n_ranks=n_ranks,
    )


def _least_loaded(counts: list[list[int]]) -> int:
    return min(range(len(counts)), key=lambda r: (counts[r][0] * 4 + counts[r][1], r))


def test_router_hashes_match_request_block_hashes():
    token_ids = list(range(12))
    cache_salt = "tenant-a"
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    block_hasher = get_request_block_hasher(4, hash_fn)

    request = Request(
        request_id="request",
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=block_hasher,
    )
    router = _make_router(block_size=4)
    engine_request = _make_engine_request(token_ids, cache_salt=cache_salt)

    assert router._prompt_signatures(engine_request) == request.block_hashes


def test_router_reuses_sticky_prefix_and_sets_branch_hint():
    router = _make_router(block_size=2)
    first = _make_engine_request([1, 2, 3, 4, 5, 6], request_id="first")
    second = _make_engine_request([1, 2, 3, 4, 5, 6, 7, 8], request_id="second")
    counts = [[0, 0], [0, 0]]

    first_decision = router.route(first, counts, _least_loaded)
    second_decision = router.route(second, counts, _least_loaded)

    assert second_decision.rank == first_decision.rank
    assert second_decision.dp_prefix_cache_prefix_len == 4
    assert second.dp_prefix_cache_prefix_len is None


def test_router_rounds_branch_hint_to_scheduler_block_size():
    router = _make_router(block_size=2, branch_block_size=4, deep_depth=4)
    first = _make_engine_request([1, 2, 3, 4, 5, 6], request_id="first")
    second = _make_engine_request([1, 2, 3, 4, 5, 6, 7, 8], request_id="second")
    counts = [[0, 0], [0, 0]]

    router.route(first, counts, _least_loaded)
    second_decision = router.route(second, counts, _least_loaded)

    assert second_decision.dp_prefix_cache_prefix_len == 4


def test_dp_lb_client_attaches_branch_hint_to_request():
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [b"\x00\x00", b"\x01\x00"]
    client.lb_engines = [[0, 0], [0, 0]]
    client.dp_prefix_cache_router = _make_router(block_size=2)
    client.client_count = 1
    client.eng_start_index = 0
    client.reqs_in_flight = {}

    first = _make_engine_request([1, 2, 3, 4, 5, 6], request_id="first")
    second = _make_engine_request([1, 2, 3, 4, 5, 6, 7, 8], request_id="second")

    DPLBAsyncMPClient.get_core_engine_for_request(client, first)
    DPLBAsyncMPClient.get_core_engine_for_request(client, second)

    assert second.dp_prefix_cache_prefix_len == 4


def test_router_does_not_set_branch_hint_for_different_final_rank():
    router = _make_router(block_size=2)
    first = _make_engine_request([1, 2, 3, 4], request_id="first")
    second = _make_engine_request([1, 2, 3, 4, 5, 6], request_id="second")
    counts = [[0, 0], [0, 0]]

    first_decision = router.route(first, counts, _least_loaded)
    signatures = router._prompt_signatures(second)
    session_key = router._session_key(signatures)
    router.session_rank[session_key] = 1 - first_decision.rank

    second_decision = router.route(second, counts, _least_loaded)

    assert second_decision.rank != first_decision.rank
    assert second_decision.dp_prefix_cache_prefix_len is None


def test_router_caps_branch_hint_at_deep_depth():
    router = _make_router(block_size=2, shallow_depth=2, deep_depth=4)
    counts = [[0, 0], [0, 0]]

    three_block = _make_engine_request([1, 2, 3, 4, 5, 6], request_id="three")
    four_block = _make_engine_request([1, 2, 3, 4, 5, 6, 7, 8], request_id="four")
    router.route(three_block, counts, _least_loaded)
    decision = router.route(four_block, counts, _least_loaded)
    assert decision.dp_prefix_cache_prefix_len == 6

    known = _make_engine_request(
        list(range(20)),
        request_id="known",
    )
    sibling = _make_engine_request(
        list(range(22)),
        request_id="sibling",
    )
    router.route(known, counts, _least_loaded)
    decision = router.route(sibling, counts, _least_loaded)
    assert decision.dp_prefix_cache_prefix_len == 8
    assert max(len(prefix) for prefix in router.prefix_stats) <= 4


def test_router_bounds_prefix_tracking_state():
    router = _make_router(block_size=1, max_prefixes=3)
    counts = [[0, 0], [0, 0]]

    for i in range(8):
        router.route(_make_engine_request([i, i + 100]), counts, _least_loaded)

    assert len(router.session_rank) <= 3
    assert len(router.prefix_stats) <= 3


def test_router_reroutes_cold_miss_to_less_loaded_rank():
    router = _make_router(block_size=2)
    request = _make_engine_request([10, 11, 12, 13], request_id="cold")
    signatures = router._prompt_signatures(request)
    base_rank = router._hash_rank(router._session_key(signatures))
    other_rank = 1 - base_rank
    counts = [[0, 0], [0, 0]]
    counts[base_rank] = [10, 0]

    decision = router.route(request, counts, _least_loaded)

    assert decision.rank == other_rank
    assert decision.miss_rerouted


def test_router_updates_sticky_rank_after_miss_reroute():
    router = _make_router(block_size=2)
    first = _make_engine_request([10, 11, 12, 13], request_id="first")
    signatures = router._prompt_signatures(first)
    session_key = router._session_key(signatures)
    base_rank = router._hash_rank(session_key)
    other_rank = 1 - base_rank
    counts = [[0, 0], [0, 0]]
    counts[base_rank] = [10, 0]

    first_decision = router.route(first, counts, _least_loaded)
    second_decision = router.route(
        _make_engine_request([10, 11, 12, 13], request_id="second"),
        [[0, 0], [0, 0]],
        _least_loaded,
    )

    assert first_decision.rank == other_rank
    assert second_decision.rank == other_rank


def test_parallel_config_validates_dp_prefix_cache_lb():
    with pytest.raises(ValueError, match="data_parallel_size > 1"):
        ParallelConfig(data_parallel_prefix_cache_lb=True)

    with pytest.raises(ValueError, match="External DP"):
        ParallelConfig(
            data_parallel_size=2,
            data_parallel_external_lb=True,
            data_parallel_prefix_cache_lb=True,
        )

    with pytest.raises(ValueError, match="enable_elastic_ep"):
        ParallelConfig(
            data_parallel_size=2,
            data_parallel_prefix_cache_lb=True,
            enable_elastic_ep=True,
        )

    with pytest.raises(ValueError, match="deep_depth"):
        ParallelConfig(
            data_parallel_size=2,
            data_parallel_prefix_cache_lb=True,
            data_parallel_prefix_cache_lb_shallow_depth=2,
            data_parallel_prefix_cache_lb_deep_depth=2,
        )


def test_vllm_config_requires_prefix_caching_for_dp_prefix_cache_lb():
    with pytest.raises(ValueError, match="requires prefix caching"):
        VllmConfig(
            cache_config=CacheConfig(enable_prefix_caching=False),
            device_config=DeviceConfig("cpu"),
            parallel_config=ParallelConfig(
                data_parallel_size=2,
                data_parallel_prefix_cache_lb=True,
            ),
        )


def test_vllm_config_requires_pythonhashseed_for_dp_prefix_cache_lb(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        VllmConfig(
            cache_config=CacheConfig(enable_prefix_caching=True),
            device_config=DeviceConfig("cpu"),
            parallel_config=ParallelConfig(
                data_parallel_size=2,
                data_parallel_prefix_cache_lb=True,
            ),
        )


def test_ready_response_sets_resolved_hash_block_size():
    vllm_config = SimpleNamespace(
        cache_config=CacheConfig(hash_block_size=None),
        model_config=SimpleNamespace(max_model_len=2048),
    )
    client = SimpleNamespace(vllm_config=vllm_config, stats_update_address=None)
    payload = msgspec.msgpack.encode(
        EngineCoreReadyResponse(
            max_model_len=1024,
            num_gpu_blocks=1,
            block_size=16,
            dp_stats_address=None,
            dtype="float16",
            vllm_version="test",
            world_size=1,
            data_parallel_size=1,
            hash_block_size=8,
        )
    )

    AsyncMPClient._apply_ready_response(client, payload)  # type: ignore[arg-type]

    assert vllm_config.cache_config.hash_block_size == 8
    assert vllm_config.cache_config.block_size == 16


def test_ready_response_rejects_hash_block_size_mismatch():
    vllm_config = SimpleNamespace(
        cache_config=CacheConfig(hash_block_size=4),
        model_config=SimpleNamespace(max_model_len=2048),
    )
    client = SimpleNamespace(vllm_config=vllm_config, stats_update_address=None)
    payload = msgspec.msgpack.encode(
        EngineCoreReadyResponse(
            max_model_len=1024,
            num_gpu_blocks=1,
            block_size=16,
            dp_stats_address=None,
            dtype="float16",
            vllm_version="test",
            world_size=1,
            data_parallel_size=1,
            hash_block_size=8,
        )
    )

    with pytest.raises(ValueError, match="hash_block_size"):
        AsyncMPClient._apply_ready_response(client, payload)  # type: ignore[arg-type]


def test_mamba_aligned_split_stops_at_dp_prefix_branch_point():
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4),
        use_eagle=False,
    )
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=16,
        num_tokens=16,
        dp_prefix_cache_prefix_len=8,
    )

    num_tokens = Scheduler._mamba_block_aligned_split(
        scheduler,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        num_new_tokens=12,
    )

    assert num_tokens == 8


def test_mamba_aligned_split_counts_external_tokens_as_computed():
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4),
        use_eagle=False,
    )
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=16,
        num_tokens=16,
        dp_prefix_cache_prefix_len=8,
    )

    num_tokens = Scheduler._mamba_block_aligned_split(
        scheduler,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        num_new_tokens=8,
        num_external_computed_tokens=4,
    )

    assert num_tokens == 4
