# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive UMBP lookup hints need neither GPUs nor a MORI service."""

import threading

import pytest

from tests.v1.kv_connector.umbp_test_utils import (
    _hybrid_kv_cache_config,
    _kv_cache_config,
    _SchedulerHandle,
)
from tests.v1.simple_kv_offload.test_scheduler import (
    _make_vllm_config,
    make_request,
    make_scheduler,
)
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.data import (
    BlockIdentityCodec,
    UMBPNamespace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPStoreConnectorScheduler,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("hit", [False, True])
def test_known_prefix_waits_for_normal_reduction_without_querying(hybrid, hit):
    gate = threading.Event()

    class Handle(_SchedulerHandle):
        def lookup(self, keys):
            self.queries.append(list(keys))
            assert gate.wait(timeout=10)
            return [hit] * len(keys)

    handle = Handle([])
    config = _make_vllm_config()
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="UMBPStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"lookup_async": True},
    )
    scheduler = UMBPStoreConnectorScheduler(
        config,
        _hybrid_kv_cache_config() if hybrid else _kv_cache_config(),
        handle,
        BlockIdentityCodec(UMBPNamespace("known-prefix")),
    )
    request = make_request(num_blocks=2)
    try:
        assert scheduler.get_known_num_matched_tokens(request) is None
        assert handle.queries == []
        assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
        assert scheduler.get_known_num_matched_tokens(request) is None
        gate.set()
        pending = scheduler._pending_lookups[request.request_id]
        pending.future.result(timeout=10)
        # The ordering hook must not run the stateful hybrid coordinator.
        assert scheduler.get_known_num_matched_tokens(request) is None
        assert scheduler._pending_lookups[request.request_id] is pending
        assert scheduler._load_specs == {}

        expected = 32 if hit else 0
        assert scheduler.get_num_new_matched_tokens(request, 0)[0] == expected
        load_specs = dict(scheduler._load_specs)
        assert scheduler.get_known_num_matched_tokens(request) == expected
        assert scheduler.get_known_num_matched_tokens(request) == expected
        assert len(handle.queries) == 1
        assert scheduler._load_specs == load_specs
        assert scheduler.reset_store()
        assert scheduler.get_known_num_matched_tokens(request) is None
    finally:
        gate.set()
        scheduler.close()


@pytest.mark.parametrize("local_tokens", [0, 16])
def test_simplecpu_known_miss_keeps_local_boundary_and_rechecks(local_tokens):
    scheduler = make_scheduler().scheduler
    request = make_request(num_blocks=2)
    assert scheduler.get_known_num_matched_tokens(request) is None
    assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (0, False)
    assert scheduler.get_known_num_matched_tokens(request) == local_tokens
    assert scheduler.get_known_num_matched_tokens(request) == local_tokens
    # Reusing a string ID cannot transfer another request's knowledge.
    replacement = make_request(request_id=request.request_id)
    assert scheduler.get_known_num_matched_tokens(replacement) is None
    scheduler.request_finished(request, [])
    assert scheduler.get_known_num_matched_tokens(request) is None
    assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (0, False)
    assert scheduler.reset()
    assert scheduler.get_known_num_matched_tokens(request) is None


@pytest.mark.parametrize("malformed", [False, True])
def test_failed_umbp_lookup_is_unknown_not_a_known_miss(malformed):
    class Handle(_SchedulerHandle):
        def lookup(self, keys):
            if malformed:
                return []
            raise TimeoutError("unavailable store")

    config = _make_vllm_config()
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="UMBPStoreConnector", kv_role="kv_both"
    )
    scheduler = UMBPStoreConnectorScheduler(
        config,
        _kv_cache_config(),
        Handle([]),
        BlockIdentityCodec(UMBPNamespace("failed-prefix")),
    )
    request = make_request(num_blocks=2)
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        assert scheduler.get_known_num_matched_tokens(request) is None
    finally:
        scheduler.close()


@pytest.mark.parametrize("change", ["tokens", "hashes", "skip_reads"])
def test_known_prefix_rejects_changed_request_context(change):
    scheduler = make_scheduler().scheduler
    request = make_request(num_blocks=2)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert scheduler.get_known_num_matched_tokens(request) == 0
    if change == "tokens":
        request.append_output_token_ids(100)
    elif change == "hashes":
        request.block_hashes[-1] = b"changed"
    else:
        request.prefix_cache_hit_limit = 0
    assert scheduler.get_known_num_matched_tokens(request) is None
