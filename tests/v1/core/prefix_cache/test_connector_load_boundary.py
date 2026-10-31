# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The end of a KV connector load as a retained Mamba "align" checkpoint.

Sparse prefix-cache retention keeps a Mamba state only at proven reuse points
(replay boundary, shared-prefix junction). vLLM tells a connector, through
``get_reusable_load_rule``, where a load must end to stay hittable under the
EAGLE block drop; for a connector that declares it ends its loads there, the
scheduler also marks the end of each completed load, so the next request of
that prefix can hit the loaded state. ``VLLM_MAMBA_CACHE_CONNECTOR_LOAD_STATE=0``
turns the marking off.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position

BLOCK = 16
FULL, MAMBA = 0, 1
PROMPT = list(range(1, 10 * BLOCK + 4))
LOAD_END = 6 * BLOCK


def _manager(spec_types=("full", "mamba"), use_eagle=True):
    init_none_hash(sha256)
    config = _make_hybrid_kv_cache_config(BLOCK, 1024, list(spec_types))
    return make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1 << 16,
        enable_caching=True,
        hash_block_size=BLOCK,
        use_eagle=use_eagle,
        # CacheConfig's default: keep only proven reuse points.
        retention_interval=0,
    )


def _load_then_prefill(manager, request, load_end, boundary):
    """A connector load of ``load_end`` tokens, then the local rest of the prompt."""
    blocks, local, junction = manager.get_computed_blocks(request)
    request.shared_prefix_boundary = junction
    assert (
        manager.allocate_slots(
            request,
            0,
            num_new_computed_tokens=local,
            new_computed_blocks=blocks,
            num_external_computed_tokens=load_end - local,
            delay_cache_blocks=True,
        )
        is not None
    )
    request.num_computed_tokens = load_end
    request.connector_load_boundary = boundary
    manager.cache_blocks(request, load_end)
    state = manager.coordinator.single_type_managers[MAMBA].req_to_blocks[
        request.request_id
    ][load_end // BLOCK - 1]
    loaded_state_cached = state.block_hash is not None

    num_new = request.num_tokens - load_end
    assert manager.allocate_slots(request, num_new) is not None
    request.num_computed_tokens += num_new
    return loaded_state_cached


@pytest.mark.parametrize("retain", [False, True])
def test_loaded_mamba_state_cached_only_with_load_boundary(retain):
    manager = _manager()
    request = make_request("load", PROMPT, BLOCK, sha256)

    cached = _load_then_prefill(manager, request, LOAD_END, LOAD_END if retain else 0)

    # Cold cache: no junction, and the load end is not the replay boundary.
    assert request.shared_prefix_boundary == 0
    assert cached == retain


@pytest.mark.parametrize("retain", [False, True])
def test_sibling_hits_the_retained_load_boundary(retain):
    manager = _manager()
    first = make_request("load", PROMPT, BLOCK, sha256)
    _load_then_prefill(manager, first, LOAD_END, LOAD_END if retain else 0)

    # Shares seven blocks, then diverges: the attention match is seven blocks,
    # the EAGLE drop lands it on the load end.
    sibling_prompt = PROMPT[: 7 * BLOCK] + [10**6 + i for i in range(3 * BLOCK)]
    sibling = make_request("sibling", sibling_prompt, BLOCK, sha256)
    _, num_hit, _ = manager.get_computed_blocks(sibling)

    assert num_hit == (LOAD_END if retain else 0)


@pytest.mark.parametrize(
    "enabled, failed", [(False, False), (True, False), (True, True)]
)
def test_completed_load_marks_its_end_only_when_enabled(enabled, failed):
    sched = MagicMock()
    sched.failed_recving_kv_req_ids = {"r"} if failed else set()
    sched.finished_recving_kv_req_ids = {"r"}
    sched._cache_connector_load_state = enabled
    sched.needs_kv_cache_zeroing = False
    sched._mark_prefix_replay.return_value = 0
    request = SimpleNamespace(
        request_id="r",
        num_computed_tokens=LOAD_END,
        num_tokens=len(PROMPT),
        connector_load_boundary=0,
    )

    Scheduler._update_waiting_for_remote_kv(sched, request)

    assert request.connector_load_boundary == (
        LOAD_END if enabled and not failed else 0
    )


def test_reusable_load_rule_is_the_eagle_mamba_checkpoint():
    granularity, drop = _manager().get_reusable_load_rule()

    assert (granularity, drop) == (BLOCK, BLOCK)
    for n in range(12 * BLOCK):
        assert max(n // granularity * granularity - drop, 0) == (
            get_mamba_prefill_checkpoint_position(n, BLOCK, drop_eagle_block=True)
        )


@pytest.mark.parametrize(
    "spec_types, use_eagle",
    [(("full", "mamba"), False), (("full",), True), (("full", "sliding_window"), True)],
)
def test_no_reusable_load_rule_without_eagle_drop_and_mamba(spec_types, use_eagle):
    assert _manager(spec_types, use_eagle).get_reusable_load_rule() is None


@pytest.mark.parametrize(
    "declares, rule, env, expected",
    [
        (True, True, None, True),
        (True, True, "0", False),
        (False, True, None, False),
        (True, False, None, False),
    ],
)
def test_load_state_retention_follows_the_handshake(
    monkeypatch, declares, rule, env, expected
):
    if env is None:
        monkeypatch.delenv("VLLM_MAMBA_CACHE_CONNECTOR_LOAD_STATE", raising=False)
    else:
        monkeypatch.setenv("VLLM_MAMBA_CACHE_CONNECTOR_LOAD_STATE", env)
    sched = MagicMock()
    sched.connector.ends_loads_on_reusable_boundary = declares
    sched.kv_cache_manager = _manager(use_eagle=rule)

    assert Scheduler._resolve_cache_connector_load_state(sched) is expected


def test_connectors_forward_the_rule_of_the_bound_manager():
    connector = SimpleNamespace(_kv_cache_manager=None)
    assert KVConnectorBase_V1.get_reusable_load_rule(connector) is None
    assert KVConnectorBase_V1.ends_loads_on_reusable_boundary is False

    children = [
        SimpleNamespace(_kv_cache_manager=_manager(), ends_loads_on_reusable_boundary=f)
        for f in (False, True)
    ]
    for child in children:
        child.get_reusable_load_rule = (
            lambda c=child: KVConnectorBase_V1.get_reusable_load_rule(c)
        )
    multi = MultiConnector.__new__(MultiConnector)
    multi._connectors = children
    assert multi.ends_loads_on_reusable_boundary is True
    assert multi.get_reusable_load_rule() == (BLOCK, BLOCK)
    multi._connectors = []
    assert multi.get_reusable_load_rule() is None
