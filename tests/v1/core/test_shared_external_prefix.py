# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.shared_external_prefix import (
    SharedExternalPrefixKey,
    SharedExternalPrefixLoadManager,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _key() -> SharedExternalPrefixKey:
    return SharedExternalPrefixKey(
        connector_namespace=("test", "source"),
        hash_block_size=16,
        num_local_tokens=0,
        num_external_tokens=32,
        local_end_hash=None,
        external_end_hash=BlockHash(b"external-prefix"),
    )


def test_failed_load_stops_accepting_followers_without_blocking_retry():
    manager = SharedExternalPrefixLoadManager()
    key = _key()
    failed_load = manager.register_owner(key, "owner-1")
    manager.add_follower(failed_load, "follower-1")
    manager.add_follower(failed_load, "follower-2")
    assert manager.detach_follower("follower-1")
    assert not manager.is_follower("follower-1")

    manager.stop_accepting_followers("owner-1")

    assert manager.find(key) is None
    assert manager.get_by_owner("owner-1") is failed_load

    retry_load = manager.register_owner(key, "owner-2")
    assert manager.find(key) is retry_load

    # Draining the failed owner must not remove the replacement key entry.
    assert manager.pop_owner("owner-1") is failed_load
    assert manager.find(key) is retry_load
    assert not manager.is_follower("follower-2")

    assert manager.pop_owner("owner-2") is retry_load
    assert not manager.has_unresolved_loads()


def test_only_global_cpu_offloading_spec_opts_in():
    class CustomCPUOffloadingSpec(CPUOffloadingSpec):
        pass

    cpu_spec = object.__new__(CPUOffloadingSpec)
    tiering_spec = object.__new__(TieringOffloadingSpec)
    custom_spec = object.__new__(CustomCPUOffloadingSpec)

    assert cpu_spec.shared_kv_load_namespace == ("cpu-offloading",)
    assert tiering_spec.shared_kv_load_namespace is None
    assert custom_spec.shared_kv_load_namespace is None
