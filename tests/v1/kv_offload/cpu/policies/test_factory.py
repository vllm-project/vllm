# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import pytest

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class _DummyCachePolicy(CachePolicy):
    """Minimal CachePolicy for CachePolicyFactory registration tests. Loaded
    by module path, so it must be importable at module scope (mirrors
    tests/v1/kv_offload/test_factory.py's SingleArgExternalOffloadingSpec)."""

    def __init__(self, cache_capacity: int) -> None:
        self.cache_capacity = cache_capacity

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return None

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        pass

    def remove(self, key: OffloadKey) -> None:
        pass

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        pass

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        return None

    def clear(self) -> None:
        pass


@pytest.fixture(autouse=True)
def restore_cache_policy_registry():
    """Save and restore CachePolicyFactory._registry between tests."""
    original = dict(CachePolicyFactory._registry)
    yield
    CachePolicyFactory._registry = original


class TestCachePolicyFactory:
    """Unit tests for CachePolicyFactory (registration/resolution by name)."""

    def test_pre_registered_policies_can_be_imported(self):
        """If someone moves a policy module but forgets to update
        factory.py, CI fails."""
        for name in CachePolicyFactory._registry:
            cls = CachePolicyFactory._registry[name]()
            assert issubclass(cls, CachePolicy)

    def test_lru_and_arc_registered(self):
        assert CachePolicyFactory.get_cache_policy_cls("lru") is LRUCachePolicy
        assert CachePolicyFactory.get_cache_policy_cls("arc") is ARCCachePolicy

    def test_register_and_resolve_custom_policy(self):
        CachePolicyFactory.register_cache_policy(
            "dummy",
            "tests.v1.kv_offload.cpu.policies.test_factory",
            "_DummyCachePolicy",
        )
        policy_cls = CachePolicyFactory.get_cache_policy_cls("dummy")
        assert policy_cls is _DummyCachePolicy

        manager = CPUOffloadingManager(num_blocks=4, cache_policy="dummy")
        assert isinstance(manager._policy, _DummyCachePolicy)

    def test_unregistered_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown cache policy"):
            CachePolicyFactory.get_cache_policy_cls("nonexistent")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="is already registered"):
            CachePolicyFactory.register_cache_policy("lru", "some.module", "SomeClass")

    def test_dynamic_load_via_cache_policy_module_path(self):
        """Out-of-tree policy loaded via cache_policy_module_path, no
        register_cache_policy() call -- this is how external projects
        integrate a custom CachePolicy without forking/patching vLLM.
        Mirrors tests/v1/kv_offload/test_factory.py's
        test_dynamic_load_via_spec_module_path."""
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            "_DummyCachePolicy", "tests.v1.kv_offload.cpu.policies.test_factory"
        )
        assert policy_cls is _DummyCachePolicy

    def test_manager_resolves_policy_via_module_path(self):
        """End-to-end: CPUOffloadingManager resolves an unregistered policy
        purely from cache_policy_module_path."""
        manager = CPUOffloadingManager(
            num_blocks=4,
            cache_policy="_DummyCachePolicy",
            cache_policy_module_path="tests.v1.kv_offload.cpu.policies.test_factory",
        )
        assert isinstance(manager._policy, _DummyCachePolicy)

    def test_unregistered_policy_without_module_path_raises(self):
        """eviction_policy not in registry + no cache_policy_module_path ->
        ValueError, same shape as the OffloadingSpecFactory error path."""
        with pytest.raises(ValueError, match="Unknown cache policy"):
            CachePolicyFactory.get_cache_policy_cls("nonexistent", None)
