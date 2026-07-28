# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu import pcp_manager
from vllm.v1.worker.gpu.pcp_manager import (
    DEFAULT_PCP_MANAGER_NAME,
    PCPManager,
    PCPManagerRegistry,
    maybe_build_pcp_manager,
)


class DummyPCPManager(PCPManager):
    validated_config = None
    init_kwargs = None

    @staticmethod
    def validate_config(vllm_config, supports_mm_inputs):
        DummyPCPManager.validated_config = (vllm_config, supports_mm_inputs)

    def __init__(self, **kwargs):
        DummyPCPManager.init_kwargs = kwargs


class NotAPCPManager:
    pass


@pytest.fixture
def test_manager_name():
    name = "_pytest_pcp_manager"
    try:
        yield name
    finally:
        PCPManagerRegistry._registry.pop(name, None)


def test_default_manager_is_registered():
    assert PCPManagerRegistry.get_manager_class(DEFAULT_PCP_MANAGER_NAME) is PCPManager


def test_register_and_resolve_manager(test_manager_name):
    PCPManagerRegistry.register_manager(
        test_manager_name,
        __name__,
        "DummyPCPManager",
    )

    assert PCPManagerRegistry.get_manager_class(test_manager_name) is DummyPCPManager


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        PCPManagerRegistry.register_manager(
            DEFAULT_PCP_MANAGER_NAME,
            __name__,
            "DummyPCPManager",
        )


def test_unknown_manager_raises():
    with pytest.raises(ValueError, match="Available PCP managers: default"):
        PCPManagerRegistry.get_manager_class("_missing_pcp_manager")


def test_registered_class_must_extend_pcp_manager(test_manager_name):
    PCPManagerRegistry.register_manager(
        test_manager_name,
        __name__,
        "NotAPCPManager",
    )

    with pytest.raises(TypeError, match="must be a subclass"):
        PCPManagerRegistry.get_manager_class(test_manager_name)


def test_maybe_build_uses_registered_manager(monkeypatch, test_manager_name):
    PCPManagerRegistry.register_manager(
        test_manager_name,
        __name__,
        "DummyPCPManager",
    )
    monkeypatch.setattr(
        pcp_manager,
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=2),
    )
    monkeypatch.setattr(
        pcp_manager,
        "get_dcp_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )

    parallel_config = SimpleNamespace(
        prefill_context_parallel_size=4,
        decode_context_parallel_size=2,
        cp_kv_cache_interleave_size=8,
    )
    scheduler_config = SimpleNamespace(
        max_num_seqs=16,
        max_num_batched_tokens=1024,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
    )
    device = object()
    req_states = object()
    block_tables = object()

    manager = maybe_build_pcp_manager(
        vllm_config,
        device,
        supports_mm_inputs=True,
        req_states=req_states,
        block_tables=block_tables,
        manager_name=test_manager_name,
    )

    assert isinstance(manager, DummyPCPManager)
    assert DummyPCPManager.validated_config == (vllm_config, True)
    assert DummyPCPManager.init_kwargs == {
        "pcp_world_size": 4,
        "pcp_rank": 2,
        "device": device,
        "req_states": req_states,
        "max_num_reqs": 16,
        "max_num_tokens": 1024,
        "block_tables": block_tables,
        "dcp_world_size": 2,
        "dcp_rank": 1,
        "cp_interleave": 8,
    }


def test_disabled_pcp_does_not_resolve_manager():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1)
    )

    assert (
        maybe_build_pcp_manager(
            vllm_config,
            device=object(),
            supports_mm_inputs=False,
            req_states=object(),
            block_tables=object(),
            manager_name="_missing_pcp_manager",
        )
        is None
    )
