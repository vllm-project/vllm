# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import CacheConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import kv_connector_manager
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole)
from vllm.plugins.extension_manager import ExtensionManager


class BaseA:

    def __init__(self) -> None:
        pass


class BaseB:

    def __init__(self) -> None:
        pass


extension_manager_a = ExtensionManager(base_cls=BaseA)
extension_manager_b = ExtensionManager(base_cls=BaseB)


@extension_manager_a.register(names=["a1"])
class ChildA1(BaseA):

    def __init__(self) -> None:
        super().__init__()


@extension_manager_a.register(names=["a2", "a2_alias"])
class ChildA2(BaseA):

    def __init__(self) -> None:
        super().__init__()


@extension_manager_b.register(names=["b1"])
class ChildB1(BaseB):

    def __init__(self) -> None:
        super().__init__()


@extension_manager_b.register(names=["b2"])
class ChildB2(BaseB):

    def __init__(self) -> None:
        super().__init__()


def test_extension_manager_can_register_and_create():
    a1_obj = extension_manager_a.create("a1")
    a2_obj = extension_manager_a.create("a2")

    assert isinstance(a1_obj, ChildA1)
    assert isinstance(a2_obj, ChildA2)

    b1_obj = extension_manager_b.create("b1")
    b2_obj = extension_manager_b.create("b2")

    assert isinstance(b1_obj, ChildB1)
    assert isinstance(b2_obj, ChildB2)


def test_extension_manager_can_register_and_get_type():
    a1_cls = extension_manager_a.get_extension_class("a1")
    a2_cls = extension_manager_a.get_extension_class("a2")

    assert a1_cls is ChildA1
    assert a2_cls is ChildA2

    b1_cls = extension_manager_b.get_extension_class("b1")
    b2_cls = extension_manager_b.get_extension_class("b2")

    assert b1_cls is ChildB1
    assert b2_cls is ChildB2


def test_extension_manager_can_register_and_create_with_alias():
    a2_alias_obj = extension_manager_a.create("a2_alias")

    assert isinstance(a2_alias_obj, ChildA2)


def test_extension_manager_throws_error_on_unknown_names():
    with pytest.raises(ValueError):
        extension_manager_a.create("c1")

    with pytest.raises(ValueError):
        extension_manager_b.create("c1")


def test_extension_manager_valid_names():
    assert extension_manager_a.get_valid_extension_names() == [
        "a1", "a2", "a2_alias"
    ]
    assert extension_manager_b.get_valid_extension_names() == ["b1", "b2"]


def test_extension_manager_must_be_unique_per_base_class():
    with pytest.raises(ValueError):
        _ = ExtensionManager(base_cls=BaseA)


def test_extension_manager_lazy_import_and_register():
    # Before any import, this should fail.
    with pytest.raises(ValueError):
        _ = kv_connector_manager.create("TestSharedStorageConnector",
                                        KVConnectorRole.WORKER)

    # Verify lazy import works.
    cache_config = CacheConfig(
        block_size=128,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="TestSharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "name": "test",
            "shared_storage_path": "local_storage"
        },
    )
    vllm_config = VllmConfig(
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
    )
    test_connector = kv_connector_manager.create_or_import(
        name="TestSharedStorageConnector",
        extension_path="tests.v1.kv_connector.unit.utils",
        config=vllm_config,
        role=KVConnectorRole.WORKER)
    assert isinstance(test_connector, KVConnectorBase_V1)
