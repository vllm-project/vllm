# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for backwards compatibility with external KV connector implementations.

This test ensures that external connectors (loaded via kv_connector_module_path)
implemented with the old signature continue to work:
- Old signature: __init__(self, vllm_config, role)
- New signature: __init__(self, vllm_config, role, kv_cache_config)
"""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from .utils import create_scheduler, create_vllm_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class OldStyleTestConnector(KVConnectorBase_V1):
    """
    Test connector using the old signature with 2 required arguments.
    This simulates external connectors that haven't been updated yet.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        # Old-style call to super().__init__ with only 2 arguments
        super().__init__(vllm_config=vllm_config, role=role)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        pass

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        return None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        pass


class NewStyleTestConnector(KVConnectorBase_V1):
    """
    Test connector using the new signature with 3 required arguments.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        # New-style call to super().__init__ with all 3 arguments
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        pass

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        return None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        pass


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_external_old_signature_factory_instantiation(role):
    """
    Test that external connectors with old signature (2 required args) loaded
    via kv_connector_module_path are correctly instantiated with backwards
    compatibility support.
    """
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "OldStyleTestConnector"
    vllm_config.kv_transfer_config.kv_connector_module_path = (
        "tests.v1.kv_connector.unit.test_backwards_compatibility"
    )

    scheduler = create_scheduler(vllm_config)
    kv_cache_config = scheduler.kv_cache_config

    connector = KVConnectorFactory.create_connector(vllm_config, role, kv_cache_config)

    assert connector is not None
    assert isinstance(connector, OldStyleTestConnector)
    assert connector.role == role
    assert connector._kv_cache_config is None


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_external_new_signature_factory_instantiation(role):
    """
    Test that external connectors with new signature (3 required args) loaded
    via kv_connector_module_path are correctly instantiated.
    """
    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "NewStyleTestConnector"
    vllm_config.kv_transfer_config.kv_connector_module_path = (
        "tests.v1.kv_connector.unit.test_backwards_compatibility"
    )

    scheduler = create_scheduler(vllm_config)
    kv_cache_config = scheduler.kv_cache_config

    connector = KVConnectorFactory.create_connector(vllm_config, role, kv_cache_config)

    assert connector is not None
    assert isinstance(connector, NewStyleTestConnector)
    assert connector.role == role
    assert connector._kv_cache_config is not None
    assert connector._kv_cache_config == kv_cache_config


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_old_signature_super_init(role):
    """
    Test that old-style connectors can call super().__init__() without
    kv_cache_config parameter.
    """
    vllm_config = create_vllm_config()

    connector = OldStyleTestConnector(vllm_config, role)

    assert connector is not None
    assert connector.role == role
    assert connector._kv_cache_config is None


def test_old_signature_super_init_with_kwargs():
    """
    Test that old-style connectors can call super().__init__() with keyword
    arguments in different orders.
    """
    vllm_config = create_vllm_config()

    # Test with vllm_config= and role= kwargs
    connector1 = OldStyleTestConnector(
        vllm_config=vllm_config, role=KVConnectorRole.SCHEDULER
    )
    assert connector1 is not None
    assert connector1._kv_cache_config is None

    # Test with role= and vllm_config= in reversed order
    connector2 = OldStyleTestConnector(
        role=KVConnectorRole.WORKER, vllm_config=vllm_config
    )
    assert connector2 is not None
    assert connector2._kv_cache_config is None


def test_internal_connector_uses_new_signature():
    """
    Test that internal connectors (registered in factory) always use the new
    signature and get kv_cache_config.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
        ExampleConnector,
    )

    vllm_config = create_vllm_config()
    vllm_config.kv_transfer_config.kv_connector = "ExampleConnector"

    scheduler = create_scheduler(vllm_config)
    kv_cache_config = scheduler.kv_cache_config

    connector = KVConnectorFactory.create_connector(
        vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
    )

    assert connector is not None
    assert isinstance(connector, ExampleConnector)
    assert connector._kv_cache_config is not None
    assert connector._kv_cache_config == kv_cache_config


def test_signature_detection_with_mocking():
    """
    Test that the factory correctly applies compat_sig flag returned from
    _get_connector_class_with_compat.
    """
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    kv_cache_config = scheduler.kv_cache_config

    # Mock _get_connector_class_with_compat to return old-style connector
    with patch.object(
        KVConnectorFactory,
        "_get_connector_class_with_compat",
        return_value=(OldStyleTestConnector, True),
    ):
        old_connector = KVConnectorFactory.create_connector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )
        assert old_connector is not None
        assert isinstance(old_connector, OldStyleTestConnector)
        assert old_connector._kv_cache_config is None

    # Mock _get_connector_class_with_compat to return new-style connector
    with patch.object(
        KVConnectorFactory,
        "_get_connector_class_with_compat",
        return_value=(NewStyleTestConnector, False),
    ):
        new_connector = KVConnectorFactory.create_connector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )
        assert new_connector is not None
        assert isinstance(new_connector, NewStyleTestConnector)
        assert new_connector._kv_cache_config is not None
        assert new_connector._kv_cache_config == kv_cache_config


# ---------------------------------------------------------------------------
# Regression tests for issue #40690:
# MultiConnector must route child construction through the compat-aware factory
# so that legacy 2-arg connectors work as children, not just at the top level.
# ---------------------------------------------------------------------------

_THIS_MODULE = "tests.v1.kv_connector.unit.test_backwards_compatibility"


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_multiconnector_old_style_child_instantiation(role):
    """
    Regression test for #40690.

    A legacy connector with the old 2-arg __init__(self, vllm_config, role)
    must work as a child of MultiConnector.  Before the fix, MultiConnector
    directly called connector_cls(config, role, kv_cache_config), which passed
    a third positional argument that the old-style connector doesn't accept,
    raising TypeError.  After the fix, child construction goes through
    KVConnectorFactory.create_connector, which applies the same compat shim
    that the top-level path uses.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "OldStyleTestConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": _THIS_MODULE,
                }
            ]
        },
    )

    scheduler = create_scheduler(vllm_config)
    kv_cache_config = scheduler.kv_cache_config

    # This must not raise TypeError before or after the fix.
    connector = KVConnectorFactory.create_connector(vllm_config, role, kv_cache_config)

    assert isinstance(connector, MultiConnector)
    assert len(connector._connectors) == 1
    child = connector._connectors[0]
    assert isinstance(child, OldStyleTestConnector)
    # Old-style connector receives no kv_cache_config
    assert child._kv_cache_config is None


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_multiconnector_new_style_child_unaffected(role):
    """
    Ensure that the fix does not break new-style 3-arg children inside
    MultiConnector.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    vllm_config = create_vllm_config(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "NewStyleTestConnector",
                    "kv_role": "kv_both",
                    "kv_connector_module_path": _THIS_MODULE,
                }
            ]
        },
    )

    scheduler = create_scheduler(vllm_config)
    kv_cache_config = scheduler.kv_cache_config

    connector = KVConnectorFactory.create_connector(vllm_config, role, kv_cache_config)

    assert isinstance(connector, MultiConnector)
    assert len(connector._connectors) == 1
    child = connector._connectors[0]
    assert isinstance(child, NewStyleTestConnector)
    # New-style connector receives kv_cache_config
    assert child._kv_cache_config is not None
    assert child._kv_cache_config == kv_cache_config
