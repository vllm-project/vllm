# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FlexKVConnectorV1.

These tests mock the ``flexkv`` package so they can run without a real FlexKV
installation.  They verify:

1. That ``FlexKVConnectorV1`` raises a helpful ``ImportError`` when FlexKV is
   not installed.
2. That all public methods are correctly delegated to the underlying
   ``FlexKVConnectorV1Impl``.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig

from .utils import create_vllm_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vllm_config(
    kv_connector: str = "FlexKVConnectorV1",
    kv_role: str = "kv_both",
) -> VllmConfig:
    """Return a minimal VllmConfig with a KVTransferConfig attached."""
    vllm_config = create_vllm_config(block_size=16, max_num_batched_tokens=512)
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector=kv_connector,
        kv_role=kv_role,
    )
    return vllm_config


def _make_kv_cache_config() -> KVCacheConfig:
    return MagicMock(spec=KVCacheConfig)


def _make_flexkv_module(
    impl_mock: MagicMock,
) -> tuple[types.ModuleType, types.ModuleType]:
    """Build a fake ``flexkv`` package hierarchy that returns *impl_mock*
    when ``FlexKVConnectorV1Impl`` is instantiated."""
    flexkv_mod = types.ModuleType("flexkv")
    integration_mod = types.ModuleType("flexkv.integration")
    vllm_mod = types.ModuleType("flexkv.integration.vllm")
    adapter_mod = types.ModuleType("flexkv.integration.vllm.vllm_v1_adapter")

    # Make FlexKVConnectorV1Impl() return our mock instance.
    # The "# type: ignore" markers below are needed because ModuleType does
    # not declare these attributes statically; they are set dynamically.
    FlexKVConnectorV1ImplCls = MagicMock(return_value=impl_mock)
    adapter_mod.FlexKVConnectorV1Impl = FlexKVConnectorV1ImplCls  # type: ignore

    flexkv_mod.integration = integration_mod  # type: ignore
    integration_mod.vllm = vllm_mod  # type: ignore
    vllm_mod.vllm_v1_adapter = adapter_mod  # type: ignore

    return flexkv_mod, adapter_mod


def _install_flexkv_mock(impl_mock: MagicMock):
    """Insert fake flexkv modules into sys.modules and return a context that
    cleans them up afterwards."""
    flexkv_mod, adapter_mod = _make_flexkv_module(impl_mock)
    mods = {
        "flexkv": flexkv_mod,
        "flexkv.integration": flexkv_mod.integration,
        "flexkv.integration.vllm": flexkv_mod.integration.vllm,
        "flexkv.integration.vllm.vllm_v1_adapter": adapter_mod,
    }
    return patch.dict(sys.modules, mods)


def _build_connector(vllm_config: VllmConfig, impl_mock: MagicMock):
    """Instantiate FlexKVConnectorV1 with faked flexkv modules."""
    from vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector import (
        FlexKVConnectorV1,
    )

    with _install_flexkv_mock(impl_mock):
        connector = FlexKVConnectorV1(
            vllm_config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=_make_kv_cache_config(),
        )
    return connector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlexKVConnectorImportError:
    """FlexKVConnectorV1 should fail with a helpful message when flexkv is
    absent."""

    def test_import_error_message(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.flexkv_connector import (
            FlexKVConnectorV1,
        )

        # Ensure flexkv is NOT in sys.modules
        for key in list(sys.modules):
            if key.startswith("flexkv"):
                del sys.modules[key]

        with pytest.raises(ImportError, match="(?i)flexkv") as exc_info:
            FlexKVConnectorV1(
                vllm_config=_make_vllm_config(),
                role=KVConnectorRole.WORKER,
                kv_cache_config=_make_kv_cache_config(),
            )

        assert "https://github.com/taco-project/FlexKV" in str(exc_info.value)


class TestFlexKVConnectorDelegation:
    """All public API methods should be forwarded to the impl."""

    @pytest.fixture()
    def connector_and_impl(self):
        impl = MagicMock()
        cfg = _make_vllm_config()
        connector = _build_connector(cfg, impl)
        return connector, impl

    def test_shutdown(self, connector_and_impl):
        connector, impl = connector_and_impl
        connector.shutdown()
        impl.shutdown.assert_called_once()

    def test_start_load_kv(self, connector_and_impl):
        connector, impl = connector_and_impl
        ctx = MagicMock()
        connector.start_load_kv(ctx, extra_arg="x")
        impl.start_load_kv.assert_called_once_with(ctx, extra_arg="x")

    def test_save_kv_layer(self, connector_and_impl):
        connector, impl = connector_and_impl
        kv_layer = torch.zeros(4, 4)
        attn_meta = MagicMock()
        connector.save_kv_layer("layer_0", kv_layer, attn_meta)
        impl.save_kv_layer.assert_called_once_with("layer_0", kv_layer, attn_meta)

    def test_wait_for_save(self, connector_and_impl):
        connector, impl = connector_and_impl
        connector.wait_for_save()
        impl.wait_for_save.assert_called_once()

    def test_get_finished(self, connector_and_impl):
        connector, impl = connector_and_impl
        impl.get_finished.return_value = ({"req1"}, None)
        result = connector.get_finished({"req1"})
        impl.get_finished.assert_called_once_with({"req1"})
        assert result == ({"req1"}, None)

    def test_register_kv_caches(self, connector_and_impl):
        connector, impl = connector_and_impl
        kv_caches = {"layer_0": torch.zeros(1)}
        connector.register_kv_caches(kv_caches)
        impl.register_kv_caches.assert_called_once_with(kv_caches)

    def test_get_num_new_matched_tokens(self, connector_and_impl):
        connector, impl = connector_and_impl
        req = MagicMock()
        impl.get_num_new_matched_tokens.return_value = (10, False)
        result = connector.get_num_new_matched_tokens(req, 5)
        impl.get_num_new_matched_tokens.assert_called_once_with(req, 5)
        assert result == (10, False)

    def test_update_state_after_alloc(self, connector_and_impl):
        connector, impl = connector_and_impl
        req = MagicMock()
        blocks = MagicMock()
        connector.update_state_after_alloc(req, blocks, 4)
        impl.update_state_after_alloc.assert_called_once_with(req, blocks, 4)

    def test_build_connector_meta(self, connector_and_impl):
        connector, impl = connector_and_impl
        sched_out = MagicMock()
        connector.build_connector_meta(sched_out)
        impl.build_connector_meta.assert_called_once_with(sched_out)

    def test_update_connector_output(self, connector_and_impl):
        connector, impl = connector_and_impl
        out = MagicMock()
        connector.update_connector_output(out)
        impl.update_connector_output.assert_called_once_with(out)

    def test_request_finished(self, connector_and_impl):
        connector, impl = connector_and_impl
        req = MagicMock()
        impl.request_finished.return_value = (True, {"key": "val"})
        result = connector.request_finished(req, [1, 2, 3])
        impl.request_finished.assert_called_once_with(req, [1, 2, 3])
        assert result == (True, {"key": "val"})

    def test_take_events(self, connector_and_impl):
        connector, impl = connector_and_impl
        impl.take_events.return_value = iter([])
        list(connector.take_events())
        impl.take_events.assert_called_once()

    def test_get_kv_connector_stats(self, connector_and_impl):
        connector, impl = connector_and_impl
        impl.get_kv_connector_stats.return_value = None
        result = connector.get_kv_connector_stats()
        impl.get_kv_connector_stats.assert_called_once()
        assert result is None

    def test_get_block_ids_with_load_errors(self, connector_and_impl):
        connector, impl = connector_and_impl
        impl.get_block_ids_with_load_errors.return_value = {7, 8}
        result = connector.get_block_ids_with_load_errors()
        assert result == {7, 8}

    def test_wait_for_layer_load(self, connector_and_impl):
        connector, impl = connector_and_impl
        connector.wait_for_layer_load("layer_0")
        impl.wait_for_layer_load.assert_called_once_with("layer_0")
