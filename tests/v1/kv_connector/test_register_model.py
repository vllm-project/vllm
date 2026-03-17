# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for register_model() on KVConnectorBase_V1.

Validates PR: add register_model() to KVConnectorBase_V1 for CacheBlend.
"""

from unittest.mock import MagicMock, patch

import torch


class TestKVConnectorBaseRegisterModel:
    """Test the base class register_model default behavior."""

    def test_base_class_has_register_model(self):
        """register_model exists on KVConnectorBase_V1."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )
        assert hasattr(KVConnectorBase_V1, "register_model")

    def test_base_class_register_model_is_noop(self):
        """Base class register_model does nothing (returns None)."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )
        # Create a minimal instance
        connector = KVConnectorBase_V1.__new__(KVConnectorBase_V1)
        model = MagicMock(spec=torch.nn.Module)
        result = connector.register_model(model)
        assert result is None


class TestActiveKVConnectorRegisterModel:
    """Test that ActiveKVConnector passes model to the connector."""

    def test_register_model_called_when_model_provided(self):
        """ActiveKVConnector calls connector.register_model(model)."""
        mock_connector = MagicMock()
        mock_model = MagicMock(spec=torch.nn.Module)

        with patch(
            "vllm.v1.worker.gpu.kv_connector.get_kv_transfer_group",
            return_value=mock_connector,
        ), patch(
            "vllm.v1.worker.gpu.kv_connector.has_kv_transfer_group",
            return_value=True,
        ):
            from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

            _connector = ActiveKVConnector(
                vllm_config=MagicMock(),
                kv_caches_dict={"layer.0": torch.zeros(1)},
                model=mock_model,
            )

        mock_connector.register_model.assert_called_once_with(mock_model)

    def test_register_model_not_called_when_no_model(self):
        """ActiveKVConnector skips register_model when model=None."""
        mock_connector = MagicMock()

        with patch(
            "vllm.v1.worker.gpu.kv_connector.get_kv_transfer_group",
            return_value=mock_connector,
        ), patch(
            "vllm.v1.worker.gpu.kv_connector.has_kv_transfer_group",
            return_value=True,
        ):
            from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

            _connector = ActiveKVConnector(
                vllm_config=MagicMock(),
                kv_caches_dict={"layer.0": torch.zeros(1)},
                model=None,
            )

        mock_connector.register_model.assert_not_called()


class TestGetKVConnectorModel:
    """Test that get_kv_connector passes model through."""

    def test_passes_model_to_active_connector(self):
        """get_kv_connector forwards model to ActiveKVConnector."""
        mock_connector = MagicMock()
        mock_model = MagicMock(spec=torch.nn.Module)

        with patch(
            "vllm.v1.worker.gpu.kv_connector.get_kv_transfer_group",
            return_value=mock_connector,
        ), patch(
            "vllm.v1.worker.gpu.kv_connector.has_kv_transfer_group",
            return_value=True,
        ):
            from vllm.v1.worker.gpu.kv_connector import get_kv_connector

            _kv_connector = get_kv_connector(
                vllm_config=MagicMock(),
                kv_caches_dict={"layer.0": torch.zeros(1)},
                model=mock_model,
            )

        mock_connector.register_model.assert_called_once_with(mock_model)

    def test_noop_when_no_transfer_group(self):
        """get_kv_connector returns NoOp when no transfer group."""
        with patch(
            "vllm.v1.worker.gpu.kv_connector.has_kv_transfer_group",
            return_value=False,
        ):
            from vllm.v1.worker.gpu.kv_connector import (
                NO_OP_KV_CONNECTOR,
                get_kv_connector,
            )

            result = get_kv_connector(
                vllm_config=MagicMock(),
                kv_caches_dict={},
                model=MagicMock(),
            )
            assert result is NO_OP_KV_CONNECTOR


class TestLMCacheConnectorRegisterModel:
    """Test LMCache connector register_model delegation."""

    def test_register_model_delegates_to_engine(self):
        """LMCacheConnectorV1.register_model delegates to _lmcache_engine."""
        from vllm.distributed.kv_transfer.kv_connector.v1.\
            lmcache_connector import LMCacheConnectorV1

        mock_engine = MagicMock()
        mock_model = MagicMock(spec=torch.nn.Module)

        connector = LMCacheConnectorV1.__new__(LMCacheConnectorV1)
        connector._lmcache_engine = mock_engine
        connector.register_model(mock_model)

        mock_engine.register_model.assert_called_once_with(mock_model)

    def test_register_model_skips_engine_without_method(self):
        """register_model is a no-op when engine lacks register_model."""
        from vllm.distributed.kv_transfer.kv_connector.v1.\
            lmcache_connector import LMCacheConnectorV1

        # Engine without register_model attribute
        mock_engine = MagicMock(spec=[])  # empty spec — no attributes
        mock_model = MagicMock(spec=torch.nn.Module)

        connector = LMCacheConnectorV1.__new__(LMCacheConnectorV1)
        connector._lmcache_engine = mock_engine
        # Should not raise
        connector.register_model(mock_model)


class TestLMCacheConnectorV1ImplRegisterModel:
    """Test LMCacheConnectorV1Impl (native) register_model logic."""

    def test_register_model_calls_vllm_model_tracker(self):
        """LMCacheConnectorV1Impl.register_model registers with VLLMModelTracker."""
        mock_tracker = MagicMock()
        mock_model = MagicMock(spec=torch.nn.Module)

        with patch.dict("sys.modules", {
            "lmcache": MagicMock(),
            "lmcache.v1": MagicMock(),
            "lmcache.v1.compute": MagicMock(),
            "lmcache.v1.compute.models": MagicMock(),
            "lmcache.v1.compute.models.utils": MagicMock(
                VLLMModelTracker=mock_tracker
            ),
        }):
            from vllm.distributed.kv_transfer.kv_connector.v1.\
                lmcache_integration.vllm_v1_adapter import LMCacheConnectorV1Impl

            impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
            impl.register_model(mock_model)

        mock_tracker.register_model.assert_called_once()
        call_args = mock_tracker.register_model.call_args
        assert call_args[0][1] is mock_model

    def test_register_model_graceful_on_import_error(self):
        """register_model doesn't crash if LMCache CacheBlend not installed."""
        from vllm.distributed.kv_transfer.kv_connector.v1.\
            lmcache_integration.vllm_v1_adapter import LMCacheConnectorV1Impl

        impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)

        # Should not raise even if lmcache imports fail
        with patch.dict("sys.modules", {
            "lmcache.v1.compute.models.utils": None,
        }):
            impl.register_model(MagicMock(spec=torch.nn.Module))


class TestMultiConnectorRegisterModel:
    """Test MultiConnector delegates register_model to sub-connectors."""

    def test_delegates_to_all_sub_connectors(self):
        """MultiConnector.register_model calls each sub-connector."""
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )

        mock_model = MagicMock(spec=torch.nn.Module)
        sub1 = MagicMock()
        sub2 = MagicMock()

        connector = MultiConnector.__new__(MultiConnector)
        connector._connectors = [sub1, sub2]
        connector.register_model(mock_model)

        sub1.register_model.assert_called_once_with(mock_model)
        sub2.register_model.assert_called_once_with(mock_model)
