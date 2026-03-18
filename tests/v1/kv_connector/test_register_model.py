# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for initialize_worker_connector() on KVConnectorBase_V1.

Validates PR: replace register_model() with WorkerConnectorInitializationData
pattern for extensible connector initialization.
"""

from unittest.mock import MagicMock, patch

import torch


class TestWorkerConnectorInitializationData:
    """Test the WorkerConnectorInitializationData dataclass."""

    def test_dataclass_default_model_is_none(self):
        """WorkerConnectorInitializationData.model defaults to None."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )

        data = WorkerConnectorInitializationData()
        assert data.model is None

    def test_dataclass_accepts_model(self):
        """WorkerConnectorInitializationData accepts an nn.Module."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )

        model = MagicMock(spec=torch.nn.Module)
        data = WorkerConnectorInitializationData(model=model)
        assert data.model is model

    def test_response_dataclass_exists(self):
        """WorkerConnectorInitializationResponse is importable."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationResponse,
        )

        resp = WorkerConnectorInitializationResponse()
        assert resp is not None


class TestKVConnectorBaseInitializeWorkerConnector:
    """Test the base class initialize_worker_connector default behavior."""

    def test_base_class_has_initialize_worker_connector(self):
        """initialize_worker_connector exists on KVConnectorBase_V1."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        assert hasattr(KVConnectorBase_V1, "initialize_worker_connector")

    def test_base_class_is_noop_returns_response(self):
        """Base class initialize_worker_connector returns a response object."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
            WorkerConnectorInitializationData,
            WorkerConnectorInitializationResponse,
        )

        connector = KVConnectorBase_V1.__new__(KVConnectorBase_V1)
        data = WorkerConnectorInitializationData(model=MagicMock(spec=torch.nn.Module))
        result = connector.initialize_worker_connector(data)
        assert isinstance(result, WorkerConnectorInitializationResponse)


class TestActiveKVConnectorInitializeWorkerConnector:
    """Test that ActiveKVConnector forwards initialization data."""

    def test_initialize_worker_connector_called_with_model(self):
        """ActiveKVConnector calls initialize_worker_connector with model."""
        mock_connector = MagicMock()
        mock_model = MagicMock(spec=torch.nn.Module)

        with (
            patch(
                "vllm.v1.worker.gpu.kv_connector.get_kv_transfer_group",
                return_value=mock_connector,
            ),
            patch(
                "vllm.v1.worker.gpu.kv_connector.has_kv_transfer_group",
                return_value=True,
            ),
        ):
            from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

            _connector = ActiveKVConnector(
                vllm_config=MagicMock(),
                kv_caches_dict={"layer.0": torch.zeros(1)},
                model=mock_model,
            )

        mock_connector.initialize_worker_connector.assert_called_once()
        call_args = mock_connector.initialize_worker_connector.call_args
        init_data = call_args[0][0]
        assert init_data.model is mock_model

    def test_initialize_worker_connector_called_with_none_model(self):
        """ActiveKVConnector still calls initialize_worker_connector when model=None."""
        mock_connector = MagicMock()

        with (
            patch(
                "vllm.v1.worker.gpu.kv_connector.get_kv_transfer_group",
                return_value=mock_connector,
            ),
            patch(
                "vllm.v1.worker.gpu.kv_connector.has_kv_transfer_group",
                return_value=True,
            ),
        ):
            from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

            _connector = ActiveKVConnector(
                vllm_config=MagicMock(),
                kv_caches_dict={"layer.0": torch.zeros(1)},
                model=None,
            )

        mock_connector.initialize_worker_connector.assert_called_once()
        call_args = mock_connector.initialize_worker_connector.call_args
        init_data = call_args[0][0]
        assert init_data.model is None


class TestGetKVConnectorInitializeWorkerConnector:
    """Test that get_kv_connector passes initialization data through."""

    def test_passes_model_via_initialization_data(self):
        """get_kv_connector forwards model in WorkerConnectorInitializationData."""
        mock_connector = MagicMock()
        mock_model = MagicMock(spec=torch.nn.Module)

        with (
            patch(
                "vllm.v1.worker.gpu.kv_connector.get_kv_transfer_group",
                return_value=mock_connector,
            ),
            patch(
                "vllm.v1.worker.gpu.kv_connector.has_kv_transfer_group",
                return_value=True,
            ),
        ):
            from vllm.v1.worker.gpu.kv_connector import get_kv_connector

            _kv_connector = get_kv_connector(
                vllm_config=MagicMock(),
                kv_caches_dict={"layer.0": torch.zeros(1)},
                model=mock_model,
            )

        mock_connector.initialize_worker_connector.assert_called_once()
        call_args = mock_connector.initialize_worker_connector.call_args
        init_data = call_args[0][0]
        assert init_data.model is mock_model

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


class TestLMCacheConnectorInitializeWorkerConnector:
    """Test LMCache connector initialize_worker_connector delegation."""

    def test_initialize_worker_connector_delegates_model_to_engine(self):
        """LMCacheConnectorV1 forwards model to engine.register_model."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )

        mock_engine = MagicMock()
        mock_model = MagicMock(spec=torch.nn.Module)

        connector = LMCacheConnectorV1.__new__(LMCacheConnectorV1)
        connector._lmcache_engine = mock_engine
        connector.initialize_worker_connector(
            WorkerConnectorInitializationData(model=mock_model)
        )

        mock_engine.register_model.assert_called_once_with(mock_model)

    def test_initialize_worker_connector_skips_engine_without_method(self):
        """No-op when engine lacks register_model."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )

        mock_engine = MagicMock(spec=[])  # empty spec — no attributes
        mock_model = MagicMock(spec=torch.nn.Module)

        connector = LMCacheConnectorV1.__new__(LMCacheConnectorV1)
        connector._lmcache_engine = mock_engine
        # Should not raise
        connector.initialize_worker_connector(
            WorkerConnectorInitializationData(model=mock_model)
        )

    def test_initialize_worker_connector_skips_when_model_none(self):
        """Does not call register_model when model is None."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )

        mock_engine = MagicMock()
        connector = LMCacheConnectorV1.__new__(LMCacheConnectorV1)
        connector._lmcache_engine = mock_engine
        connector.initialize_worker_connector(WorkerConnectorInitializationData())
        mock_engine.register_model.assert_not_called()


class TestLMCacheConnectorV1ImplInitializeWorkerConnector:
    """Test LMCacheConnectorV1Impl initialize_worker_connector logic."""

    def test_registers_model_with_vllm_model_tracker(self):
        """Impl registers model with VLLMModelTracker when model provided."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )

        mock_tracker = MagicMock()
        mock_model = MagicMock(spec=torch.nn.Module)

        with patch.dict(
            "sys.modules",
            {
                "lmcache": MagicMock(),
                "lmcache.v1": MagicMock(),
                "lmcache.v1.compute": MagicMock(),
                "lmcache.v1.compute.models": MagicMock(),
                "lmcache.v1.compute.models.utils": MagicMock(
                    VLLMModelTracker=mock_tracker
                ),
            },
        ):
            from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter import (  # noqa: E501
                LMCacheConnectorV1Impl,
            )

            impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
            impl.initialize_worker_connector(
                WorkerConnectorInitializationData(model=mock_model)
            )

        mock_tracker.register_model.assert_called_once()
        call_args = mock_tracker.register_model.call_args
        assert call_args[0][1] is mock_model

    def test_graceful_on_import_error(self):
        """Doesn't crash if LMCache CacheBlend not installed."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter import (  # noqa: E501
            LMCacheConnectorV1Impl,
        )

        impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)

        with patch.dict(
            "sys.modules",
            {"lmcache.v1.compute.models.utils": None},
        ):
            impl.initialize_worker_connector(
                WorkerConnectorInitializationData(model=MagicMock(spec=torch.nn.Module))
            )

    def test_skips_tracker_when_model_none(self):
        """Does not call VLLMModelTracker when model is None."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )

        mock_tracker = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "lmcache": MagicMock(),
                "lmcache.v1": MagicMock(),
                "lmcache.v1.compute": MagicMock(),
                "lmcache.v1.compute.models": MagicMock(),
                "lmcache.v1.compute.models.utils": MagicMock(
                    VLLMModelTracker=mock_tracker
                ),
            },
        ):
            from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter import (  # noqa: E501
                LMCacheConnectorV1Impl,
            )

            impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
            impl.initialize_worker_connector(WorkerConnectorInitializationData())

        mock_tracker.register_model.assert_not_called()


class TestMultiConnectorInitializeWorkerConnector:
    """Test MultiConnector delegates initialize_worker_connector to sub-connectors."""

    def test_delegates_to_all_sub_connectors(self):
        """MultiConnector.initialize_worker_connector calls each sub-connector."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            WorkerConnectorInitializationData,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )

        mock_model = MagicMock(spec=torch.nn.Module)
        sub1 = MagicMock()
        sub2 = MagicMock()

        connector = MultiConnector.__new__(MultiConnector)
        connector._connectors = [sub1, sub2]
        data = WorkerConnectorInitializationData(model=mock_model)
        connector.initialize_worker_connector(data)

        sub1.initialize_worker_connector.assert_called_once_with(data)
        sub2.initialize_worker_connector.assert_called_once_with(data)
