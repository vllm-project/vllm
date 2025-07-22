# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_vllm_config():
    """Provides a mock VllmConfig object for tests."""
    config = MagicMock()
    config.model_config = MagicMock()
    config.cache_config = MagicMock()
    config.lora_config = MagicMock()
    config.load_config = MagicMock()
    config.parallel_config = MagicMock()
    config.scheduler_config = MagicMock()
    config.device_config = MagicMock()
    config.speculative_config = MagicMock()
    config.prompt_adapter_config = MagicMock()
    config.observability_config = MagicMock()
    config.parallel_config.rank = 0
    config.model_config.seed = 0
    config.model_config.trust_remote_code = False
    config.cache_config.cache_dtype = "auto"
    return config


# --- Test Suite ---


def test_tpu_worker_initialization(mock_vllm_config):
    """
    Tests that TPUWorker is initialized correctly based on whether
    tpu_commons is installed.
    """
    # Since we are modifying sys.modules, we need to ensure the module
    # is reloaded for the test.
    if "vllm.v1.worker.tpu_worker" in sys.modules:
        del sys.modules["vllm.v1.worker.tpu_worker"]

    is_tpu_commons_installed = importlib.util.find_spec(
        "tpu_commons") is not None

    if is_tpu_commons_installed:

        with patch("tpu_commons.worker.get_tpu_worker_cls"
                   ) as mock_get_worker_cls:
            # Arrange
            mock_worker_instance = MagicMock()
            mock_worker_cls = MagicMock(return_value=mock_worker_instance)
            mock_get_worker_cls.return_value = mock_worker_cls

            # Act
            from vllm.v1.worker import tpu_worker

            # The TPUWorker class in the module should now be the VllmTPUBackend
            assert tpu_worker.TPUWorker.__name__ == "VllmTPUBackend"

            # Instantiate the worker
            worker = tpu_worker.TPUWorker(
                vllm_config=mock_vllm_config,
                local_rank=0,
                rank=0,
                distributed_init_method="env://",
                is_driver_worker=True,
            )

            # Assert
            # 1. Assert it's an instance of the bridge
            assert isinstance(worker, tpu_worker.VllmTPUBackend)

            # 2. Assert the correct worker was created and initialized
            mock_get_worker_cls.assert_called_once()
            mock_worker_cls.assert_called_once_with(
                host_interface=None,
                vllm_config=mock_vllm_config,
                local_rank=0,
                rank=0,
                distributed_init_method="env://",
                is_driver_worker=True)

            # 3. Test pass-through methods
            worker.init_device()
            mock_worker_instance.init_device.assert_called_once()

            worker.load_model()
            mock_worker_instance.load_model.assert_called_once()

            worker.execute_model(MagicMock())
            mock_worker_instance.execute_model.assert_called_once()
    else:
        # Re-import the module to trigger the except block
        from vllm.v1.worker import tpu_worker

        # The TPUWorker class should be the original implementation
        assert tpu_worker.TPUWorker.__name__ == "TPUWorker"

        # Instantiate the worker
        worker = tpu_worker.TPUWorker(
            vllm_config=mock_vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
            is_driver_worker=True,
        )

        # Assert it's NOT an instance of the TPUBackend
        assert isinstance(worker, tpu_worker.TPUWorker)
