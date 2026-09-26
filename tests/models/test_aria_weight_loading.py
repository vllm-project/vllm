from unittest.mock import MagicMock, patch

import pytest

from vllm.config import VllmConfig
from vllm.model_executor.models.aria import AriaForConditionalGeneration


@pytest.fixture(autouse=True)
def disable_global_cleanup(monkeypatch):
    monkeypatch.setenv("VLLM_SKIP_GLOBAL_CLEANUP", "1")


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


def test_aria_load_weights_returns_set():
    vllm_config = MagicMock(spec=VllmConfig)

    with patch(
        "vllm.model_executor.models.aria.AutoWeightsLoader"
    ) as mock_auto_loader_cls:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_weights.return_value = {"dummy_weight"}
        mock_auto_loader_cls.return_value = mock_loader_instance

        with patch.object(AriaForConditionalGeneration, "__init__", return_value=None):
            model = AriaForConditionalGeneration(vllm_config=vllm_config)
            model.hf_to_vllm_mapper = MagicMock()

            loaded = model.load_weights([])
            assert loaded == {"dummy_weight"}
