import pytest
from unittest.mock import patch, MagicMock
from vllm.model_executor.tensorizer_loader import (load_with_tensorizer,
                                                   _is_vllm_model,
                                                   TensorizerArgs)
from vllm.config import ModelConfig


@pytest.fixture(autouse=True)
def model_config():
    config = ModelConfig(
        "Qwen/Qwen1.5-7B",
        "Qwen/Qwen1.5-7B",
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format="dummy",
        seed=0,
        dtype="float16",
        revision=None,
    )
    config.tensorizer_args = TensorizerArgs(tensorizer_uri="vllm", )
    return config


@patch('vllm.model_executor.tensorizer_loader.TensorizerAgent')
def test_load_with_tensorizer(mock_agent, model_config):
    mock_model_cls = MagicMock()
    mock_agent_instance = mock_agent.return_value
    mock_agent_instance.deserialize.return_value = MagicMock()

    result = load_with_tensorizer(mock_model_cls, model_config)

    mock_agent.assert_called_once_with(mock_model_cls, model_config)
    mock_agent_instance.deserialize.assert_called_once()
    assert result == mock_agent_instance.deserialize.return_value


def test_is_vllm_model_with_vllm_in_uri(model_config):
    model_config.tensorizer_args.tensorizer_uri = "vllm"

    result = _is_vllm_model(model_config)

    assert result is True


def test_is_vllm_model_without_vllm_in_uri(model_config):
    model_config.tensorizer_args.tensorizer_uri = "blabla"

    result = _is_vllm_model(model_config)

    assert result is False
