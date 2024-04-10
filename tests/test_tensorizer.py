from unittest.mock import MagicMock, patch

import pytest

from vllm import SamplingParams
from vllm.config import ModelConfig
from vllm.model_executor.tensorizer_loader import (TensorizerArgs,
                                                   _is_vllm_model,
                                                   load_with_tensorizer)
from tensorizer import TensorSerializer, stream_io
import gc
import torch

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)

model_ref = "facebook/opt-125m"

dtype = "bfloat16"


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
    model_config.tensorizer_args.vllm_tensorized = True

    result = _is_vllm_model(model_config)

    assert result is True


def test_is_vllm_model_without_vllm_in_uri(model_config):
    model_config.tensorizer_args.vllm_tensorized = False

    result = _is_vllm_model(model_config)

    assert result is False


def test_deserialized_vllm_model_has_same_outputs(vllm_runner, tmp_path):
    vllm_model = vllm_runner(model_ref, dtype=dtype)
    model_path = tmp_path / (model_ref + ".tensors")
    outputs = vllm_model.generate(prompts, sampling_params)
    model = (vllm_model.model.llm_engine.model_executor.driver_worker.
             model_runner.model)
    with stream_io.open_stream(model_path, "wb+") as stream:
        serializer = TensorSerializer(stream)
        serializer.write_module(model)
    del vllm_model, model
    gc.collect()
    torch.cuda.empty_cache()
    loaded_vllm_model = vllm_runner(model_ref,
                                    tensorizer_args=TensorizerArgs(
                                        tensorizer_uri=model_path,
                                        num_readers=1,
                                        vllm_tensorized=True),
                                    dtype=dtype)
    deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)

    # Assumes SamplingParams being seeded ensures the outputs are deterministic
    assert outputs == deserialized_outputs

def test_deserialized_hf_model_has_same_outputs(hf_runner, vllm_runner, tmp_path):
    hf_model = hf_runner(model_ref, dtype=dtype)
    model_path = tmp_path / (model_ref + ".tensors")
    max_tokens = 50
    outputs = hf_model.generate_greedy(prompts, max_tokens=max_tokens)
    with stream_io.open_stream(model_path, "wb+") as stream:
        serializer = TensorSerializer(stream)
        serializer.write_module(hf_model.model)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    loaded_hf_model = vllm_runner(model_ref,
                                    tensorizer_args=TensorizerArgs(
                                        tensorizer_uri=model_path,
                                        num_readers=1,
                                        vllm_tensorized=False),
                                    dtype=dtype)
    deserialized_outputs = loaded_hf_model.generate_greedy(prompts, max_tokens=max_tokens)

    assert outputs == deserialized_outputs
