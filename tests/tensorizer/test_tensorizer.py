import gc
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm import SamplingParams
from vllm.config import ModelConfig, TensorizerConfig
from vllm.model_executor.tensorizer_loader import (
    EncryptionParams, TensorSerializer, is_vllm_serialized_tensorizer,
    load_with_tensorizer, open_stream)

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
    return config


@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm", vllm_tensorized=True)
    return config


@patch('vllm.model_executor.tensorizer_loader.TensorizerAgent')
def test_load_with_tensorizer(mock_agent, tensorizer_config):
    mock_linear_method = MagicMock()
    mock_agent_instance = mock_agent.return_value
    mock_agent_instance.deserialize.return_value = MagicMock()

    result = load_with_tensorizer(tensorizer_config,
                                  linear_method=mock_linear_method)

    mock_agent.assert_called_once_with(tensorizer_config,
                                       linear_method=mock_linear_method)
    mock_agent_instance.deserialize.assert_called_once()
    assert result == mock_agent_instance.deserialize.return_value


def test_is_vllm_model_with_vllm_in_uri(tensorizer_config):
    tensorizer_config.vllm_tensorized = True

    result = is_vllm_serialized_tensorizer(tensorizer_config)

    assert result is True


def test_is_vllm_model_without_vllm_in_uri(tensorizer_config):
    tensorizer_config.vllm_tensorized = False

    result = is_vllm_serialized_tensorizer(tensorizer_config)

    assert result is False


def test_deserialized_vllm_model_has_same_outputs(vllm_runner, tmp_path):
    vllm_model = vllm_runner(model_ref, dtype=dtype)
    model_path = tmp_path / (model_ref + ".tensors")
    outputs = vllm_model.generate(prompts, sampling_params)
    model = (vllm_model.model.llm_engine.model_executor.driver_worker.
             model_runner.model)
    with open_stream(model_path, "wb+") as stream:
        serializer = TensorSerializer(stream)
        serializer.write_module(model)
    del vllm_model, model
    gc.collect()
    torch.cuda.empty_cache()
    loaded_vllm_model = vllm_runner(model_ref,
                                    load_format="tensorizer",
                                    tensorizer_uri=model_path,
                                    num_readers=1,
                                    vllm_tensorized=True,
                                    dtype=dtype)
    deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)

    # Assumes SamplingParams being seeded ensures the outputs are deterministic
    assert outputs == deserialized_outputs


def test_can_deserialize_s3(vllm_runner, tmp_path):
    model_ref = "EleutherAI/pythia-1.4b"
    tensorized_path = f"s3://tensorized/{model_ref}/fp16/model.tensors"

    loaded_hf_model = vllm_runner(model_ref,
                                  tensorizer_uri=tensorized_path,
                                  load_format="tensorizer",
                                  num_readers=1,
                                  vllm_tensorized=False,
                                  dtype=dtype)
    deserialized_outputs = loaded_hf_model.generate(prompts, sampling_params)

    assert deserialized_outputs


def test_deserialized_encrypted_vllm_model_has_same_outputs(
        vllm_runner, tmp_path):
    vllm_model = vllm_runner(model_ref, dtype=dtype)
    model_path = tmp_path / (model_ref + ".tensors")
    key_path = tmp_path / (model_ref + ".key")
    outputs = vllm_model.generate(prompts, sampling_params)
    model = (vllm_model.model.llm_engine.model_executor.driver_worker.
             model_runner.model)

    encryption_params = EncryptionParams.random()
    with open_stream(model_path, "wb+") as stream:
        serializer = TensorSerializer(stream, encryption=encryption_params)
        serializer.write_module(model)
    with open_stream(key_path, "wb+") as stream:
        stream.write(encryption_params.key)
    del vllm_model, model
    gc.collect()
    torch.cuda.empty_cache()
    loaded_vllm_model = vllm_runner(model_ref,
                                    tensorizer_uri=model_path,
                                    load_format="tensorizer",
                                    encryption_keyfile=key_path,
                                    num_readers=1,
                                    vllm_tensorized=True,
                                    dtype=dtype)
    deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)

    # Assumes SamplingParams being seeded ensures the outputs are deterministic
    assert outputs == deserialized_outputs


def test_deserialized_hf_model_has_same_outputs(hf_runner, vllm_runner,
                                                tmp_path):
    hf_model = hf_runner(model_ref, dtype=dtype)
    model_path = tmp_path / (model_ref + ".tensors")
    max_tokens = 50
    outputs = hf_model.generate_greedy(prompts, max_tokens=max_tokens)
    with open_stream(model_path, "wb+") as stream:
        serializer = TensorSerializer(stream)
        serializer.write_module(hf_model.model)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    loaded_hf_model = vllm_runner(model_ref,
                                  tensorizer_uri=model_path,
                                  load_format="tensorizer",
                                  num_readers=1,
                                  vllm_tensorized=False,
                                  dtype=dtype)
    deserialized_outputs = loaded_hf_model.generate_greedy(
        prompts, max_tokens=max_tokens)

    assert outputs == deserialized_outputs


def test_vllm_model_with_lora_has_same_outputs(vllm_runner, tmp_path):
    from huggingface_hub import snapshot_download

    from examples.multilora_inference import (create_test_prompts,
                                              process_requests)

    model_ref = "meta-llama/Llama-2-7b-hf"
    lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    test_prompts = create_test_prompts(lora_path)

    vllm_model = vllm_runner(
        model_ref,
        dtype=dtype,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
        max_cpu_loras=2,
        max_num_seqs=50,
        max_model_len=1000,
    )
    model_path = tmp_path / (model_ref + ".tensors")
    outputs = process_requests(vllm_model.model.llm_engine, test_prompts)
    model = (vllm_model.model.llm_engine.model_executor.driver_worker.
             model_runner.model)
    with open_stream(model_path, "wb+") as stream:
        serializer = TensorSerializer(stream)
        serializer.write_module(model)
    del vllm_model, model
    gc.collect()
    torch.cuda.empty_cache()
    loaded_vllm_model = vllm_runner(model_ref,
                                    tensorizer_uri=model_path,
                                    load_format="tensorizer",
                                    num_readers=1,
                                    vllm_tensorized=True,
                                    enable_lora=True,
                                    max_loras=1,
                                    max_lora_rank=8,
                                    max_cpu_loras=2,
                                    max_num_seqs=50,
                                    max_model_len=1000,
                                    dtype=dtype)
    deserialized_outputs = process_requests(loaded_vllm_model.model.llm_engine,
                                            test_prompts)

    assert outputs == deserialized_outputs


def test_load_without_tensorizer_load_format(vllm_runner):
    with pytest.raises(ValueError):
        vllm_runner(model_ref, tensorizer_uri="test")
