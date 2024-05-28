import gc
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
from unittest.mock import MagicMock, patch

import openai
import pytest
import ray
import torch
from tensorizer import EncryptionParams

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
# yapf: disable
from vllm.model_executor.model_loader.tensorizer import (TensorizerConfig,
                                                         TensorSerializer,
                                                         is_vllm_tensorized,
                                                         load_with_tensorizer,
                                                         open_stream,
                                                         serialize_vllm_model,
                                                         tensorize_vllm_model)

from ..conftest import VllmRunner, cleanup
from ..utils import ServerRunner

# yapf conflicts with isort for this docstring


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)

model_ref = "facebook/opt-125m"
tensorize_model_for_testing_script = os.path.join(
    os.path.dirname(__file__), "tensorize_vllm_model_for_testing.py")


def is_curl_installed():
    try:
        subprocess.check_call(['curl', '--version'])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_torch_model(vllm_runner: VllmRunner):
    return vllm_runner \
            .model \
            .llm_engine \
            .model_executor \
            .driver_worker \
            .model_runner \
            .model

def write_keyfile(keyfile_path: str):
    encryption_params = EncryptionParams.random()
    pathlib.Path(keyfile_path).parent.mkdir(parents=True, exist_ok=True)
    with open(keyfile_path, 'wb') as f:
        f.write(encryption_params.key)

@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config


@patch('vllm.model_executor.model_loader.tensorizer.TensorizerAgent')
def test_load_with_tensorizer(mock_agent, tensorizer_config):
    mock_linear_method = MagicMock()
    mock_agent_instance = mock_agent.return_value
    mock_agent_instance.deserialize.return_value = MagicMock()

    result = load_with_tensorizer(tensorizer_config,
                                  quant_method=mock_linear_method)

    mock_agent.assert_called_once_with(tensorizer_config,
                                       quant_method=mock_linear_method)
    mock_agent_instance.deserialize.assert_called_once()
    assert result == mock_agent_instance.deserialize.return_value


@pytest.mark.skipif(not is_curl_installed(), reason="cURL is not installed")
def test_can_deserialize_s3(vllm_runner):
    model_ref = "EleutherAI/pythia-1.4b"
    tensorized_path = f"s3://tensorized/{model_ref}/fp16/model.tensors"

    loaded_hf_model = vllm_runner(model_ref,
                                  load_format="tensorizer",
                                  model_loader_extra_config=TensorizerConfig(
                                      tensorizer_uri=tensorized_path,
                                      num_readers=1,
                                      s3_endpoint="object.ord1.coreweave.com",
                                  ))

    deserialized_outputs = loaded_hf_model.generate(prompts, sampling_params)

    assert deserialized_outputs


@pytest.mark.skipif(not is_curl_installed(), reason="cURL is not installed")
def test_deserialized_encrypted_vllm_model_has_same_outputs(
        vllm_runner, tmp_path):
    vllm_model = vllm_runner(model_ref)
    model_path = tmp_path / (model_ref + ".tensors")
    key_path = tmp_path / (model_ref + ".key")
    write_keyfile(key_path)

    outputs = vllm_model.generate(prompts, sampling_params)

    config_for_serializing = TensorizerConfig(
        tensorizer_uri=model_path,
        encryption_keyfile=key_path
    )
    serialize_vllm_model(get_torch_model(vllm_model),
                         config_for_serializing)

    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()

    config_for_deserializing = TensorizerConfig(tensorizer_uri=model_path,
                                                encryption_keyfile=key_path)

    loaded_vllm_model = vllm_runner(
        model_ref,
        load_format="tensorizer",
        model_loader_extra_config=config_for_deserializing)

    deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)

    assert outputs == deserialized_outputs


def test_deserialized_hf_model_has_same_outputs(hf_runner, vllm_runner,
                                                tmp_path):
    hf_model = hf_runner(model_ref)
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
                                  load_format="tensorizer",
                                  model_loader_extra_config=TensorizerConfig(
                                      tensorizer_uri=model_path,
                                      num_readers=1,
                                  ))

    deserialized_outputs = loaded_hf_model.generate_greedy(
        prompts, max_tokens=max_tokens)

    assert outputs == deserialized_outputs


def test_vllm_model_can_load_with_lora(vllm_runner, tmp_path):
    from huggingface_hub import snapshot_download

    from examples.multilora_inference import (create_test_prompts,
                                              process_requests)

    model_ref = "meta-llama/Llama-2-7b-hf"
    lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    test_prompts = create_test_prompts(lora_path)

    # Serialize model before deserializing and binding LoRA adapters
    vllm_model = vllm_runner(model_ref, )
    model_path = tmp_path / (model_ref + ".tensors")

    serialize_vllm_model(get_torch_model(vllm_model),
                         TensorizerConfig(tensorizer_uri=model_path))

    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()
    loaded_vllm_model = vllm_runner(
        model_ref,
        load_format="tensorizer",
        model_loader_extra_config=TensorizerConfig(
            tensorizer_uri=model_path,
            num_readers=1,
        ),
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
        max_cpu_loras=2,
        max_num_seqs=50,
        max_model_len=1000,
    )
    process_requests(loaded_vllm_model.model.llm_engine, test_prompts)

    assert loaded_vllm_model


def test_load_without_tensorizer_load_format(vllm_runner):
    with pytest.raises(ValueError):
        vllm_runner(
            model_ref,
            model_loader_extra_config=TensorizerConfig(tensorizer_uri="test"))


@pytest.mark.skipif(not is_curl_installed(), reason="cURL is not installed")
def test_openai_apiserver_with_tensorizer(vllm_runner, tmp_path):
    ## Serialize model
    vllm_model = vllm_runner(model_ref, )
    model_path = tmp_path / (model_ref + ".tensors")

    serialize_vllm_model(get_torch_model(vllm_model),
                         TensorizerConfig(tensorizer_uri=model_path))

    model_loader_extra_config = {
        "tensorizer_uri": str(model_path),
    }

    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()

    ## Start OpenAI API server
    openai_args = [
        "--model", model_ref, "--dtype", "float16", "--load-format",
        "tensorizer", "--model-loader-extra-config",
        json.dumps(model_loader_extra_config), "--port", "8000"
    ]

    server = ServerRunner.remote(openai_args)

    assert ray.get(server.ready.remote())
    print("Server ready.")

    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    completion = client.completions.create(model=model_ref,
                                           prompt="Hello, my name is",
                                           max_tokens=5,
                                           temperature=0.0)

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1
    assert completion.choices[0].text is not None and len(
        completion.choices[0].text) >= 5
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11)


def test_raise_value_error_on_invalid_load_format(vllm_runner):
    with pytest.raises(ValueError):
        vllm_runner(
            model_ref,
            load_format="safetensors",
            model_loader_extra_config=TensorizerConfig(tensorizer_uri="test"))


@pytest.mark.skip("Requires multiple GPUs")
def test_tensorizer_with_tp_path_without_template(vllm_runner):
    with pytest.raises(ValueError):
        model_ref = "EleutherAI/pythia-1.4b"
        tensorized_path = f"s3://tensorized/{model_ref}/fp16/model.tensors"

        vllm_runner(
            model_ref,
            load_format="tensorizer",
            model_loader_extra_config=TensorizerConfig(
                tensorizer_uri=tensorized_path,
                num_readers=1,
                s3_endpoint="object.ord1.coreweave.com",
            ),
            tensor_parallel_size=2,
        )

@pytest.mark.skip("Requires multiple GPUs")
def test_deserialized_encrypted_vllm_model_with_tp_has_same_outputs(vllm_runner,
                                                                    tmp_path):
    model_ref = "EleutherAI/pythia-1.4b"
    # record outputs from un-sharded un-tensorized model
    base_model = vllm_runner(
        model_ref,
    )
    outputs = base_model.generate(prompts, sampling_params)

    base_model.model.llm_engine.model_executor.shutdown()
    del base_model
    cleanup()

    # load model with two shards and serialize with encryption
    model_path = str(tmp_path / (model_ref + "-%02d.tensors"))
    key_path = tmp_path / (model_ref + ".key")

    tensorizer_config = TensorizerConfig(
        tensorizer_uri=model_path,
        encryption_keyfile=key_path,
    )
    # FIXME: launching multiple distributed engines within the same program
    # results in a hang... launch serialization in a separate process as a work
    # around
    serialization_proc = mp.get_context('spawn').Process(
        target=tensorize_vllm_model,
        kwargs={
            "engine_args": EngineArgs(
                model=model_ref,
                tensor_parallel_size=2,
                enforce_eager=True,
            ),
            "tensorizer_config": tensorizer_config,
        },
    )
    serialization_proc.start()
    serialization_proc.join()
    assert os.path.isfile(model_path % 0), "Serialization subprocess failed"
    assert os.path.isfile(model_path % 1), "Serialization subprocess failed"

    loaded_vllm_model = vllm_runner(
        model_ref,
        tensor_parallel_size=2,
        load_format="tensorizer",
        enforce_eager=True,
        model_loader_extra_config=tensorizer_config)

    deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)

    assert outputs == deserialized_outputs


def test_vllm_tensorized_model_has_same_outputs(vllm_runner, tmp_path):
    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(tensorizer_uri=str(model_path))

    vllm_model = vllm_runner(model_ref)
    outputs = vllm_model.generate(prompts, sampling_params)
    serialize_vllm_model(get_torch_model(vllm_model), config)

    assert is_vllm_tensorized(config)
    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()

    loaded_vllm_model = vllm_runner(model_ref,
                                    load_format="tensorizer",
                                    model_loader_extra_config=config)
    deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)

    assert outputs == deserialized_outputs
