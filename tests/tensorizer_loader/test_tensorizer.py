import gc
import json
import os
import pathlib
import subprocess
from functools import partial
from unittest.mock import MagicMock, patch

import openai
import pytest
import torch
from huggingface_hub import snapshot_download

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
# yapf conflicts with isort for this docstring
# yapf: disable
from vllm.model_executor.model_loader.tensorizer import (TensorizerConfig,
                                                         TensorSerializer,
                                                         is_vllm_tensorized,
                                                         load_with_tensorizer,
                                                         open_stream,
                                                         serialize_vllm_model,
                                                         tensorize_vllm_model)
# yapf: enable
from vllm.utils import PlaceholderModule, import_from_path

from ..utils import VLLM_PATH, RemoteOpenAIServer
from .conftest import retry_until_skip

try:
    from tensorizer import EncryptionParams
except ImportError:
    tensorizer = PlaceholderModule("tensorizer")  # type: ignore[assignment]
    EncryptionParams = tensorizer.placeholder_attr("EncryptionParams")

EXAMPLES_PATH = VLLM_PATH / "examples"

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


def write_keyfile(keyfile_path: str):
    encryption_params = EncryptionParams.random()
    pathlib.Path(keyfile_path).parent.mkdir(parents=True, exist_ok=True)
    with open(keyfile_path, 'wb') as f:
        f.write(encryption_params.key)


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

    with vllm_runner(model_ref,
                     load_format="tensorizer",
                     model_loader_extra_config=TensorizerConfig(
                         tensorizer_uri=tensorized_path,
                         num_readers=1,
                         s3_endpoint="object.ord1.coreweave.com",
                     )) as loaded_hf_model:
        deserialized_outputs = loaded_hf_model.generate(
            prompts, sampling_params)
        # noqa: E501

        assert deserialized_outputs


@pytest.mark.skipif(not is_curl_installed(), reason="cURL is not installed")
def test_deserialized_encrypted_vllm_model_has_same_outputs(
        vllm_runner, tmp_path):
    with vllm_runner(model_ref) as vllm_model:
        model_path = tmp_path / (model_ref + ".tensors")
        key_path = tmp_path / (model_ref + ".key")
        write_keyfile(key_path)

        outputs = vllm_model.generate(prompts, sampling_params)

        config_for_serializing = TensorizerConfig(tensorizer_uri=model_path,
                                                  encryption_keyfile=key_path)

        vllm_model.apply_model(
            partial(serialize_vllm_model,
                    tensorizer_config=config_for_serializing))

    config_for_deserializing = TensorizerConfig(tensorizer_uri=model_path,
                                                encryption_keyfile=key_path)

    with vllm_runner(model_ref,
                     load_format="tensorizer",
                     model_loader_extra_config=config_for_deserializing
                     ) as loaded_vllm_model:  # noqa: E501

        deserialized_outputs = loaded_vllm_model.generate(
            prompts, sampling_params)
        # noqa: E501

        assert outputs == deserialized_outputs


def test_deserialized_hf_model_has_same_outputs(hf_runner, vllm_runner,
                                                tmp_path):
    with hf_runner(model_ref) as hf_model:
        model_path = tmp_path / (model_ref + ".tensors")
        max_tokens = 50
        outputs = hf_model.generate_greedy(prompts, max_tokens=max_tokens)
        with open_stream(model_path, "wb+") as stream:
            serializer = TensorSerializer(stream)
            serializer.write_module(hf_model.model)

    with vllm_runner(model_ref,
                     load_format="tensorizer",
                     model_loader_extra_config=TensorizerConfig(
                         tensorizer_uri=model_path,
                         num_readers=1,
                     )) as loaded_hf_model:
        deserialized_outputs = loaded_hf_model.generate_greedy(
            prompts, max_tokens=max_tokens)

        assert outputs == deserialized_outputs


def test_vllm_model_can_load_with_lora(vllm_runner, tmp_path):
    multilora_inference = import_from_path(
        "examples.offline_inference.multilora_inference",
        EXAMPLES_PATH / "offline_inference/multilora_inference.py",
    )

    model_ref = "meta-llama/Llama-2-7b-hf"
    lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    test_prompts = multilora_inference.create_test_prompts(lora_path)

    # Serialize model before deserializing and binding LoRA adapters
    with vllm_runner(model_ref, ) as vllm_model:
        model_path = tmp_path / (model_ref + ".tensors")

        vllm_model.apply_model(
            partial(
                serialize_vllm_model,
                tensorizer_config=TensorizerConfig(tensorizer_uri=model_path)))

    with vllm_runner(
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
    ) as loaded_vllm_model:
        multilora_inference.process_requests(
            loaded_vllm_model.model.llm_engine, test_prompts)

        assert loaded_vllm_model


def test_load_without_tensorizer_load_format(vllm_runner):
    model = None
    with pytest.raises(ValueError):
        model = vllm_runner(
            model_ref,
            model_loader_extra_config=TensorizerConfig(tensorizer_uri="test"))
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not is_curl_installed(), reason="cURL is not installed")
def test_openai_apiserver_with_tensorizer(vllm_runner, tmp_path):
    ## Serialize model
    with vllm_runner(model_ref, ) as vllm_model:
        model_path = tmp_path / (model_ref + ".tensors")

        vllm_model.apply_model(
            partial(
                serialize_vllm_model,
                tensorizer_config=TensorizerConfig(tensorizer_uri=model_path)))

        model_loader_extra_config = {
            "tensorizer_uri": str(model_path),
        }

    ## Start OpenAI API server
    openai_args = [
        "--dtype",
        "float16",
        "--load-format",
        "tensorizer",
        "--model-loader-extra-config",
        json.dumps(model_loader_extra_config),
    ]

    with RemoteOpenAIServer(model_ref, openai_args) as server:
        print("Server ready.")

        client = server.get_client()
        completion = client.completions.create(model=model_ref,
                                               prompt="Hello, my name is",
                                               max_tokens=5,
                                               temperature=0.0)

        assert completion.id is not None
        assert len(completion.choices) == 1
        assert len(completion.choices[0].text) >= 5
        assert completion.choices[0].finish_reason == "length"
        assert completion.usage == openai.types.CompletionUsage(
            completion_tokens=5, prompt_tokens=6, total_tokens=11)


def test_raise_value_error_on_invalid_load_format(vllm_runner):
    model = None
    with pytest.raises(ValueError):
        model = vllm_runner(
            model_ref,
            load_format="safetensors",
            model_loader_extra_config=TensorizerConfig(tensorizer_uri="test"))
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
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
            disable_custom_all_reduce=True,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
def test_deserialized_encrypted_vllm_model_with_tp_has_same_outputs(
        vllm_runner, tmp_path):
    model_ref = "EleutherAI/pythia-1.4b"
    # record outputs from un-sharded un-tensorized model
    with vllm_runner(
            model_ref,
            disable_custom_all_reduce=True,
            enforce_eager=True,
    ) as base_model:
        outputs = base_model.generate(prompts, sampling_params)
        base_model.model.llm_engine.model_executor.shutdown()

    # load model with two shards and serialize with encryption
    model_path = str(tmp_path / (model_ref + "-%02d.tensors"))
    key_path = tmp_path / (model_ref + ".key")

    tensorizer_config = TensorizerConfig(
        tensorizer_uri=model_path,
        encryption_keyfile=key_path,
    )

    tensorize_vllm_model(
        engine_args=EngineArgs(
            model=model_ref,
            tensor_parallel_size=2,
            disable_custom_all_reduce=True,
            enforce_eager=True,
        ),
        tensorizer_config=tensorizer_config,
    )
    assert os.path.isfile(model_path % 0), "Serialization subprocess failed"
    assert os.path.isfile(model_path % 1), "Serialization subprocess failed"

    with vllm_runner(
            model_ref,
            tensor_parallel_size=2,
            load_format="tensorizer",
            disable_custom_all_reduce=True,
            enforce_eager=True,
            model_loader_extra_config=tensorizer_config) as loaded_vllm_model:
        deserialized_outputs = loaded_vllm_model.generate(
            prompts, sampling_params)

    assert outputs == deserialized_outputs


@retry_until_skip(3)
def test_vllm_tensorized_model_has_same_outputs(vllm_runner, tmp_path):
    gc.collect()
    torch.cuda.empty_cache()
    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(tensorizer_uri=str(model_path))

    with vllm_runner(model_ref) as vllm_model:
        outputs = vllm_model.generate(prompts, sampling_params)

        vllm_model.apply_model(
            partial(serialize_vllm_model, tensorizer_config=config))

        assert is_vllm_tensorized(config)

    with vllm_runner(model_ref,
                     load_format="tensorizer",
                     model_loader_extra_config=config) as loaded_vllm_model:
        deserialized_outputs = loaded_vllm_model.generate(
            prompts, sampling_params)
        # noqa: E501

        assert outputs == deserialized_outputs
