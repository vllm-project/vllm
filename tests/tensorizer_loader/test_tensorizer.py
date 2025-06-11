# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import copy
import functools
import gc
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Type, Union, Any
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest
import torch

from vllm import SamplingParams, LLM
from vllm.engine.arg_utils import EngineArgs

# yapf: disable
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig,
    TensorSerializer,
    is_vllm_tensorized,
    load_with_tensorizer,
    open_stream,
    tensorize_vllm_model
)
from vllm.model_executor.model_loader.tensorizer_loader import (
    BLACKLISTED_TENSORIZER_ARGS)
import vllm.model_executor.model_loader.tensorizer
from .conftest import DummyExecutor
# yapf: enable
from vllm.utils import (
    PlaceholderModule, get_distributed_init_method,
    get_open_port, get_ip,
)
from .conftest import assert_from_collective_rpc

from ..utils import VLLM_PATH

try:
    import tensorizer
    from tensorizer import EncryptionParams
except ImportError:
    tensorizer = PlaceholderModule("tensorizer")  # type: ignore[assignment]
    EncryptionParams = tensorizer.placeholder_attr("EncryptionParams")


class TensorizerCaughtError(Exception):
    pass


EXAMPLES_PATH = VLLM_PATH / "examples"

pytest_plugins = "pytest_asyncio",

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


def patch_init_and_catch_error(self, obj, method_name, expected_error: Type[Exception]):
    original = getattr(obj, method_name, None)
    if original is None:
        raise ValueError("Method '{}' not found.".format(method_name))

    def wrapper(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except expected_error:
            raise TensorizerCaughtError

    setattr(obj, method_name, wrapper)

    self.load_model()


def assert_specific_tensorizer_error_is_raised(
    executor,
    obj: Any,
    method_name: str,
    expected_error: Type[Exception],
    ):
    with pytest.raises(TensorizerCaughtError):
        executor.collective_rpc(patch_init_and_catch_error,
                                args=(obj, method_name, expected_error,))

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
    args = EngineArgs(model=model_ref)
    with vllm_runner(model_ref) as vllm_model:
        model_path = tmp_path / (model_ref + ".tensors")
        key_path = tmp_path / (model_ref + ".key")
        write_keyfile(key_path)

        outputs = vllm_model.generate(prompts, sampling_params)

    config_for_serializing = TensorizerConfig(tensorizer_uri=str(model_path),
                                              encryption_keyfile=str(key_path))

    tensorize_vllm_model(args, config_for_serializing)

    config_for_deserializing = TensorizerConfig(
        tensorizer_uri=str(model_path), encryption_keyfile=str(key_path))

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


def test_load_without_tensorizer_load_format(vllm_runner, capfd):
    model = None
    try:
        model = vllm_runner(
            model_ref,
            model_loader_extra_config=TensorizerConfig(tensorizer_uri="test"))
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert ("ValueError: Model loader extra config "
                "is not supported for load "
                "format LoadFormat.AUTO") in combined_output
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def test_raise_value_error_on_invalid_load_format(vllm_runner, capfd):
    model = None
    try:
        model = vllm_runner(
            model_ref,
            load_format="safetensors",
            model_loader_extra_config=TensorizerConfig(tensorizer_uri="test"))
    except RuntimeError:
        out, err = capfd.readouterr()

        combined_output = out + err
        assert ("ValueError: Model loader extra config is not supported "
                "for load format LoadFormat.SAFETENSORS") in combined_output
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
def test_tensorizer_with_tp_path_without_template(vllm_runner, capfd):
    try:
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
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert ("ValueError: For a sharded model, tensorizer_uri "
                "should include a string format template like '%04d' "
                "to be formatted with the rank "
                "of the shard") in combined_output


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

    # load model with two shards and serialize with encryption
    model_path = str(tmp_path / (model_ref + "-%02d.tensors"))
    key_path = tmp_path / (model_ref + ".key")

    tensorizer_config = TensorizerConfig(
        tensorizer_uri=model_path,
        encryption_keyfile=str(key_path),
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


@pytest.mark.flaky(reruns=3)
def test_vllm_tensorized_model_has_same_outputs(vllm_runner, tmp_path):
    gc.collect()
    torch.cuda.empty_cache()
    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(tensorizer_uri=str(model_path))
    args = EngineArgs(model=model_ref)

    with vllm_runner(model_ref) as vllm_model:
        outputs = vllm_model.generate(prompts, sampling_params)

    tensorize_vllm_model(args, config)
    assert is_vllm_tensorized(config)

    with vllm_runner(model_ref,
                     load_format="tensorizer",
                     model_loader_extra_config=config) as loaded_vllm_model:
        deserialized_outputs = loaded_vllm_model.generate(
            prompts, sampling_params)
        # noqa: E501

        assert outputs == deserialized_outputs

def test_assert_serialization_kwargs_passed_to_tensor_serializer(tmp_path):

    serialization_params = {
        "limit_cpu_concurrency": 2,
    }
    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(tensorizer_uri=str(model_path),
                              serialization_kwargs=serialization_params)
    llm = LLM(
        model=model_ref,
    )


    def serialization_test(self, *args, **kwargs):
        # This is performed in the ephemeral worker process, so monkey-patching
        # will actually work, and cleanup is guaranteed so don't
        # need to reset things

        original_dict = serialization_params
        to_compare = {}

        original = tensorizer.serialization.TensorSerializer.__init__

        def tensorizer_serializer_wrapper(self, *args, **kwargs):
            nonlocal to_compare
            to_compare = kwargs.copy()
            return original(self, *args, **kwargs)

        tensorizer.serialization.TensorSerializer.__init__ = tensorizer_serializer_wrapper

        tensorizer_config = TensorizerConfig(**kwargs["tensorizer_config"])
        self.save_tensorized_model(
            tensorizer_config=tensorizer_config, )
        return to_compare | original_dict == to_compare

    kwargs = {
        "tensorizer_config": config.to_dict()
    }

    assert assert_from_collective_rpc(llm, serialization_test, kwargs)


def test_assert_deserialization_kwargs_passed_to_tensor_deserializer(tmp_path, capfd):

    expected_error = TypeError

    deserialization_kwargs = {
        "num_readers": "bar", # illegal value
    }

    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(tensorizer_uri=str(model_path),
                              serialization_kwargs=serialization_params)

    args = EngineArgs(model=model_ref)
    tensorize_vllm_model(args, config)

    loader_tc = TensorizerConfig(
        tensorizer_uri=str(model_path),
        deserialization_kwargs=deserialization_kwargs,
    )

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        load_format = "tensorizer",
        model_loader_extra_config=loader_tc.to_dict(),)

    vllm_config = engine_args.create_engine_config()
    executor = DummyExecutor(vllm_config)

    assert_specific_tensorizer_error_is_raised(executor,
                                               tensorizer.serialization.TensorDeserializer,
                                               "__init__",
                                               TypeError,
                                               )

def test_assert_stream_kwargs_passed_to_tensor_deserializer(tmp_path, capfd):

    deserialization_kwargs = {
        "num_readers": 1,
    }

    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(tensorizer_uri=str(model_path),
                              serialization_kwargs=serialization_params)

    args = EngineArgs(model=model_ref)
    tensorize_vllm_model(args, config)

    stream_kwargs = {
        "mode": "foo"
    }


    loader_tc = TensorizerConfig(
        tensorizer_uri=str(model_path),
        deserialization_kwargs=deserialization_kwargs,
        stream_kwargs=stream_kwargs,
    )

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        load_format = "tensorizer",
        model_loader_extra_config=loader_tc.to_dict(),)

    vllm_config = engine_args.create_engine_config()
    executor = DummyExecutor(vllm_config)

    assert_specific_tensorizer_error_is_raised(
        executor,
        vllm.model_executor.model_loader.tensorizer,
        "open_stream",
        ValueError,
    )

@pytest.mark.asyncio
async def test_serialize_and_serve_entrypoints(tmp_path):
    model_ref = "facebook/opt-125m"

    suffix = "test"
    try:
        result = subprocess.run([
            sys.executable,
            f"{VLLM_PATH}/examples/others/tensorize_vllm_model.py", "--model",
            model_ref, "serialize", "--serialized-directory",
            str(tmp_path), "--suffix", suffix, "--serialization-kwargs",
            '{"limit_cpu_concurrency": 4}'
        ],
                                check=True,
                                capture_output=True,
                                text=True)
    except subprocess.CalledProcessError as e:
        print("Tensorizing failed.")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    assert "Successfully serialized" in result.stdout

    # Next, try to serve with vllm serve
    model_uri = tmp_path / "vllm" / model_ref / suffix / "model.tensors"

    model_loader_extra_config = {
        "tensorizer_uri": str(model_uri),
        "stream_kwargs": {
            "force_http": False,
        },
        "deserialization_kwargs": {
            "verify_hash": True,
            "num_readers": 8,
        }
    }

    cmd = [
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "--host",
        "localhost",
        "--load-format",
        "tensorizer",
        model_ref,
        "--model-loader-extra-config",
        json.dumps(model_loader_extra_config, indent=2)
    ]

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


    try:
        async with asyncio.timeout(180):
            await proc.stdout.readuntil(b"Application startup complete.")
    except asyncio.TimeoutError:
        pytest.fail("Server did not start successfully")
    finally:
        proc.terminate()
    await proc.communicate()

@pytest.mark.parametrize("illegal_value", BLACKLISTED_TENSORIZER_ARGS)
def test_blacklisted_parameter_for_loading(tmp_path, vllm_runner, capfd,
                                           illegal_value):

    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(tensorizer_uri=str(model_path),
                              serialization_kwargs=serialization_params)

    args = EngineArgs(model=model_ref)
    tensorize_vllm_model(args, config)

    loader_tc = {
        "tensorizer_uri": str(model_path),
        illegal_value: "foo"
    }

    try:
        vllm_runner(
            model_ref,
            load_format="tensorizer",
            model_loader_extra_config=loader_tc,
        )
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert (f"ValueError: {illegal_value} is not an allowed "
                f"Tensorizer argument.") in combined_output


