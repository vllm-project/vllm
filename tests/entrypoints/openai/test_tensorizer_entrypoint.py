# SPDX-License-Identifier: Apache-2.0

import gc
import json
import tempfile

import openai
import pytest
import pytest_asyncio
import torch.cuda

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig, tensorize_lora_adapter, tensorize_vllm_model)

from ...utils import RemoteOpenAIServer

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LORA_PATH = "yard1/llama-2-7b-sql-lora-test"


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def cleanup():
    _cleanup()


@pytest.fixture(scope='module')
def tmp_dir():
    with tempfile.TemporaryDirectory() as path:
        yield path


@pytest.fixture(scope='module')
def model_uri(tmp_dir):
    yield f"{tmp_dir}/model.tensors"


@pytest.fixture(scope="module")
def tensorize_model_and_lora(tmp_dir, model_uri):
    tensorizer_config = TensorizerConfig(tensorizer_uri=model_uri,
                                         lora_dir=tmp_dir)
    args = EngineArgs(model=MODEL_NAME)

    tensorize_lora_adapter(LORA_PATH, tensorizer_config)
    tensorize_vllm_model(args, tensorizer_config)

    # Manually invoke a _cleanup() here, as the cleanup()
    # fixture won't be guaranteed to be called after this
    # when this fixture is used for a test
    _cleanup()


@pytest.fixture(scope="module")
def server(model_uri, tensorize_model_and_lora):
    model_loader_extra_config = {
        "tensorizer_uri": model_uri,
    }

    ## Start OpenAI API server
    args = [
        "--load-format", "tensorizer", "--model-loader-extra-config",
        json.dumps(model_loader_extra_config), "--enable-lora"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_single_completion(client: openai.AsyncOpenAI, model_name: str):
    completion = await client.completions.create(model=model_name,
                                                 prompt="Hello, my name is",
                                                 max_tokens=5,
                                                 temperature=0.0)

    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1
    assert completion.model == MODEL_NAME
    assert len(completion.choices) == 1
    assert len(completion.choices[0].text) >= 5
    assert completion.choices[0].finish_reason == "length"
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=5, prompt_tokens=6, total_tokens=11)


def test_confirm_deserialize_and_serve(model_uri, tmp_dir,
                                       tensorize_model_and_lora):
    tc = TensorizerConfig(tensorizer_uri=model_uri, lora_dir=tmp_dir)
    llm = LLM(MODEL_NAME,
              load_format="tensorizer",
              model_loader_extra_config=tc,
              enable_lora=True)

    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=256,
                                     stop=["[/assistant]"])

    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
    ]

    llm.generate(prompts,
                 sampling_params,
                 lora_request=LoRARequest("sql-lora",
                                          1,
                                          tmp_dir,
                                          tensorizer_config=tc))
