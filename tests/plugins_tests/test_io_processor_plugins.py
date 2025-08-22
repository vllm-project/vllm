# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm import AsyncEngineArgs
from vllm.config import VllmConfig
from vllm.entrypoints.llm import LLM
from vllm.entrypoints.openai.protocol import IOProcessorResponse
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM"

image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"  # noqa: E501


def test_loading_missing_plugin():
    vllm_config = VllmConfig()
    with pytest.raises(ValueError):
        get_io_processor(vllm_config, "plugin")


def test_loading_engine_with_wrong_plugin():

    os.environ['VLLM_USE_V1'] = '1'

    engine_args = AsyncEngineArgs(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        enforce_eager=True,
        skip_tokenizer_init=True,
        io_processor_plugin="plugin")

    with pytest.raises(ValueError):
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_prithvi_mae_plugin_offline(model_name: str):

    img_prompt = dict(
        data=image_url,
        data_format="url",
        image_format="tiff",
        out_data_format="b64_json",
    )

    llm = LLM(
        model=model_name,
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        # Limit the maximum number of parallel requests
        # to avoid the model going OOM in CI.
        max_num_seqs=32,
        io_processor_plugin="prithvi_to_tiff_valencia",
    )

    pooling_params = PoolingParams(task="encode", softmax=False)
    output = llm.encode_with_io_processor(
        img_prompt,
        pooling_params=pooling_params,
    )

    # verify the output is formatted as expected for this plugin
    assert all(
        hasattr(output, attr)
        for attr in ["type", "format", "data", "request_id"])

    # verify the output image in base64 is of the corerct length
    assert len(output.data) == 218752


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--enforce-eager",
        "--trust-remote-code",
        "--skip-tokenizer-init",
        # Limit the maximum number of parallel requests
        # to avoid the model going OOM in CI.
        "--max-num-seqs",
        "32",
        "--io-processor-plugin",
        "prithvi_to_tiff_valencia"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_prithvi_mae_plugin_online(
    server: RemoteOpenAIServer,
    model_name: str,
):

    request_payload_url = {
        "data": {
            "data": image_url,
            "data_format": "url",
            "image_format": "tiff",
            "out_data_format": "b64_json",
        },
        "priority": 0,
        "model": model_name,
    }

    ret = requests.post(
        server.url_for("io_processor_pooling"),
        json=request_payload_url,
    )

    response = ret.json()

    # verify the request response is in the correct format
    assert (parsed_response := IOProcessorResponse(**response))

    # verify the output is formatted as expected for this plugin
    plugin_data = parsed_response.data
    assert all(
        plugin_data.get(attr)
        for attr in ["type", "format", "data", "request_id"])

    # verify the output image in base64 is of the corerct length
    assert len(plugin_data["data"]) == 218752
