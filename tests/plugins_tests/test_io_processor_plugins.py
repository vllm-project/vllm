# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import IOProcessorResponse
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams

MODEL_NAME = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"

image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"  # noqa: E501


def test_loading_missing_plugin():
    vllm_config = VllmConfig()
    with pytest.raises(ValueError):
        get_io_processor(vllm_config, "wrong_plugin")


@pytest.fixture(scope="function")
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
        "prithvi_to_tiff",
        "--model-impl",
        "terratorch",
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
        "softmax": False
    }

    ret = requests.post(
        server.url_for("pooling"),
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

    # We just check that the output is a valid base64 string.
    # Raises an exception and fails the test if the string is corrupted.
    base64.b64decode(plugin_data["data"])


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_prithvi_mae_plugin_offline(vllm_runner, model_name: str):

    img_prompt = dict(
        data=image_url,
        data_format="url",
        image_format="tiff",
        out_data_format="b64_json",
    )

    pooling_params = PoolingParams(task="encode", softmax=False)

    with vllm_runner(
            model_name,
            runner="pooling",
            skip_tokenizer_init=True,
            trust_remote_code=True,
            enforce_eager=True,
            # Limit the maximum number of parallel requests
            # to avoid the model going OOM in CI.
            max_num_seqs=1,
            model_impl="terratorch",
            io_processor_plugin="prithvi_to_tiff",
    ) as llm_runner:
        pooler_output = llm_runner.get_llm().encode(
            img_prompt,
            pooling_params=pooling_params,
        )
    output = pooler_output[0].outputs

    # verify the output is formatted as expected for this plugin
    assert all(
        hasattr(output, attr)
        for attr in ["type", "format", "data", "request_id"])

    # We just check that the output is a valid base64 string.
    # Raises an exception and fails the test if the string is corrupted.
    base64.b64decode(output.data)
