# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import io

import imagehash
import pytest
import requests
from PIL import Image

from tests.utils import RemoteOpenAIServer
from vllm.config import VllmConfig
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.plugins.io_processors import get_io_processor

models_config = {
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11": {
        "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",  # noqa: E501
        "out_hash": "aa6d92ad25926a5e",
        "plugin": "prithvi_to_tiff",
    },
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars": {
        "image_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/resolve/main/examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif",  # noqa: E501
        "out_hash": "c07f4f602da73552",
        "plugin": "prithvi_to_tiff",
    },
}


def _compute_image_hash(base64_data: str) -> str:
    # Decode the base64 output and create image from byte stream
    decoded_image = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(decoded_image))

    # Compute perceptual hash of the output image
    return str(imagehash.phash(image))


def test_loading_missing_plugin():
    vllm_config = VllmConfig()
    with pytest.raises(ValueError):
        get_io_processor(vllm_config, "wrong_plugin")


@pytest.fixture(scope="function")
def server(model_name, plugin):
    args = [
        "--runner",
        "pooling",
        "--enforce-eager",
        "--skip-tokenizer-init",
        # Limit the maximum number of parallel requests
        # to avoid the model going OOM in CI.
        "--max-num-seqs",
        "32",
        "--io-processor-plugin",
        plugin,
        "--enable-mm-embeds",
    ]

    with RemoteOpenAIServer(model_name, args) as remote_server:
        yield remote_server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name, image_url, plugin, expected_hash",
    [
        (model_name, config["image_url"], config["plugin"], config["out_hash"])
        for model_name, config in models_config.items()
    ],
)
async def test_prithvi_mae_plugin_online(
    server: RemoteOpenAIServer,
    model_name: str,
    image_url: str | dict,
    plugin: str,
    expected_hash: str,
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
        "softmax": False,
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
    assert all(plugin_data.get(attr) for attr in ["type", "format", "data"])

    # Compute the output image hash and compare it against the expected hash
    image_hash = _compute_image_hash(plugin_data["data"])
    assert image_hash == expected_hash, (
        f"Image hash mismatch: expected {expected_hash}, got {image_hash}"
    )


@pytest.mark.parametrize(
    "model_name, image_url, plugin, expected_hash",
    [
        (model_name, config["image_url"], config["plugin"], config["out_hash"])
        for model_name, config in models_config.items()
    ],
)
def test_prithvi_mae_plugin_offline(
    vllm_runner, model_name: str, image_url: str | dict, plugin: str, expected_hash: str
):
    img_data = dict(
        data=image_url,
        data_format="url",
        image_format="tiff",
        out_data_format="b64_json",
    )

    prompt = dict(data=img_data)

    with vllm_runner(
        model_name,
        runner="pooling",
        skip_tokenizer_init=True,
        enable_mm_embeds=True,
        enforce_eager=True,
        # Limit the maximum number of parallel requests
        # to avoid the model going OOM in CI.
        max_num_seqs=32,
        io_processor_plugin=plugin,
        default_torch_num_threads=1,
    ) as llm_runner:
        pooler_output = llm_runner.get_llm().encode(prompt, pooling_task="plugin")
    output = pooler_output[0].outputs

    # verify the output is formatted as expected for this plugin
    assert all(hasattr(output, attr) for attr in ["type", "format", "data"])

    # Compute the output image hash and compare it against the expected hash
    image_hash = _compute_image_hash(output.data)
    assert image_hash == expected_hash, (
        f"Image hash mismatch: expected {expected_hash}, got {image_hash}"
    )
