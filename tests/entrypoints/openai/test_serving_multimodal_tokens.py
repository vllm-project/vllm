# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for multimodal features through the /inference/v1/generate endpoint.

Mirrors test_serving_tokens.py but exercises the multimodal piping
using Qwen/Qwen3-VL-2B-Instruct with pre-processed image features.
"""

import os

import httpx
import pytest
import pytest_asyncio
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from vllm.entrypoints.serve.disagg.mm_serde import encode_mm_kwargs_item
from vllm.entrypoints.serve.disagg.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
)
from vllm.multimodal.utils import encode_image_url

from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
GEN_ENDPOINT = "/inference/v1/generate"


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def model_config():
    return AutoConfig.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def test_image():
    return Image.new("RGB", (224, 224), color=(255, 0, 0))


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]

    envs = os.environ.copy()
    envs["VLLM_ROCM_USE_SKINNY_GEMM"] = "0"

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=envs) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server: RemoteOpenAIServer):
    transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
    headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
    async with httpx.AsyncClient(
        transport=transport,
        base_url=server.url_root,
        timeout=600,
        headers=headers,
    ) as c:
        yield c


def build_mm_payload(
    processor,
    model_config,
    image: Image.Image,
    text: str,
    sampling_params: dict,
    model_name: str = MODEL_NAME,
) -> dict:
    """Simulate what a coordinator does: pre-process an image with the HF
    processor and build a GenerateRequest payload with serialized features."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    token_ids = inputs["input_ids"][0].tolist()
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    image_token_id = model_config.image_token_id
    first_idx = token_ids.index(image_token_id)
    length = 0
    for i in range(first_idx, len(token_ids)):
        if token_ids[i] == image_token_id:
            length += 1
        else:
            break

    num_pixels = int(image_grid_thw[0].prod().item())
    mm_item = MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=pixel_values,
                field=MultiModalFlatField(slices=[slice(0, num_pixels)], dim=0),
            ),
            "image_grid_thw": MultiModalFieldElem(
                data=image_grid_thw[0],
                field=MultiModalBatchedField(keep_on_cpu=True),
            ),
        }
    )

    encoded = encode_mm_kwargs_item(mm_item)
    mm_hash = f"test_mm_hash_{id(image)}"

    features = MultiModalFeatures(
        mm_hashes={"image": [mm_hash]},
        mm_placeholders={
            "image": [PlaceholderRangeInfo(offset=first_idx, length=length)]
        },
        kwargs_data={"image": [encoded]},
    )

    return {
        "model": model_name,
        "token_ids": token_ids,
        "features": features.model_dump(),
        "sampling_params": sampling_params,
        "stream": False,
    }


@pytest.mark.asyncio
async def test_generate_endpoint(client, processor, model_config, test_image):
    payload = build_mm_payload(
        processor,
        model_config,
        test_image,
        "What color is this image? Answer in one word.",
        {"max_tokens": 10, "temperature": 0.0},
    )
    resp = await client.post(GEN_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) >= 1
    assert data["choices"][0]["token_ids"] is not None
    assert len(data["choices"][0]["token_ids"]) > 0

    text = processor.tokenizer.decode(
        data["choices"][0]["token_ids"], skip_special_tokens=True
    )
    assert "red" in text.lower(), (
        f"Expected model to identify the red image, got: {text!r}"
    )


RENDER_ENDPOINT = "/v1/chat/completions/render"


@pytest.mark.asyncio
async def test_render_to_generate_roundtrip(client, processor, test_image):
    """End-to-end: render a multimodal chat → feed into generate → decode."""
    data_url = encode_image_url(test_image, format="PNG")

    render_payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": "What color is this image? Answer in one word.",
                    },
                ],
            }
        ],
    }

    render_resp = await client.post(RENDER_ENDPOINT, json=render_payload)
    render_resp.raise_for_status()
    render_data = render_resp.json()

    # Validate render output structure
    assert "token_ids" in render_data
    assert "features" in render_data
    assert render_data["features"] is not None
    assert "kwargs_data" in render_data["features"]

    # Build generate request from render output
    generate_payload = render_data
    generate_payload["sampling_params"] = {
        "max_tokens": 10,
        "temperature": 0.0,
    }

    gen_resp = await client.post(GEN_ENDPOINT, json=generate_payload)
    gen_resp.raise_for_status()
    gen_data = gen_resp.json()

    assert "choices" in gen_data
    assert len(gen_data["choices"]) >= 1
    assert len(gen_data["choices"][0]["token_ids"]) > 0

    text = processor.tokenizer.decode(
        gen_data["choices"][0]["token_ids"], skip_special_tokens=True
    )
    assert "red" in text.lower(), (
        f"Expected model to identify the red image, got: {text!r}"
    )
