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
from vllm.entrypoints.serve.disagg.protocol import GenerateMultiModalFeature
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
)
from vllm.multimodal.utils import encode_image_url
from vllm.v1.engine.detokenizer import check_stop_strings

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

    kwargs_data = encode_mm_kwargs_item(mm_item)

    feature = GenerateMultiModalFeature(
        modality="image",
        mm_hash=f"test_mm_hash_{id(image)}",
        offset=first_idx,
        length=length,
        kwargs_data=kwargs_data,
    )

    return {
        "model": model_name,
        "token_ids": token_ids,
        "features": [feature.model_dump()],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("logprobs_value", [0, 1, 5])
async def test_generate_logprobs(
    client, processor, model_config, test_image, logprobs_value
):
    payload = build_mm_payload(
        processor,
        model_config,
        test_image,
        "What color is this?",
        {
            "max_tokens": 5,
            "temperature": 0.0,
            "logprobs": logprobs_value,
        },
    )
    resp = await client.post(GEN_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    assert choice["logprobs"] is not None
    logprobs_content = choice["logprobs"]["content"]
    assert len(logprobs_content) == len(choice["token_ids"])
    for entry in logprobs_content:
        assert "logprob" in entry
        assert len(entry["top_logprobs"]) >= 1
        assert len(entry["top_logprobs"]) == max(logprobs_value, 1)


@pytest.mark.asyncio
async def test_same_response_as_chat_completions(
    client, processor, model_config, test_image
):
    question = "What color is this image?"
    image_data_url = encode_image_url(test_image)

    payload = build_mm_payload(
        processor,
        model_config,
        test_image,
        question,
        {
            "max_tokens": 24,
            "temperature": 0.0,
            "detokenize": False,
        },
    )
    generate_resp = await client.post(GEN_ENDPOINT, json=payload)
    generate_resp.raise_for_status()
    generate_data = generate_resp.json()
    gen_token_ids = generate_data["choices"][0]["token_ids"]
    generate_res = processor.tokenizer.decode(gen_token_ids, skip_special_tokens=True)

    completions_payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": question},
                ],
            }
        ],
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": False,
    }
    completions_resp = await client.post(
        "/v1/chat/completions", json=completions_payload
    )
    completions_resp.raise_for_status()
    completions_data = completions_resp.json()
    completions_res = completions_data["choices"][0]["message"]["content"]

    assert generate_res == completions_res
    assert "red" in generate_res.lower(), (
        f"Expected model to identify the red image, got: {generate_res!r}"
    )


@pytest.mark.asyncio
async def test_stop_string_workflow(client, processor, model_config, test_image):
    question = "Describe the color of this image in detail."

    payload = build_mm_payload(
        processor,
        model_config,
        test_image,
        question,
        {
            "max_tokens": 48,
            "temperature": 0.0,
            "detokenize": False,
            "stop": ["red"],
        },
    )
    with pytest.raises(httpx.HTTPStatusError):
        generate_resp = await client.post(GEN_ENDPOINT, json=payload)
        generate_resp.raise_for_status()

    payload["sampling_params"]["stop"] = None
    generate_resp = await client.post(
        GEN_ENDPOINT, json=payload, headers={"X-Request-Id": "mm-42"}
    )
    generate_resp.raise_for_status()
    generate_data = generate_resp.json()
    generate_res = processor.tokenizer.decode(
        generate_data["choices"][0]["token_ids"], skip_special_tokens=True
    )

    stop_str, truncate_to = check_stop_strings(
        generate_res, len(generate_res), ["red"], False
    )
    if stop_str is not None:
        generate_res = generate_res[:truncate_to]

    image_data_url = encode_image_url(test_image)
    completions_payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": question},
                ],
            }
        ],
        "max_tokens": 48,
        "temperature": 0.0,
        "stream": False,
        "stop": ["red"],
    }
    completions_resp = await client.post(
        "/v1/chat/completions", json=completions_payload
    )
    completions_resp.raise_for_status()
    completions_data = completions_resp.json()
    completions_res = completions_data["choices"][0]["message"]["content"]

    assert generate_res == completions_res
