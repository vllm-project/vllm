# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import requests
from transformers import AutoProcessor

from tests.utils import VLLM_PATH, RemoteOpenAIServer
from vllm.entrypoints.pooling.embed.protocol import EmbeddingResponse
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.utils import fetch_image

MODEL_NAME = "TIGER-Lab/VLM2Vec-Full"
MAXIMUM_IMAGES = 2

vlm2vec_jinja_path = VLLM_PATH / "examples/template_vlm2vec_phi3v.jinja"
assert vlm2vec_jinja_path.exists()

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/RGBA_comp.png",
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "pooling",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"image": MAXIMUM_IMAGES}),
        "--chat-template",
        str(vlm2vec_jinja_path),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def get_hf_prompt_tokens(model_name, content, image_url):
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, num_crops=4
    )

    placeholder = "<|image_1|> "
    prompt = f"{placeholder}{content}"
    image = fetch_image(image_url)
    # Unwrap MediaWithBytes if present
    if isinstance(image, MediaWithBytes):
        image = image.media
    images = [image]
    inputs = processor(prompt, images, return_tensors="pt")
    return inputs.input_ids.shape[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_image_embedding(
    server: RemoteOpenAIServer, model_name: str, image_url: str
):
    content_text = "Represent the given image."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": content_text},
            ],
        }
    ]

    response = requests.post(
        server.url_for("v1/embeddings"),
        json={"model": model_name, "messages": messages, "encoding_format": "float"},
    )
    response.raise_for_status()
    embeddings = EmbeddingResponse.model_validate(response.json())

    hf_prompt_tokens = get_hf_prompt_tokens(model_name, content_text, image_url)

    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 3072
    assert embeddings.usage.completion_tokens == 0
    assert embeddings.usage.prompt_tokens == hf_prompt_tokens
    assert embeddings.usage.total_tokens == hf_prompt_tokens
