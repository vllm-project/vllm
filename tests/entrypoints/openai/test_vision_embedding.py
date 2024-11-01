from typing import Dict

import pytest
import pytest_asyncio
import requests

from vllm.multimodal.utils import encode_image_base64, fetch_image

from ...utils import VLLM_PATH, RemoteOpenAIServer

MODEL_NAME = "TIGER-Lab/VLM2Vec-Full"
MAXIMUM_IMAGES = 2

vlm2vec_jinja_path = VLLM_PATH / "examples/template_vlm2vec.jinja"
assert vlm2vec_jinja_path.exists()

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Venn_diagram_rgb.svg/1280px-Venn_diagram_rgb.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png",
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--task",
        "embedding",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        f"image={MAXIMUM_IMAGES}",
        "--chat-template",
        str(vlm2vec_jinja_path),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="session")
def base64_encoded_image() -> Dict[str, str]:
    return {
        image_url: encode_image_base64(fetch_image(image_url))
        for image_url in TEST_IMAGE_URLS
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_image_embedding(server: RemoteOpenAIServer, model_name: str,
                               image_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
                "text": "Represent the given image."
            },
        ],
    }]

    response = requests.post(server.url_for("v1/embeddings"),
                             json={
                                 "model": model_name,
                                 "messages": messages,
                                 "encoding_format": "float"
                             })
    response.raise_for_status()

    embeddings = response.json()
    assert embeddings["id"] is not None
    assert len(embeddings["data"]) == 1
    assert len(embeddings["data"][0]["embedding"]) == 3072
    assert embeddings["usage"]["completion_tokens"] == 0
    assert embeddings["usage"]["prompt_tokens"] == 762
    assert embeddings["usage"]["total_tokens"] == 762
