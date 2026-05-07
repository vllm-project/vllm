# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for mixing `prompt_embeds` with image content parts in a single
Chat Completions request.
"""

import json

import openai
import pytest
import pytest_asyncio
import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from tests.utils import RemoteOpenAIServer
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url
from vllm.utils.serial_utils import tensor2base64

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Use the model's native dtype to skip the implicit cast inside
# `safe_load_prompt_embeds` (mismatched floating-point dtypes are cast to the
# model's dtype automatically).
MODEL_DTYPE = torch.bfloat16


@pytest.fixture(scope="module")
def server_args() -> list[str]:
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.4",
        "--limit-mm-per-prompt",
        json.dumps({"image": 1}),
        "--enable-prompt-embeds",
        "--enable-mm-embeds",
    ]


@pytest.fixture(scope="module")
def server(server_args):
    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        max_wait_seconds=600,
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="module")
def image_url() -> str:
    """Stable real image as a data URL, kept identical across both the
    text and prompt_embeds requests so any output difference must come from
    how the text content is delivered."""
    return encode_image_url(ImageAsset("stop_sign").pil_image)


@pytest.fixture(scope="module")
def aligned_content_and_embeds_b64() -> tuple[str, str]:
    """`(content, base64_embeds)` where the embeddings are the model's
    embedding of `content` tokenized WITHOUT special tokens.

    Loads only the `embed_tokens` shard from disk on CPU instead of the full
    model on GPU, so the fixture has zero VRAM footprint and won't contend
    with the running vLLM server.
    """
    content = "Describe this image."
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    index_path = hf_hub_download(MODEL_NAME, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    embed_key = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
    shard_path = hf_hub_download(MODEL_NAME, weight_map[embed_key])
    with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor(embed_key)
    embed_layer = nn.Embedding.from_pretrained(embed_weight.to(MODEL_DTYPE))

    ids = tokenizer(content, add_special_tokens=False, return_tensors="pt").input_ids
    embeds = embed_layer(ids).squeeze(0)
    return content, tensor2base64(embeds)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_first",
    [True, False],
    ids=["image_url-then-text", "text-then-image_url"],
)
async def test_text_content_and_prompt_embeds_match_with_image_url(
    client: openai.AsyncOpenAI,
    image_url: str,
    aligned_content_and_embeds_b64: tuple[str, str],
    image_first: bool,
):
    """Same content as text vs `prompt_embeds` should yield identical Chat
    Completions output when mixed with an `image_url` part in the same
    message under greedy decoding.
    """
    content, encoded_text_embeds = aligned_content_and_embeds_b64

    image_part = {"type": "image_url", "image_url": {"url": image_url}}
    text_part = {"type": "text", "text": content}
    embeds_part = {"type": "prompt_embeds", "data": encoded_text_embeds}

    if image_first:
        text_content = [image_part, text_part]
        embeds_content = [image_part, embeds_part]
    else:
        text_content = [text_part, image_part]
        embeds_content = [embeds_part, image_part]

    text_resp = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=10,
        temperature=0.0,
        messages=[{"role": "user", "content": text_content}],
    )
    embeds_resp = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=10,
        temperature=0.0,
        messages=[{"role": "user", "content": embeds_content}],
    )

    text_out = text_resp.choices[0].message.content
    embeds_out = embeds_resp.choices[0].message.content
    assert text_out is not None and len(text_out) > 0
    assert embeds_out is not None and len(embeds_out) > 0
    assert text_out == embeds_out


@pytest.fixture(scope="module")
def image_embeds_b64() -> dict[str, str]:
    """Synthetic but stable `image_embeds` for Qwen2-VL."""
    grid = (1, 4, 4)
    spatial_merge_size = 2
    num_patches = (grid[1] // spatial_merge_size) * (grid[2] // spatial_merge_size)
    text_hidden_size = 1536  # Qwen2-VL-2B
    torch.manual_seed(0)
    return {
        "image_embeds": tensor2base64(
            torch.randn(num_patches, text_hidden_size, dtype=MODEL_DTYPE)
        ),
        "image_grid_thw": tensor2base64(torch.tensor(grid)),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_first",
    [True, False],
    ids=["image_embeds-then-text", "text-then-image_embeds"],
)
async def test_text_content_and_prompt_embeds_match_with_image_embeds(
    client: openai.AsyncOpenAI,
    image_embeds_b64: dict[str, str],
    aligned_content_and_embeds_b64: tuple[str, str],
    image_first: bool,
):
    """Same content as text vs `prompt_embeds` should yield identical Chat
    Completions output when mixed with a precomputed `image_embeds` part in
    the same message under greedy decoding.
    """
    content, encoded_text_embeds = aligned_content_and_embeds_b64

    image_part = {"type": "image_embeds", "image_embeds": image_embeds_b64}
    text_part = {"type": "text", "text": content}
    embeds_part = {"type": "prompt_embeds", "data": encoded_text_embeds}

    if image_first:
        text_content = [image_part, text_part]
        embeds_content = [image_part, embeds_part]
    else:
        text_content = [text_part, image_part]
        embeds_content = [embeds_part, image_part]

    text_resp = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=10,
        temperature=0.0,
        messages=[{"role": "user", "content": text_content}],
    )
    embeds_resp = await client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=10,
        temperature=0.0,
        messages=[{"role": "user", "content": embeds_content}],
    )

    text_out = text_resp.choices[0].message.content
    embeds_out = embeds_resp.choices[0].message.content
    assert text_out is not None and len(text_out) > 0
    assert embeds_out is not None and len(embeds_out) > 0
    assert text_out == embeds_out
