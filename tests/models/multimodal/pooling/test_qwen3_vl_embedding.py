# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Qwen3-VL-Embedding embeddings from vLLM and Hugging Face.

The model card preprocesses media with `qwen_vl_utils`. This reference uses the
transformers processors, as vLLM does, so the test covers the model and its
last-token pooling.
"""

from typing import Any, NamedTuple

import numpy.typing as npt
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText

from vllm.assets.video import VideoAsset
from vllm.tokenizers import TokenizerLike

from ....conftest import HfRunner, ImageTestAssets, VllmRunner
from ...utils import check_embeddings_close
from ..generation.vlm_utils import model_utils

MODELS = ["Qwen/Qwen3-VL-Embedding-2B"]
INSTRUCTION = "Represent the user's input."
VIDEO_ASSET = VideoAsset("baby_reading", num_frames=8)


class EmbedInput(NamedTuple):
    text: str | None = None
    images: list[Image.Image] | None = None
    video: tuple[npt.NDArray, dict[str, Any]] | None = None


def _build_prompt(tokenizer: TokenizerLike, item: EmbedInput) -> str:
    # Same layout as the model card: video, then images, then text.
    content: list[dict[str, Any]] = []
    if item.video is not None:
        content.append({"type": "video"})
    content.extend({"type": "image"} for _ in item.images or [])
    if item.text is not None:
        content.append({"type": "text", "text": item.text})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(prompt, str)
    return prompt


@pytest.mark.parametrize("model", MODELS)
# In bfloat16, rounding alone moves these embeddings up to 1e-3 (cosine) away
# from float32, in HF and vLLM alike. In float32 the two agree within 1e-4.
@pytest.mark.parametrize("dtype", ["float"])
def test_models(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
    model: str,
    dtype: str,
) -> None:
    stop_sign = image_assets[0].pil_image
    cherry_blossom = image_assets[1].pil_image
    video = (VIDEO_ASSET.np_ndarrays, VIDEO_ASSET.metadata)
    items = [
        EmbedInput(text="A woman playing with her dog on a beach at sunset."),
        EmbedInput(images=[stop_sign]),
        EmbedInput(text="A stop sign on a street corner.", images=[stop_sign]),
        EmbedInput(images=[stop_sign, cherry_blossom]),
        EmbedInput(video=video),
        EmbedInput(text="A child reading a book.", images=[stop_sign], video=video),
    ]
    images = [item.images for item in items]
    videos = [item.video for item in items]

    # Run vLLM first: it needs a process where CUDA is not initialized yet.
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=8192,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 2, "video": 1},
    ) as vllm_model:
        tokenizer = vllm_model.llm.get_tokenizer()
        prompts = [_build_prompt(tokenizer, item) for item in items]
        vllm_outputs = vllm_model.embed(prompts, images=images, videos=videos)

    # The model's Sentence Transformers config ends with a Normalize module.
    norms = torch.tensor(vllm_outputs).norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0, atol=1e-3)

    with hf_runner(
        model, dtype=dtype, auto_cls=AutoModelForImageTextToText
    ) as hf_model:
        hf_model = model_utils.qwen3_vl_patch_hf_runner(hf_model)
        all_inputs = hf_model.get_inputs(prompts, images=images, videos=videos)
        hf_outputs = []
        with torch.no_grad():
            for inputs in all_inputs:
                hidden_states = hf_model.model.model(
                    **hf_model.wrap_device(inputs)
                ).last_hidden_state
                pooled = F.normalize(hidden_states[0, -1].float(), dim=-1)
                hf_outputs.append(pooled.tolist())

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
