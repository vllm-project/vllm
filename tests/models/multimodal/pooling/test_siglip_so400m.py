# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional

import pytest
import requests
import torch
import torch.nn.functional as F
from PIL import Image

from vllm.entrypoints.llm import LLM
from vllm.inputs import TextPrompt
from vllm.model_executor.utils import set_random_seed

from ....conftest import VllmRunner

MODEL = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"
TOKENIZER_ID = "google/siglip-base-patch16-224"

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
CORRECT_TEXT = "a photo of a cat"
INCORRECT_TEXT = "a photo of a dog"

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


def encode_multimodal(llm: LLM, prompts: list[Optional[str]],
                      images: list[Optional[Image.Image]]):
    batch_size = 0
    if prompts is not None:
        batch_size = len(prompts)
    elif images is not None:
        batch_size = len(images)
    else:
        return []

    inputs = []
    for i in range(batch_size):
        prompt_text = prompts[i] if prompts and prompts[i] is not None else ""
        image_data = images[i] if images else None

        inputs.append(
            TextPrompt(prompt=prompt_text,
                       multi_modal_data={'image': image_data}
                       if image_data else None))

    outputs = llm.encode(inputs)

    all_embeddings = []
    for output in outputs:
        squeezed_data = output.outputs.data.squeeze(1)
        all_embeddings.extend(squeezed_data.tolist())

    return all_embeddings


@pytest.mark.parametrize("model", [MODEL])
@pytest.mark.parametrize("dtype", ["half"])
def test_siglip_so400m_model_functionality(
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
) -> None:
    """
    The core functionality of the SiglipSo400m model 
    is tested by verifying the relative scores of image and text matching.
    """
    set_random_seed(0)
    image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert("RGB")

    with vllm_runner(model,
                     dtype=dtype,
                     enforce_eager=True,
                     trust_remote_code=True,
                     max_model_len=64,
                     tokenizer_name=TOKENIZER_ID,
                     gpu_memory_utilization=0.8) as vllm_model:

        vllm_text_embeds_list = encode_multimodal(
            llm=vllm_model.llm,
            prompts=[CORRECT_TEXT, INCORRECT_TEXT],
            images=None)

        vllm_image_embeds_list = encode_multimodal(llm=vllm_model.llm,
                                                   prompts=[None],
                                                   images=[image])

        image_embed = torch.tensor(vllm_image_embeds_list[0])
        correct_text_embed = torch.tensor(vllm_text_embeds_list[0])
        incorrect_text_embed = torch.tensor(vllm_text_embeds_list[1])

        sim_correct = F.cosine_similarity(image_embed,
                                          correct_text_embed,
                                          dim=0)
        sim_incorrect = F.cosine_similarity(image_embed,
                                            incorrect_text_embed,
                                            dim=0)

        assert sim_correct >= sim_incorrect, (
            "Model failed the sanity check: "
            "Correct text should have higher similarity than incorrect text. "
            f"(Got identical scores: {sim_correct.item()})")
