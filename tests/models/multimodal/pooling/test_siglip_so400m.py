# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import requests
from PIL import Image

from ....conftest import HfRunner, VllmRunner
from ...utils import check_embeddings_close

MODEL = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TEXTS = ["a photo of a dog", "a photo of a cat", "a person riding a bike"]


@pytest.mark.parametrize("model", [MODEL])
@pytest.mark.parametrize("dtype", ["half"])
def test_siglip_so400m_model(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
) -> None:
    """
    Test the SiglipSo400m pooling model by comparing its text and image
    embeddings with the original Hugging Face implementation.
    """
    image = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert("RGB")

    with vllm_runner(model,
                     runner="pooling",
                     dtype=dtype,
                     enforce_eager=True,
                     trust_remote_code=True) as vllm_model:
        vllm_outputs = vllm_model.embed(TEXTS, images=[image] * len(TEXTS))

    vllm_image_embeds = [o['image_embeds'] for o in vllm_outputs]
    vllm_text_embeds = [o['text_embeds'] for o in vllm_outputs]

    with hf_runner(model, dtype=dtype, trust_remote_code=True) as hf_model:
        hf_processor = hf_model.processor
        hf_model = hf_model.model

        image_inputs = hf_processor(images=image, return_tensors="pt").to(
            hf_model.device, dtype=hf_model.dtype)
        hf_image_embeds_tensor = hf_model.get_image_features(**image_inputs)

        text_inputs = hf_processor(text=TEXTS,
                                   padding=True,
                                   return_tensors="pt").to(hf_model.device)
        hf_text_embeds_tensor = hf_model.get_text_features(**text_inputs)

    hf_image_embeds = [hf_image_embeds_tensor[0].tolist()] * len(TEXTS)
    hf_text_embeds = [embed.tolist() for embed in hf_text_embeds_tensor]

    check_embeddings_close(
        embeddings_0_lst=hf_image_embeds,
        embeddings_1_lst=vllm_image_embeds,
        name_0="hf_image",
        name_1="vllm_image",
    )

    check_embeddings_close(
        embeddings_0_lst=hf_text_embeds,
        embeddings_1_lst=vllm_text_embeds,
        name_0="hf_text",
        name_1="vllm_text",
    )
