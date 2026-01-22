# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import LLM
from vllm.inputs.data import TextPrompt
from vllm.multimodal.utils import fetch_image

# Initialize model
model = LLM(
    model="jinaai/jina-embeddings-v4-vllm-text-matching",
    runner="pooling",
    max_model_len=1024,
    gpu_memory_utilization=0.8,
)

# Create text prompts
text1 = "Ein wunderschöner Sonnenuntergang am Strand"
text1_prompt = TextPrompt(prompt=f"Query: {text1}")

text2 = "浜辺に沈む美しい夕日"
text2_prompt = TextPrompt(prompt=f"Query: {text2}")

# Create image prompt
image = fetch_image(
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/eskimo.jpg"  # noqa: E501
)
image_prompt = TextPrompt(
    prompt="<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n",  # noqa: E501
    multi_modal_data={"image": image},
)

# Encode all prompts
prompts = [text1_prompt, text2_prompt, image_prompt]
outputs = model.encode(prompts, pooling_task="token_embed")


def get_embeddings(outputs):
    VISION_START_TOKEN_ID, VISION_END_TOKEN_ID = 151652, 151653

    embeddings = []
    for output in outputs:
        if VISION_START_TOKEN_ID in output.prompt_token_ids:
            # Gather only vision tokens
            img_start_pos = torch.where(
                torch.tensor(output.prompt_token_ids) == VISION_START_TOKEN_ID
            )[0][0]
            img_end_pos = torch.where(
                torch.tensor(output.prompt_token_ids) == VISION_END_TOKEN_ID
            )[0][0]
            embeddings_tensor = output.outputs.data.detach().clone()[
                img_start_pos : img_end_pos + 1
            ]
        else:
            # Use all tokens for text-only prompts
            embeddings_tensor = output.outputs.data.detach().clone()

        # Pool and normalize embeddings
        pooled_output = (
            embeddings_tensor.sum(dim=0, dtype=torch.float32)
            / embeddings_tensor.shape[0]
        )
        embeddings.append(torch.nn.functional.normalize(pooled_output, dim=-1))
    return embeddings


embeddings = get_embeddings(outputs)

for embedding in embeddings:
    print(embedding.shape)
