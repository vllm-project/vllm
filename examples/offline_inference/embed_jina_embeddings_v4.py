# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example of offline inference with Jina Embeddings V4 multimodal model.

This example demonstrates:
1. Text-only embeddings
2. Image-only embeddings
3. Cross-modal embeddings (text-to-image similarity)

The model supports both text and vision inputs through a unified architecture.
"""

import torch

from vllm import LLM
from vllm.config import PoolerConfig
from vllm.inputs.data import TextPrompt
from vllm.multimodal.utils import fetch_image


def get_embeddings(outputs):
    """Extract and normalize embeddings from model outputs."""
    VISION_START_TOKEN_ID, VISION_END_TOKEN_ID = 151652, 151653

    embeddings = []
    for output in outputs:
        if VISION_START_TOKEN_ID in output.prompt_token_ids:
            # For vision inputs, extract only vision token embeddings
            img_start_pos = output.prompt_token_ids.index(VISION_START_TOKEN_ID)
            img_end_pos = output.prompt_token_ids.index(VISION_END_TOKEN_ID)
            embeddings_tensor = output.outputs.data.detach().clone()[
                img_start_pos : img_end_pos + 1
            ]
        else:
            # For text-only inputs, use all token embeddings
            embeddings_tensor = output.outputs.data.detach().clone()

        # Pool and normalize embeddings
        pooled_output = embeddings_tensor.mean(dim=0, dtype=torch.float32)
        embeddings.append(torch.nn.functional.normalize(pooled_output, dim=-1))
    return embeddings


def main():
    # Initialize the model
    model = LLM(
        model="jinaai/jina-embeddings-v4-vllm-retrieval",
        task="embed",
        override_pooler_config=PoolerConfig(pooling_type="ALL", normalize=False),
        dtype="float16",
    )

    # Example 1: Text-only embeddings
    print("=== Text Embeddings ===")
    query = "Overview of climate change impacts on coastal cities"
    query_prompt = TextPrompt(prompt=f"Query: {query}")

    passage = """The impacts of climate change on coastal cities are significant
    and multifaceted. Rising sea levels threaten infrastructure, while increased
    storm intensity poses risks to populations and economies."""
    passage_prompt = TextPrompt(prompt=f"Passage: {passage}")

    # Generate embeddings
    text_outputs = model.encode([query_prompt, passage_prompt])
    text_embeddings = get_embeddings(text_outputs)

    # Calculate similarity
    similarity = torch.dot(text_embeddings[0], text_embeddings[1]).item()
    print(f"Query: {query[:50]}...")
    print(f"Passage: {passage[:50]}...")
    print(f"Similarity: {similarity:.4f}\n")

    # Example 2: Image embeddings
    print("=== Image Embeddings ===")
    # Fetch sample images
    image1_url = "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
    image2_url = "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png"

    image1 = fetch_image(image1_url)
    image2 = fetch_image(image2_url)

    # Create image prompts with the required format
    image1_prompt = TextPrompt(
        prompt="<|im_start|>user\n<|vision_start|><|image_pad|>"
        "<|vision_end|>Describe the image.<|im_end|>\n",
        multi_modal_data={"image": image1},
    )

    image2_prompt = TextPrompt(
        prompt="<|im_start|>user\n<|vision_start|><|image_pad|>"
        "<|vision_end|>Describe the image.<|im_end|>\n",
        multi_modal_data={"image": image2},
    )

    # Generate embeddings
    image_outputs = model.encode([image1_prompt, image2_prompt])
    image_embeddings = get_embeddings(image_outputs)

    # Calculate similarity
    similarity = torch.dot(image_embeddings[0], image_embeddings[1]).item()
    print(f"Image 1: {image1_url.split('/')[-1]}")
    print(f"Image 2: {image2_url.split('/')[-1]}")
    print(f"Similarity: {similarity:.4f}\n")

    # Example 3: Cross-modal similarity (text vs image)
    print("=== Cross-modal Similarity ===")
    query = "scientific paper with markdown formatting"
    query_prompt = TextPrompt(prompt=f"Query: {query}")

    # Generate embeddings for text query and second image
    cross_outputs = model.encode([query_prompt, image2_prompt])
    cross_embeddings = get_embeddings(cross_outputs)

    # Calculate cross-modal similarity
    similarity = torch.dot(cross_embeddings[0], cross_embeddings[1]).item()
    print(f"Text query: {query}")
    print(f"Image: {image2_url.split('/')[-1]}")
    print(f"Cross-modal similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
