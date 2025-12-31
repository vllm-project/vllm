#!/usr/bin/env python3
"""
Example: Running vLLM in embedding-only mode (text mode) with embedding inputs

This example demonstrates:
1. How to start the server with embedding-only mode for memory savings
2. How to create and send embedding inputs via the OpenAI API
"""

import json
import torch
from openai import OpenAI
from vllm.utils.serial_utils import tensor2base64

# ============================================================================
# STEP 1: Start the server with this command:
# ============================================================================
"""
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --enable-mm-embeds \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --port 8000
"""

# ============================================================================
# STEP 2: Create a client and example embedding
# ============================================================================

def create_example_image_embedding(
    num_images: int = 1,
    feature_size: int = 256,  # Typical image feature size
    hidden_size: int = 4096,   # Hidden size for Qwen2.5-VL models
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Create a dummy image embedding tensor.
    
    Shape: (num_images, feature_size, hidden_size)
    For Qwen2.5-VL models: (1, 256, 4096)
    """
    # Create random embeddings (in practice, these would come from your encoder)
    embedding = torch.randn(
        num_images, 
        feature_size, 
        hidden_size,
        dtype=dtype
    )
    return embedding


def main():
    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",  # Not needed for local server
        base_url="http://localhost:8000/v1",
    )
    
    # Get the model name from the server (must match the model the server was started with)
    try:
        models = client.models.list()
        if models.data:
            model_name = models.data[0].id
            print(f"Using model from server: {model_name}")
        else:
            raise ValueError("No models found on server")
    except Exception as e:
        print(f"Warning: Could not get model from server: {e}")
        print("Using default model name. Make sure it matches your server.")
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # Fallback: must match server
    
    # Create example image embedding
    # Shape: (num_images, feature_size, hidden_size)
    # For Qwen2.5-VL models, typical shape is (1, 256, 4096)
    print("Creating example image embedding...")
    image_embeds = create_example_image_embedding(
        num_images=1,
        feature_size=256,
        hidden_size=4096,
        dtype=torch.float16
    )
    print(f"Embedding shape: {image_embeds.shape}")
    
    # Convert to base64 for API transmission
    base64_image_embedding = tensor2base64(image_embeds)
    print("Embedding serialized to base64")
    
    # Send request with embedding input
    print("\nSending chat completion request...")
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Describe it briefly."
                    },
                    {
                        "type": "image_embeds",
                        "image_embeds": base64_image_embedding,
                    },
                ],
            },
        ],
        max_tokens=100,
        temperature=0.7,
    )
    
    print("\n" + "="*60)
    print("Response:")
    print("="*60)
    print(chat_completion.choices[0].message.content)
    print("="*60)


if __name__ == "__main__":
    main()

