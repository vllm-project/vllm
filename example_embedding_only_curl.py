#!/usr/bin/env python3
"""
Helper script to generate a curl command for testing embedding-only mode.

This script creates an example embedding and outputs a curl command that you can run.
"""

import json
import torch
from vllm.utils.serial_utils import tensor2base64

def create_example_image_embedding(
    num_images: int = 1,
    feature_size: int = 256,
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Create a dummy image embedding tensor."""
    embedding = torch.randn(
        num_images, 
        feature_size, 
        hidden_size,
        dtype=dtype
    )
    return embedding

def main():
    # Create example embedding
    image_embeds = create_example_image_embedding(
        num_images=1,
        feature_size=256,
        hidden_size=4096,
        dtype=torch.float16
    )
    
    # Convert to base64
    base64_image_embedding = tensor2base64(image_embeds)
    
    # Create the request payload
    payload = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "messages": [
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
        "max_tokens": 100,
        "temperature": 0.7,
    }
    
    # Print curl command
    print("="*70)
    print("CURL COMMAND:")
    print("="*70)
    print("\ncurl -X POST http://localhost:8000/v1/chat/completions \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{}'".format(json.dumps(payload, indent=2)))
    print("\n" + "="*70)
    print("\nNote: The embedding is included in the JSON payload above.")
    print("Embedding shape:", image_embeds.shape)
    print("="*70)

if __name__ == "__main__":
    main()

