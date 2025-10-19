#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example script demonstrating how to load models from OCI registries.

This example shows how to use vLLM to load models that are stored as OCI
(Open Container Initiative) artifacts in container registries.

Requirements:
    - vLLM with OCI support
    - Access to an OCI registry with model artifacts

OCI Model Format:
    Models are stored as OCI artifacts with:
    - Safetensors layers (application/vnd.docker.ai.safetensors)
      - Layer order defines shard order
    - Config tar layer (application/vnd.docker.ai.vllm.config.tar)
      - Contains tokenizer config, vocab files, etc.
      - Must be extracted after downloading

Model Reference Format:
    [registry/]repository[:tag|@digest]

    Examples:
        - "user/model:tag"              -> docker.io/user/model:tag
        - "docker.io/user/model:v1"     -> docker.io/user/model:v1
        - "ghcr.io/org/model@sha256:..." -> ghcr.io/org/model@sha256:...
"""

from vllm import LLM, SamplingParams


def example_basic_oci_loading():
    """Basic example of loading a model from OCI registry."""
    print("=" * 80)
    print("Example 1: Basic OCI Model Loading")
    print("=" * 80)

    # Initialize LLM with OCI model
    # The model will be downloaded from docker.io by default
    llm = LLM(
        model="aistaging/smollm2-vllm",
        load_format="oci",
        # Optional: specify download directory
        # download_dir="/path/to/cache"
    )

    # Generate text
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
    outputs = llm.generate(prompts, sampling_params)

    # Print outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print("-" * 80)


def example_explicit_registry():
    """Example using explicit registry specification."""
    print("=" * 80)
    print("Example 2: Explicit Registry Specification")
    print("=" * 80)

    # Load from GitHub Container Registry (ghcr.io)
    llm = LLM(
        model="ghcr.io/myorg/mymodel:v1.0",
        load_format="oci",
    )

    # Use the model as normal
    prompt = "What is the meaning of life?"
    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
    outputs = llm.generate([prompt], sampling_params)

    print(f"Prompt: {prompt!r}")
    print(f"Generated: {outputs[0].outputs[0].text!r}")


def example_with_digest():
    """Example using digest instead of tag."""
    print("=" * 80)
    print("Example 3: Using Digest Reference")
    print("=" * 80)

    # Load specific version by digest (immutable reference)
    llm = LLM(
        model="user/model@sha256:1234567890abcdef...",
        load_format="oci",
    )

    # Generate text
    prompt = "Once upon a time"
    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
    outputs = llm.generate([prompt], sampling_params)

    print(f"Prompt: {prompt!r}")
    print(f"Generated: {outputs[0].outputs[0].text!r}")


def example_caching_behavior():
    """Example demonstrating caching behavior."""
    print("=" * 80)
    print("Example 4: Caching Behavior")
    print("=" * 80)

    # First load - downloads from registry
    print("First load: Downloading from OCI registry...")
    _ = LLM(
        model="user/model:latest",
        load_format="oci",
    )

    # Second load - uses cached layers
    print("\nSecond load: Using cached layers...")
    _ = LLM(
        model="user/model:latest",
        load_format="oci",
    )

    print("Both loads use the same cached model data.")
    print("Cache location: ~/.cache/vllm/oci/ (by default)")


def example_private_registry_with_auth():
    """Example using a private registry with authentication."""
    print("=" * 80)
    print("Example 5: Private Registry with Authentication")
    print("=" * 80)

    # Before running this example, authenticate with docker login:
    # $ docker login ghcr.io
    # Enter your username and password/token

    print("Loading model from private registry...")
    print("Make sure you have run 'docker login ghcr.io' first")
    
    llm = LLM(
        model="ghcr.io/myorg/private-model:latest",
        load_format="oci",
    )

    # Use the model
    prompt = "Hello, world!"
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    outputs = llm.generate([prompt], sampling_params)

    print(f"Prompt: {prompt!r}")
    print(f"Generated: {outputs[0].outputs[0].text!r}")
    print("\nNote: Authentication is handled automatically via Docker config")


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("vLLM OCI Model Loader Examples")
    print("=" * 80)
    print("\n")

    # Note: These examples are for demonstration purposes.
    # Uncomment and modify with actual model references to run.

    print("To run these examples:")
    print("1. Ensure you have access to an OCI registry with model artifacts")
    print("2. Modify the model references to point to your models")
    print("3. For private registries, run 'docker login <registry>' first")
    print("4. Run: python examples/oci_model_example.py")
    print("\n")

    # Uncomment to run examples:
    # example_basic_oci_loading()
    # example_explicit_registry()
    # example_with_digest()
    # example_caching_behavior()
    # example_private_registry_with_auth()

    print("Examples completed successfully!")


if __name__ == "__main__":
    main()
