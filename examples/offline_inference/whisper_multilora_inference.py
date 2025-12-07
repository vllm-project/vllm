# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use multi-LoRA functionality with
Whisper models for speech-to-text transcription.

Usage:
    python whisper_multilora_inference.py

Note: Replace LORA_PATH with your actual LoRA adapter path.
If you don't have a LoRA adapter, the example will run with
the base model only.
"""

import os

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest


def create_whisper_prompt(language: str = "en") -> dict:
    """Create a Whisper transcription prompt with audio input.

    Args:
        language: ISO 639-1 language code (e.g., "en", "ko", "ja")

    Returns:
        Dictionary with prompt and multi-modal data
    """
    # Load sample audio from vLLM assets
    audio_asset = AudioAsset("mary_had_lamb")
    audio_data = audio_asset.audio_and_sample_rate

    # Whisper prompt format:
    # <|startoftranscript|><|language|><|task|><|notimestamps|>
    prompt = f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|>"

    return {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_data,
        },
    }


def run_base_model_inference(llm: LLM, sampling_params: SamplingParams) -> None:
    """Run inference using the base Whisper model without LoRA."""
    print("\n" + "=" * 60)
    print("Running inference with BASE MODEL (no LoRA)")
    print("=" * 60)

    inputs = create_whisper_prompt(language="en")
    outputs = llm.generate([inputs], sampling_params=sampling_params)

    for output in outputs:
        print(f"Transcription: {output.outputs[0].text}")


def run_lora_inference(
    llm: LLM,
    sampling_params: SamplingParams,
    lora_path: str,
    lora_name: str,
    lora_id: int,
) -> None:
    """Run inference using a specific LoRA adapter.

    Args:
        llm: The vLLM engine
        sampling_params: Sampling parameters
        lora_path: Path to the LoRA adapter
        lora_name: Name identifier for the LoRA
        lora_id: Unique integer ID for the LoRA
    """
    print("\n" + "=" * 60)
    print(f"Running inference with LoRA: {lora_name}")
    print("=" * 60)

    inputs = create_whisper_prompt(language="en")
    lora_request = LoRARequest(lora_name, lora_id, lora_path)

    outputs = llm.generate(
        [inputs],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    for output in outputs:
        print(f"Transcription: {output.outputs[0].text}")


def main():
    """Main function demonstrating Whisper Multi-LoRA inference."""
    # Initialize Whisper model with LoRA support enabled
    print("Initializing Whisper model with Multi-LoRA support...")
    llm = LLM(
        model="openai/whisper-large-v3-turbo",
        enable_lora=True,
        max_loras=4,  # Maximum number of LoRAs to keep in memory
        max_lora_rank=64,  # Maximum LoRA rank supported
        max_model_len=448,  # Whisper's max target positions
        dtype="half",
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=200,
    )

    # Run base model inference
    run_base_model_inference(llm, sampling_params)

    # Example LoRA paths - replace with your actual LoRA adapters
    lora_paths = [
        ("lora_adapter_1", "/path/to/your/lora_adapter_1"),
        ("lora_adapter_2", "/path/to/your/lora_adapter_2"),
    ]

    # Run inference with each LoRA adapter (if paths exist)
    for lora_id, (lora_name, lora_path) in enumerate(lora_paths, start=1):
        if os.path.exists(lora_path):
            run_lora_inference(llm, sampling_params, lora_path, lora_name, lora_id)
        else:
            print(f"\nSkipping {lora_name}: path does not exist ({lora_path})")
            print("To use LoRA adapters, update lora_paths with valid paths.")

    print("\n" + "=" * 60)
    print("Multi-LoRA inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
