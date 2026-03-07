# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Create a reduced-layer version of the Maverick model for testing purposes.

This script creates a new model with fewer layers by:
1. Loading the original Maverick model configuration
2. Creating a reduced configuration
3. Generating compatible safetensors files with appropriate weights
4. Creating the necessary index files for vLLM compatibility
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    Llama4Config,
    Llama4ForConditionalGeneration,
)

from vllm import LLM, SamplingParams
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec, FullAttentionSpec

from ....utils import multi_gpu_test

# Sample prompts for testing
PROMPTS: list[str] = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def run_maverick_serving(model: str):
    """Test Llama-4-Maverick model with vLLM LLM class using CLI equivalent
    options with reduced layers.
    """

    try:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        llm = LLM(
            model=model,
            max_model_len=2048,
            enforce_eager=True,
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
            kv_cache_dtype="fp8",
        )

        outputs = llm.generate(PROMPTS, sampling_params)

        # Print the outputs
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)

    except Exception as e:
        print(f"Error initializing or running model: {e}")
        raise


def get_rope_layers_config(model_path: str) -> list[int]:
    """
    Get the interleaved RoPE configuration from HuggingFace config

    Args:
        model_path: Path to the local directory containing the reduced
            Maverick model checkpoint

    Returns:
        List of 0 or 1 indicating whether each layer uses RoPE and local attn
        0 indicates that RoPE is not used while 1 indicates that RoPE is used.
    """
    config_path = Path(model_path) / "config.json"
    model_config = json.loads(config_path.read_text())
    text_config = model_config["text_config"]
    no_rope_layers = text_config["no_rope_layers"]
    print(f"Found no_rope_layers: {no_rope_layers}")
    return no_rope_layers


def create_reduced_maverick_model(
    original_model_name: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    output_dir: str = "/tmp/reduced_maverick",
    text_layers: int = 4,
    num_experts: int = 4,
    vision_layers: int = 2,
    force_recreate: bool = False,
) -> str:
    """
    Create a reduced-layer version of the Maverick model.

    Args:
        original_model_name: Name of the original Maverick model
        output_dir: Directory to save the reduced model
        text_layers: Number of text transformer layers
        num_experts: Number of experts per layer
        vision_layers: Number of vision transformer layers
        force_recreate: Whether to recreate if output_dir already exists

    Returns:
        Path to the created reduced model directory
    """

    print(
        f"Creating reduced Maverick model with {text_layers} text layers and "
        f"{vision_layers} vision layers..."
    )

    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        if force_recreate:
            shutil.rmtree(output_path)
        else:
            print(
                f"Output directory {output_dir} already exists. "
                "Use --force-recreate to overwrite."
            )
            return str(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print("Loading original model configuration...")
        original_config = AutoConfig.from_pretrained(
            original_model_name, trust_remote_code=True
        )
        print("Creating reduced configuration...")
        reduced_config = create_reduced_config(
            original_config, text_layers, num_experts, vision_layers
        )

        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(reduced_config, f, indent=2)
        print(f"Saved reduced config to {config_path}")

        print("Copying tokenizer files...")
        copy_tokenizer_files(original_model_name, output_path)

        print("Creating reduced safetensors files...")
        create_reduced_safetensors(original_config, reduced_config, output_path)

        print("Creating preprocessor config...")
        create_preprocessor_config(original_config, output_path)

        try:
            gen_config = GenerationConfig.from_pretrained(original_model_name)
            gen_config.save_pretrained(output_path)
            print("Copied generation config")
        except Exception as e:
            print(f"Could not copy generation config: {e}")

        print(f"Successfully created reduced Maverick model at {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"Error creating reduced model: {e}")
        # Clean up on failure
        if output_path.exists():
            shutil.rmtree(output_path)
        raise


def create_reduced_config(
    original_config: Any, text_layers: int, num_experts: int, vision_layers: int
) -> dict[str, Any]:
    """Create a reduced configuration based on the original."""

    # Convert config to dictionary
    config_dict = original_config.to_dict()

    # Reduce text layers
    if "text_config" in config_dict:
        original_text_layers = config_dict["text_config"]["num_hidden_layers"]
        config_dict["text_config"]["num_hidden_layers"] = text_layers
        original_layer_types = config_dict["text_config"]["layer_types"]
        config_dict["text_config"]["layer_types"] = original_layer_types[:text_layers]
        print(f"Reduced text layers from {original_text_layers} to {text_layers}")

        original_num_experts = config_dict["text_config"]["num_local_experts"]
        config_dict["text_config"]["num_local_experts"] = num_experts
        print(f"Reduced num experts from {original_num_experts} to {num_experts}")

        hidden_dim_divisor = 4

        original_hidden_size = config_dict["text_config"]["hidden_size"]
        new_hidden_size = original_hidden_size // hidden_dim_divisor
        config_dict["text_config"]["hidden_size"] = new_hidden_size
        print(f"Reduced hidden size from {original_hidden_size} to {new_hidden_size}")

        original_head_dim = config_dict["text_config"]["head_dim"]
        new_head_dim = original_head_dim // hidden_dim_divisor
        config_dict["text_config"]["head_dim"] = new_head_dim
        print(f"Reduced head dim from {original_head_dim} to {new_head_dim}")

    # Reduce vision layers
    if "vision_config" in config_dict:
        original_vision_layers = config_dict["vision_config"]["num_hidden_layers"]
        config_dict["vision_config"]["num_hidden_layers"] = vision_layers
        print(f"Reduced vision layers from {original_vision_layers} to {vision_layers}")

    # Update model name to indicate it's a reduced version
    config_dict["_name_or_path"] = f"reduced_maverick_{text_layers}t_{vision_layers}v"

    return config_dict


def copy_tokenizer_files(original_model_name: str, output_path: Path) -> None:
    """Copy tokenizer files from the original model."""

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            original_model_name, trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)
        print("Tokenizer files copied successfully")
    except Exception as e:
        print(f"Warning: Could not copy tokenizer files: {e}")


def create_preprocessor_config(original_config: Any, output_path: Path) -> None:
    """Create preprocessor_config.json for multimodal model."""

    # Try to load the original preprocessor config
    try:
        processor = AutoProcessor.from_pretrained(
            original_config._name_or_path
            or "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            trust_remote_code=True,
        )
        processor.save_pretrained(output_path)
        print("Copied original preprocessor config")
        return
    except Exception as e:
        print(f"Could not copy original preprocessor config: {e}")
        raise


def create_reduced_safetensors(
    original_config: Any, reduced_config: dict[str, Any], output_path: Path
) -> None:
    """Create safetensors files with weights for the reduced model."""

    print("Generating synthetic weights for reduced model...")
    reduced_hf_config = Llama4Config(**reduced_config)
    model = Llama4ForConditionalGeneration._from_config(config=reduced_hf_config)

    print("Saving weights to safetensors files...")
    model.save_pretrained(output_path, safe_serialization="safetensors")


def check_attention_spec_interleaved_rope(
    llm: LLM,
    num_attention_layers: int,
    num_ranks: int,
    rope_layers: list[int],
):
    """Check that the attention spec is correct."""
    assert isinstance(llm.llm_engine.model_executor, Executor)
    kv_cache_specs_per_rank = llm.llm_engine.model_executor.get_kv_cache_specs()
    for rank in range(num_ranks):
        kv_cache_specs = kv_cache_specs_per_rank[rank]
        assert len(kv_cache_specs.keys()) == num_attention_layers
        for i in range(num_attention_layers):
            if rope_layers[i] == 0:
                expected_spec = FullAttentionSpec
            else:
                expected_spec = ChunkedLocalAttentionSpec
            assert isinstance(
                kv_cache_specs[f"language_model.model.layers.{i}.self_attn.attn"],
                expected_spec,
            )


def run_reduced_model(llm: LLM, should_profile: bool = False) -> None:
    """Test the created reduced model with vLLM."""
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

    if should_profile:
        llm.start_profile()
    outputs = llm.generate(PROMPTS, sampling_params)
    if should_profile:
        llm.stop_profile()

    print("Test generation successful!")
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")
        print("-" * 40)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "original_model_name,text_layers,num_experts,vision_layers,",
    [("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", 4, 4, 2)],
)
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("tp,ep", [(2, True)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dummy_maverick(
    monkeypatch,
    original_model_name: str,
    text_layers: int,
    num_experts: int,
    vision_layers: int,
    enforce_eager: bool,
    tp: int,
    ep: bool,
    output_dir: str = "/tmp/reduced_maverick",
    force_recreate: bool = True,
    profile: bool = False,
) -> None:
    # Disable multiprocessing allows us to access model executor from LLM engine
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    model_path = create_reduced_maverick_model(
        original_model_name=original_model_name,
        output_dir=output_dir,
        text_layers=text_layers,
        num_experts=num_experts,
        vision_layers=vision_layers,
        force_recreate=force_recreate,
    )

    print(f"\nReduced model created successfully at: {model_path}")

    rope_layers = get_rope_layers_config(model_path)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=512,  # Small context for testing
        gpu_memory_utilization=0.3,  # Conservative memory usage
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp,
        enable_expert_parallel=ep,
    )

    check_attention_spec_interleaved_rope(
        llm,
        text_layers,
        tp,
        rope_layers,
    )

    print(f"\nTesting reduced model at {model_path}...")
    run_reduced_model(llm=llm, should_profile=profile)


def main():
    """Main function to create and test the reduced model."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Create a reduced-layer Maverick model"
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/reduced_maverick",
        help="Output directory for the reduced model",
    )
    parser.add_argument(
        "--text-layers",
        type=int,
        default=4,
        help="Number of text transformer layers",
    )
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts")
    parser.add_argument(
        "--vision-layers",
        type=int,
        default=2,
        help="Number of vision transformer layers",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation if output directory exists",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the created model with vLLM"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile the created model with vLLM"
    )
    parser.add_argument(
        "--test-original",
        action="store_true",
        help="Test the original model with vLLM",
    )
    parser.add_argument(
        "--original-model",
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        help="Original model name to base the reduction on",
    )

    args = parser.parse_args()

    if args.test:
        test_dummy_maverick(
            original_model_name=args.original_model,
            output_dir=args.output_dir,
            text_layers=args.text_layers,
            num_experts=args.num_experts,
            vision_layers=args.vision_layers,
            force_recreate=args.force_recreate,
            tp=2,
            ep=True,
            enforce_eager=True,
            profile=args.profile,
        )

    if args.test_original:
        run_maverick_serving(args.original_model)


if __name__ == "__main__":
    exit(main())
