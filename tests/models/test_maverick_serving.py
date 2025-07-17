# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from transformers import AutoConfig

from vllm import LLM, SamplingParams

# Sample prompts for testing
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def test_maverick_serving():
    """Test Llama-4-Maverick model with vLLM LLM class using CLI equivalent
    options.
    """

    model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    print("Initializing vLLM with Maverick model...")
    print(f"Model: {model}")

    try:
        # Create an LLM with CLI equivalent parameters
        llm = LLM(
            model=model,
            max_model_len=8192,
            kv_cache_dtype="fp8",
            enable_expert_parallel=True,
            enforce_eager=True,  # for faster testing
            tensor_parallel_size=8,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
        )

        # Print model configuration
        print("Model loaded successfully!")
        print("\nModel Configuration:")
        print("-" * 40)

        model_config = llm.llm_engine.model_config
        print(f"Model: {model_config.model}")
        print(f"Architecture: {model_config.architectures}")
        print(f"Max Model Length: {model_config.max_model_len}")
        print(f"Dtype: {model_config.dtype}")
        print(f"Quantization: {model_config.quantization}")
        print(f"Trust Remote Code: {model_config.trust_remote_code}")
        print(f"Sliding Window: {model_config.get_sliding_window()}")
        print(f"Vocab Size: {model_config.get_vocab_size()}")
        print(f"Hidden Size: {model_config.get_hidden_size()}")

        # Print HuggingFace model cache path

        try:
            # Try to get the model path from HuggingFace cache
            hf_cache_dir = os.environ.get(
                "HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            print(f"HuggingFace Cache Directory: {hf_cache_dir}")

            # Try to load the config to get more info about the model path
            config = AutoConfig.from_pretrained(model_config.model,
                                                trust_remote_code=True)
            print(f"Model Config loaded from: {config.name_or_path}")

            # Check if there's a specific model path in the load config
            load_config = llm.llm_engine.vllm_config.load_config
            if (hasattr(load_config, "download_dir")
                    and load_config.download_dir):
                print(f"Custom Download Directory: {load_config.download_dir}")

        except Exception as e:
            print(f"Could not determine exact model path: {e}")

        # Try to find the actual cached model directory
        try:
            import glob

            model_name_safe = model_config.model.replace("/", "--")
            cache_pattern = os.path.join(hf_cache_dir, "hub",
                                         f"models--{model_name_safe}*")
            cached_dirs = glob.glob(cache_pattern)
            if cached_dirs:
                print(f"Cached Model Directory: {cached_dirs[0]}")
                # Look for the actual model files
                snapshot_dirs = glob.glob(
                    os.path.join(cached_dirs[0], "snapshots", "*"))
                if snapshot_dirs:
                    print(f"Model Snapshot Directory: {snapshot_dirs[0]}")
            else:
                print("No cached model directory found in standard HF "
                      "cache location")
        except Exception as e:
            print(f"Error finding cached model directory: {e}")

        # Print cache configuration
        cache_config = llm.llm_engine.cache_config
        print("\nCache Configuration:")
        print(f"Block Size: {cache_config.block_size}")
        print(f"GPU Memory Utilization: {cache_config.gpu_memory_utilization}")
        print(f"KV Cache Dtype: {cache_config.cache_dtype}")
        print(f"Enable Prefix Caching: {cache_config.enable_prefix_caching}")

        # Print parallel configuration
        parallel_config = llm.llm_engine.vllm_config.parallel_config
        print("\nParallel Configuration:")
        print(f"Tensor Parallel Size: {parallel_config.tensor_parallel_size}")
        print(
            f"Pipeline Parallel Size: {parallel_config.pipeline_parallel_size}"
        )
        print(
            f"Enable Expert Parallel: {parallel_config.enable_expert_parallel}"
        )
        print(f"Num Attention Heads: "
              f"{model_config.get_num_attention_heads(parallel_config)}")
        print(f"Num Key Value Heads: "
              f"{model_config.get_num_kv_heads(parallel_config)}")
        print(f"Num Layers: {model_config.get_num_layers(parallel_config)}")
        print("-" * 40)
        print("Generating text from sample prompts...")

        # Generate texts from the prompts
        outputs = llm.generate(prompts, sampling_params)

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


def main():
    """Main function to run the Maverick serving test."""
    test_maverick_serving()


if __name__ == "__main__":
    main()
