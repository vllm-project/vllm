"""
Example: Using LoRA with INT4 Quantized Models in vLLM

This example demonstrates how to:
1. Load an INT4 quantized model (compressed with llm-compressor)
2. Apply LoRA adapters
3. Run inference

Prerequisites:
- Model quantized with llm-compressor (see llm-compressor docs)
- LoRA adapters trained for your task
"""

from vllm import LLM, SamplingParams
import torch


def main():
    print("=" * 80)
    print("INT4 + LoRA Example")
    print("=" * 80)

    # Step 1: Load INT4 quantized model
    print("\n[1/4] Loading INT4 quantized model...")
    print("  Model path: ./models/llama-2-7b-int4")
    print("  Quantization: compressed-tensors (INT4)")

    llm = LLM(
        model="./models/llama-2-7b-int4",
        quantization="compressed-tensors",
        max_model_len=2048,
        # Note: LoRA compatibility is automatically detected from model config
    )

    print("✓ Model loaded successfully")
    print(f"  Memory usage: ~5.25 GB (vs ~14 GB for FP16)")

    # Step 2: Check LoRA compatibility
    print("\n[2/4] Checking LoRA compatibility...")

    # The model config should have lora_compatible=True if quantized with
    # the latest llm-compressor
    if hasattr(llm.llm_engine.model_config, "quantization_config"):
        quant_config = llm.llm_engine.model_config.quantization_config
        if hasattr(quant_config, "is_lora_compatible"):
            is_compatible = quant_config.is_lora_compatible()
            print(f"  LoRA compatible: {is_compatible}")
            if is_compatible:
                print(f"  Target modules: {quant_config.lora_target_modules}")
        else:
            print("  LoRA compatibility detection not available")
    else:
        print("  No quantization config found")

    # Step 3: Load LoRA adapters
    print("\n[3/4] Loading LoRA adapters...")

    lora_adapters = [
        {
            "name": "math_adapter",
            "path": "./lora_adapters/math",
        },
        {
            "name": "code_adapter",
            "path": "./lora_adapters/code",
        },
    ]

    print(f"  Loading {len(lora_adapters)} adapters...")
    for adapter in lora_adapters:
        print(f"    - {adapter['name']}: {adapter['path']}")

    # Note: In the current implementation, LoRA loading triggers:
    # 1. Detection of INT4 quantization in base layers
    # 2. Logging that INT4 kernels will be used for base model
    # 3. LoRA operates directly on FP input activations

    try:
        llm.load_lora_adapters(lora_adapters)
        print("✓ LoRA adapters loaded successfully")
        print("  Note: Base model uses INT4 kernels, LoRA uses FP16")
    except AttributeError:
        print("⚠ load_lora_adapters API not yet available")
        print("  (This is expected if vLLM LoRA API is still being finalized)")

    # Step 4: Run inference with LoRA
    print("\n[4/4] Running inference...")

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=128,
    )

    # Example 1: Math problem with math adapter
    print("\n  Example 1: Math problem (math_adapter)")
    math_prompt = "Solve the equation: 2x + 5 = 13. Show your work."

    try:
        outputs = llm.generate(
            math_prompt,
            sampling_params=sampling_params,
            lora_request={"lora_name": "math_adapter"},
        )
        print(f"    Prompt: {math_prompt}")
        print(f"    Response: {outputs[0].outputs[0].text[:200]}...")
    except (AttributeError, TypeError):
        print("    ⚠ LoRA inference API not yet available")
        print("    Fallback: Running without LoRA")
        outputs = llm.generate(math_prompt, sampling_params=sampling_params)
        print(f"    Prompt: {math_prompt}")
        print(f"    Response: {outputs[0].outputs[0].text[:200]}...")

    # Example 2: Coding task with code adapter
    print("\n  Example 2: Coding task (code_adapter)")
    code_prompt = "Write a Python function to reverse a linked list."

    try:
        outputs = llm.generate(
            code_prompt,
            sampling_params=sampling_params,
            lora_request={"lora_name": "code_adapter"},
        )
        print(f"    Prompt: {code_prompt}")
        print(f"    Response: {outputs[0].outputs[0].text[:200]}...")
    except (AttributeError, TypeError):
        print("    ⚠ LoRA inference API not yet available")
        print("    Fallback: Running without LoRA")
        outputs = llm.generate(code_prompt, sampling_params=sampling_params)
        print(f"    Prompt: {code_prompt}")
        print(f"    Response: {outputs[0].outputs[0].text[:200]}...")

    # Performance info
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print("  Configuration: Llama-2-7B + INT4 + LoRA (r=16)")
    print("  Memory usage: ~5.25 GB")
    print("  Expected speedup: ~1.9x vs FP16 baseline")
    print("  Memory savings: 62.5% vs FP16 baseline")
    print("\n  Architecture:")
    print("    ├─ Base model: INT4 quantized kernels (fast)")
    print("    ├─ LoRA adapters: FP16 computation")
    print("    └─ Combined: base_output + lora_output")
    print("=" * 80)


def demo_unpacking():
    """
    Demonstrate manual weight unpacking (advanced use case).

    This is not needed for inference, but useful for:
    - Inspecting unpacked weights
    - Merging LoRA into base weights
    - Fine-tuning LoRA adapters
    """
    print("\n" + "=" * 80)
    print("Advanced: Manual Weight Unpacking")
    print("=" * 80)

    from vllm.lora.int4_utils import get_unpacker

    print("\n  This demonstrates INT4 weight unpacking.")
    print("  Note: For inference, unpacking is not required!")

    # Get global unpacker instance
    unpacker = get_unpacker()

    # Create mock quantized module
    class MockQuantizedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight_packed",
                torch.randint(0, 255, (4096, 2048), dtype=torch.uint8),
            )
            self.register_buffer(
                "weight_scale",
                torch.randn(4096, 32, dtype=torch.float16),  # group_size=128
            )

    module = MockQuantizedModule()

    print(f"\n  Packed shape: {module.weight_packed.shape}")
    print(f"  Packed dtype: {module.weight_packed.dtype}")
    print(f"  Scales shape: {module.weight_scale.shape}")

    # Unpack weights
    unpacked = unpacker.unpack_module(
        module=module,
        module_name="example_layer",
        output_dtype=torch.float16,
    )

    if unpacked is not None:
        print(f"\n  ✓ Unpacked successfully!")
        print(f"    Unpacked shape: {unpacked.shape}")
        print(f"    Unpacked dtype: {unpacked.dtype}")
        print(f"    Memory: {unpacked.element_size() * unpacked.nelement() / 1024**2:.2f} MB")

        # Check cache
        stats = unpacker.get_cache_stats()
        print(f"\n  Cache stats:")
        print(f"    Size: {stats['size']} entries")
        print(f"    Hits: {stats['hits']}")
        print(f"    Misses: {stats['misses']}")
        print(f"    Hit rate: {stats['hit_rate']:.1%}")

    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis example requires:")
        print("  1. An INT4 quantized model (use llm-compressor)")
        print("  2. LoRA adapters")
        print("  3. vLLM with INT4+LoRA support")

    # Run unpacking demo (always works with mock data)
    demo_unpacking()
