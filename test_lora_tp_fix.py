"""
Test script to verify the fix for LoRA + Tensor Parallelism IndexError.

This test reproduces the issue reported in #36372 where using LoRA with
Tensor Parallelism on Qwen3.5-27B causes an IndexError.

The bug occurs in two locations:
1. slice_lora_b method (fixed by PR #36378)
2. set_lora method (fixed by this PR)

Both locations need bounds checking before accessing lora_a[i] and lora_b[i].
"""

from vllm import LLM, SamplingParams

if __name__ == '__main__':
    print("=" * 60)
    print("Testing LoRA + TP Fix for Issue #36372")
    print("=" * 60)

    try:
        # Configuration that triggers the bug:
        # - Qwen3.5-27B model with vision components
        # - Tensor Parallelism (TP=2)
        # - LoRA enabled
        llm = LLM(
            model="Qwen/Qwen3.5-27B",  # or local path
            tensor_parallel_size=2,
            enable_lora=True,
            max_model_len=512,
            gpu_memory_utilization=0.5,
            trust_remote_code=True
        )

        print("\n✓ Model loaded successfully with TP=2 and LoRA enabled!")

        # Run inference
        prompts = ["Hello, how are you?"]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)
        outputs = llm.generate(prompts, sampling_params)

        print("\n✓ Inference completed successfully!")
        print("\nGenerated output:")
        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")

        print("\n" + "=" * 60)
        print("✅ Test passed! The fix works correctly.")
        print("=" * 60)

    except IndexError as e:
        print(f"\n❌ IndexError occurred: {e}")
        print("The bug is NOT fixed.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
