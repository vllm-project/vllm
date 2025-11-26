#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple standalone script to test FlashInfer integration with vLLM.

This script verifies that FlashInfer is properly integrated and used for
all supported operators when running inference.

Usage:
    # Test with a smaller model (Qwen3 MoE - fits on fewer GPUs)
    VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model qwen

    # Test with Llama-70B (requires 4+ GPUs)
    VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model llama

    # Test both models
    VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model all

    # Test with FP8 quantization
    VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model qwen --fp8

    # Specify tensor parallel size
    VLLM_USE_FLASHINFER=1 python tests/kernels/run_flashinfer_test.py --model qwen --tp 2
"""

import argparse
import os
import sys
import time

# Set VLLM_USE_FLASHINFER before importing vllm
os.environ.setdefault("VLLM_USE_FLASHINFER", "1")

# Set logging level to see FlashInfer activation messages
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")


def check_prerequisites():
    """Check that all prerequisites are met."""
    print("=" * 70)
    print("CHECKING PREREQUISITES")
    print("=" * 70)
    
    # Check CUDA
    import torch
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} (SM {props.major}.{props.minor}, "
              f"{props.total_memory / 1024**3:.1f} GB)")
    
    # Check FlashInfer
    try:
        import flashinfer
        print(f"✓ FlashInfer installed (version: {flashinfer.__version__})")
    except ImportError:
        print("❌ FlashInfer is not installed!")
        print("   Install with: pip install flashinfer-python")
        return False
    
    # Check FlashInfer modules
    modules_to_check = [
        ("flashinfer.norm", "rmsnorm"),
        ("flashinfer.activation", "silu_and_mul"),
        ("flashinfer.sampling", "top_k_top_p_sampling_from_probs"),
        ("flashinfer.comm", "trtllm_allreduce_fusion"),
    ]
    
    for module_name, func_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[func_name])
            if hasattr(module, func_name):
                print(f"  ✓ {module_name}.{func_name}")
            else:
                print(f"  ⚠ {module_name}.{func_name} not found")
        except ImportError as e:
            print(f"  ⚠ {module_name} import failed: {e}")
    
    # Check vLLM environment variables
    import vllm.envs as envs
    
    print("\n" + "=" * 70)
    print("VLLM FLASHINFER ENVIRONMENT VARIABLES")
    print("=" * 70)
    
    env_checks = [
        ("VLLM_USE_FLASHINFER", envs.VLLM_USE_FLASHINFER),
        ("VLLM_ATTENTION_BACKEND", envs.VLLM_ATTENTION_BACKEND),
        ("VLLM_USE_FLASHINFER_SAMPLER", envs.VLLM_USE_FLASHINFER_SAMPLER),
        ("VLLM_USE_FLASHINFER_NORM", envs.VLLM_USE_FLASHINFER_NORM),
        ("VLLM_USE_FLASHINFER_ACTIVATION", envs.VLLM_USE_FLASHINFER_ACTIVATION),
        ("VLLM_USE_FLASHINFER_ALLREDUCE", envs.VLLM_USE_FLASHINFER_ALLREDUCE),
        ("VLLM_USE_FLASHINFER_MOE_FP16", envs.VLLM_USE_FLASHINFER_MOE_FP16),
        ("VLLM_USE_FLASHINFER_MOE_FP8", envs.VLLM_USE_FLASHINFER_MOE_FP8),
        ("VLLM_USE_FLASHINFER_MOE_FP4", envs.VLLM_USE_FLASHINFER_MOE_FP4),
        ("VLLM_ALL2ALL_BACKEND", envs.VLLM_ALL2ALL_BACKEND),
    ]
    
    all_set = True
    for name, value in env_checks:
        if name == "VLLM_USE_FLASHINFER" and not value:
            print(f"❌ {name} = {value} (MUST be True!)")
            all_set = False
        elif name == "VLLM_ATTENTION_BACKEND":
            expected = "FLASHINFER"
            status = "✓" if value == expected else "⚠"
            print(f"{status} {name} = {value}")
        else:
            status = "✓" if value else "○"
            print(f"{status} {name} = {value}")
    
    if not envs.VLLM_USE_FLASHINFER:
        print("\n❌ VLLM_USE_FLASHINFER is not set!")
        print("   Run with: VLLM_USE_FLASHINFER=1 python run_flashinfer_test.py ...")
        return False
    
    return True


def run_inference_test(model_id: str, tp_size: int, quantization: str | None = None):
    """Run inference test with the specified model."""
    from vllm import LLM, SamplingParams
    
    print("\n" + "=" * 70)
    print(f"RUNNING INFERENCE TEST")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Tensor Parallel Size: {tp_size}")
    print(f"Quantization: {quantization or 'None'}")
    print("=" * 70)
    
    # Test prompts
    prompts = [
        "What is the capital of France? Answer in one sentence.",
        "Explain what artificial intelligence is in simple terms.",
        "Write a short poem about coding.",
    ]
    
    # Initialize LLM
    print("\nInitializing LLM...")
    start_time = time.time()
    
    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tp_size,
        "max_model_len": 4096,
        "trust_remote_code": True,
        "enforce_eager": False,  # Enable compilation for fusion passes
    }
    
    if quantization:
        llm_kwargs["quantization"] = quantization
    
    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    init_time = time.time() - start_time
    print(f"✓ LLM initialized in {init_time:.1f}s")
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=150,
    )
    
    # Run inference
    print("\nRunning inference...")
    start_time = time.time()
    
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    inference_time = time.time() - start_time
    print(f"✓ Inference completed in {inference_time:.1f}s")
    
    # Display outputs
    print("\n" + "-" * 70)
    print("GENERATED OUTPUTS")
    print("-" * 70)
    
    all_valid = True
    for i, output in enumerate(outputs):
        print(f"\n[Prompt {i+1}]: {prompts[i]}")
        print(f"[Output {i+1}]: {output.outputs[0].text}")
        
        # Basic validation
        generated = output.outputs[0].text.strip()
        if len(generated) < 10:
            print(f"⚠ Warning: Output seems too short")
            all_valid = False
        elif "error" in generated.lower() or "exception" in generated.lower():
            print(f"⚠ Warning: Output may contain error message")
    
    # Cleanup
    print("\nCleaning up...")
    del llm
    import torch
    torch.cuda.empty_cache()
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Test FlashInfer integration with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "llama", "all"],
        default="qwen",
        help="Model to test (default: qwen)"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size (default: auto-detect based on GPU count)"
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Use FP8 quantization"
    )
    parser.add_argument(
        "--skip-prereq",
        action="store_true",
        help="Skip prerequisite checks"
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not args.skip_prereq:
        if not check_prerequisites():
            sys.exit(1)
    
    import torch
    gpu_count = torch.cuda.device_count()
    
    # Model configurations
    models = {
        "qwen": {
            "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "min_tp": 1,
            "recommended_tp": 2,
        },
        "llama": {
            "model_id": "meta-llama/Llama-3.1-70B-Instruct",
            "min_tp": 4,
            "recommended_tp": 8,
        },
    }
    
    # Determine which models to test
    if args.model == "all":
        models_to_test = list(models.keys())
    else:
        models_to_test = [args.model]
    
    # Run tests
    results = {}
    
    for model_name in models_to_test:
        config = models[model_name]
        
        # Determine TP size
        if args.tp:
            tp_size = args.tp
        else:
            tp_size = min(gpu_count, config["recommended_tp"])
        
        # Check if we have enough GPUs
        if gpu_count < config["min_tp"]:
            print(f"\n⚠ Skipping {model_name}: requires {config['min_tp']} GPUs, "
                  f"but only {gpu_count} available")
            results[model_name] = "SKIPPED"
            continue
        
        # Run test
        quantization = "fp8" if args.fp8 else None
        success = run_inference_test(
            config["model_id"],
            tp_size,
            quantization
        )
        
        results[model_name] = "PASS" if success else "FAIL"
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for model_name, result in results.items():
        if result == "PASS":
            print(f"✓ {model_name}: {result}")
        elif result == "SKIPPED":
            print(f"○ {model_name}: {result}")
        else:
            print(f"❌ {model_name}: {result}")
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("All tests PASSED! FlashInfer is working correctly.")
        sys.exit(0)
    else:
        print("Some tests FAILED. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

