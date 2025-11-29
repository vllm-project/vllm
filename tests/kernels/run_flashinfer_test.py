#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple standalone script to test FlashInfer integration with vLLM.

This script verifies that FlashInfer is properly integrated and used for
all supported operators when running inference.

Note: VLLM_USE_FLASHINFER is automatically set to 1 by this script.
      AllReduce fusion is automatically enabled for TP >= 2.

Usage:
    # Test with a smaller model (Qwen3 MoE - fits on fewer GPUs)
    python tests/kernels/run_flashinfer_test.py --model qwen

    # Test with Llama-70B (requires 4+ GPUs)
    python tests/kernels/run_flashinfer_test.py --model llama

    # Test with GPT-OSS-120B (OpenAI's open-source model with MXFP4 quantization)
    python tests/kernels/run_flashinfer_test.py --model gpt-oss

    # Test all models
    python tests/kernels/run_flashinfer_test.py --model all

    # Test with FP8 quantization
    python tests/kernels/run_flashinfer_test.py --model qwen --fp8

    # Specify tensor parallel size
    python tests/kernels/run_flashinfer_test.py --model qwen --tp 2
    
    # Disable AllReduce fusion (if needed)
    VLLM_USE_FLASHINFER_ALLREDUCE=0 python tests/kernels/run_flashinfer_test.py --model qwen --tp 2
"""

import argparse
import os
import sys
import time

# This script is for testing FlashInfer - always enable it
os.environ["VLLM_USE_FLASHINFER"] = "1"

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
    
    # Check FlashInfer modules (core features)
    modules_to_check = [
        ("flashinfer.norm", "rmsnorm", True),  # (module, func, required)
        ("flashinfer.activation", "silu_and_mul", True),
        ("flashinfer.sampling", "top_k_top_p_sampling_from_probs", True),
        # Note: allreduce_fusion exists in 0.5.2/0.5.3 but has JIT compilation issues
        ("flashinfer.comm", "trtllm_allreduce_fusion", False),
    ]
    
    for module_name, func_name, required in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[func_name])
            if hasattr(module, func_name):
                note = " (has JIT issues in 0.5.2/0.5.3)" if "allreduce" in func_name else ""
                print(f"  ✓ {module_name}.{func_name}{note}")
            else:
                status = "⚠" if not required else "❌"
                print(f"  {status} {module_name}.{func_name} not found")
        except ImportError as e:
            status = "⚠" if not required else "❌"
            print(f"  {status} {module_name} import failed: {e}")
    
    # Check vLLM environment variables
    import vllm.envs as envs
    
    print("\n" + "=" * 70)
    print("VLLM FLASHINFER ENVIRONMENT VARIABLES")
    print("=" * 70)
    
    env_checks = [
        ("VLLM_USE_FLASHINFER", envs.VLLM_USE_FLASHINFER, "master switch"),
        ("VLLM_ATTENTION_BACKEND", envs.VLLM_ATTENTION_BACKEND, "attention"),
        ("VLLM_USE_FLASHINFER_SAMPLER", envs.VLLM_USE_FLASHINFER_SAMPLER, "sampling"),
        ("VLLM_USE_FLASHINFER_NORM", envs.VLLM_USE_FLASHINFER_NORM, "RMSNorm"),
        ("VLLM_USE_FLASHINFER_ACTIVATION", envs.VLLM_USE_FLASHINFER_ACTIVATION, "activations"),
        ("VLLM_USE_FLASHINFER_ALLREDUCE", envs.VLLM_USE_FLASHINFER_ALLREDUCE, "allreduce"),
        ("VLLM_USE_FLASHINFER_MOE_FP16", envs.VLLM_USE_FLASHINFER_MOE_FP16, "MoE FP16"),
        ("VLLM_USE_FLASHINFER_MOE_FP8", envs.VLLM_USE_FLASHINFER_MOE_FP8, "MoE FP8"),
        ("VLLM_USE_FLASHINFER_MOE_FP4", envs.VLLM_USE_FLASHINFER_MOE_FP4, "MoE FP4"),
        ("VLLM_ALL2ALL_BACKEND", envs.VLLM_ALL2ALL_BACKEND, "all2all"),
    ]
    
    all_set = True
    for name, value, description in env_checks:
        if name == "VLLM_USE_FLASHINFER" and not value:
            print(f"❌ {name} = {value} (MUST be True!)")
            all_set = False
        elif name == "VLLM_ATTENTION_BACKEND":
            expected = "FLASHINFER"
            status = "✓" if value == expected else "⚠"
            print(f"{status} {name} = {value}")
        elif name == "VLLM_USE_FLASHINFER_ALLREDUCE":
            # Allreduce is NOT auto-enabled (has JIT issues in FlashInfer 0.5.2/0.5.3)
            status = "✓" if value else "○"
            note = " (not auto-enabled, has JIT issues in FI 0.5.2/0.5.3)" if not value else ""
            print(f"{status} {name} = {value}{note}")
        else:
            status = "✓" if value else "○"
            print(f"{status} {name} = {value}")
    
    if not envs.VLLM_USE_FLASHINFER:
        print("\n❌ VLLM_USE_FLASHINFER is not set!")
        print("   Run with: VLLM_USE_FLASHINFER=1 python run_flashinfer_test.py ...")
        return False
    
    return True


# Return values for run_inference_test
RESULT_PASS = "PASS"
RESULT_FAIL = "FAIL"
RESULT_SKIPPED = "SKIPPED"


def run_inference_test(
    model_id: str,
    tp_size: int,
    quantization: str | None = None,
    enable_allreduce: bool = False,
):
    """Run inference test with the specified model.
    
    Returns:
        Tuple of (result_status, skip_reason) where result_status is one of
        RESULT_PASS, RESULT_FAIL, or RESULT_SKIPPED. skip_reason is set only
        when RESULT_SKIPPED is returned.
    """
    from vllm import LLM, SamplingParams
    
    print("\n" + "=" * 70)
    print(f"RUNNING INFERENCE TEST")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Tensor Parallel Size: {tp_size}")
    print(f"Quantization: {quantization or 'None'}")
    print(f"AllReduce Fusion: {'ENABLED' if enable_allreduce else 'DISABLED'}")
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
    
    # Add AllReduce fusion if requested
    if enable_allreduce and tp_size >= 2:
        from vllm.config import CompilationConfig, PassConfig
        pass_config = PassConfig(
            enable_fi_allreduce_fusion=True,
            enable_noop=True,
        )
        llm_kwargs["compilation_config"] = CompilationConfig(pass_config=pass_config)
        print("✓ AllReduce fusion compilation config enabled")
    
    try:
        llm = LLM(**llm_kwargs)
    except NotImplementedError as e:
        # Handle FlashInfer FP8 CUDA version incompatibility
        error_msg = str(e)
        if "FP8 block scaling not implemented" in error_msg:
            print(f"⚠ FlashInfer FP8 kernel not available: {error_msg}")
            return RESULT_SKIPPED, "FlashInfer FP8 MoE requires newer CUDA (see FlashInfer package)"
        raise
    except RuntimeError as e:
        # The error may be wrapped in a RuntimeError from worker process
        error_msg = str(e)
        if "FP8 block scaling not implemented" in error_msg:
            print(f"⚠ FlashInfer FP8 kernel not available: {error_msg}")
            return RESULT_SKIPPED, "FlashInfer FP8 MoE requires newer CUDA (see FlashInfer package)"
        print(f"❌ Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        return RESULT_FAIL, None
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        import traceback
        traceback.print_exc()
        return RESULT_FAIL, None
    
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
        return RESULT_FAIL, None
    
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
    
    return RESULT_PASS if all_valid else RESULT_FAIL, None


def main():
    parser = argparse.ArgumentParser(
        description="Test FlashInfer integration with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        choices=["qwen", "llama", "gpt-oss", "all"],
        default="qwen",
        help="Model to test (default: qwen). gpt-oss uses MXFP4 quantization."
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
        help="Use FP8 quantization (uses pre-quantized FP8 model variants)"
    )
    parser.add_argument(
        "--skip-prereq",
        action="store_true",
        help="Skip prerequisite checks"
    )
    parser.add_argument(
        "--enable-allreduce",
        action="store_true",
        help="Force enable AllReduce fusion (enabled by default for TP >= 2)"
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not args.skip_prereq:
        if not check_prerequisites():
            sys.exit(1)
    
    import torch
    gpu_count = torch.cuda.device_count()
    
    # Get CUDA version for compatibility checks
    # Use runtime_version() which is more reliable than torch.version.cuda
    cuda_version = None
    cuda_version_str = "unknown"
    if torch.cuda.is_available():
        try:
            # Get CUDA runtime version (e.g., 12060 for CUDA 12.6.0)
            runtime_version = torch.cuda.runtime_version()
            if runtime_version:
                major = runtime_version // 1000
                minor = (runtime_version % 1000) // 10
                cuda_version = (major, minor)
                cuda_version_str = f"{major}.{minor}"
        except Exception:
            # Fallback to compiled version string
            cuda_str = torch.version.cuda
            if cuda_str:
                parts = cuda_str.split('.')
                cuda_version = (int(parts[0]), int(parts[1]))
                cuda_version_str = cuda_str
    
    print(f"\nDetected CUDA version: {cuda_version_str}")
    
    # Model configurations
    # When --fp8 is passed, use the FP8 variant of the model
    # Note: For FP8 models, the quantization is auto-detected from model config
    # (e.g., RedHatAI models use compressed-tensors, Qwen FP8 uses fp8)
    # We should NOT pass quantization="fp8" for models that have it in their config.
    models = {
        "qwen": {
            "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "model_id_fp8": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "min_tp": 1,
            "recommended_tp": 2,
            # Qwen FP8 needs explicit quantization arg
            "fp8_needs_quant_arg": True,
        },
        "llama": {
            "model_id": "meta-llama/Llama-3.1-70B-Instruct",
            "model_id_fp8": "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8",
            "min_tp": 4,
            "recommended_tp": 8,
            # RedHatAI uses compressed-tensors, auto-detected from config
            "fp8_needs_quant_arg": False,
        },
        # GPT-OSS-120B: OpenAI's open-source reasoning model
        # See: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
        # Uses MXFP4 quantization with different backends per GPU:
        #
        # ATTENTION SINKS:
        # - GPT-OSS uses "attention sinks" for memory efficiency
        # - FlashInfer supports sinks ONLY on Blackwell (SM100) via TRTLLM attention
        # - On Hopper (SM90), must use FlashAttention3 for attention
        #
        # MoE:
        # - Blackwell (SM100): FlashInfer CUTLASS MXFP4 MoE
        # - Hopper (SM90): Marlin/Triton MXFP4 MoE (FI SM90 backend is broken)
        "gpt-oss": {
            "model_id": "openai/gpt-oss-120b",
            "model_id_fp8": None,  # No FP8 variant, uses MXFP4
            "min_tp": 1,
            "recommended_tp": 2,  # TP=2 recommended for best performance
            "fp8_needs_quant_arg": False,
            # Special handling: model uses built-in MXFP4 quantization
            "is_mxfp4": True,
            # Model uses attention sinks:
            # - Hopper: FlashInfer doesn't support sinks, use FlashAttention3
            # - Blackwell: FlashInfer supports sinks via TRTLLM attention
            "skip_flashinfer_attention_hopper_only": True,
            # FlashInfer MXFP4 MoE is ONLY for Blackwell (SM 100).
            # On Hopper (SM 90), vLLM's SM90_FI_MXFP4_BF16 backend is broken - 
            # it tries to use SM100 kernels. Use Marlin/Triton instead.
            "env_vars_blackwell_only": {
                "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16": "1",
            },
            # Empty env_vars for Hopper - uses Marlin/Triton MXFP4 by default
            "env_vars": {},
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
        
        # Enable AllReduce by default if TP >= 2, unless explicitly disabled
        if args.enable_allreduce or (tp_size >= 2 and os.getenv("VLLM_USE_FLASHINFER_ALLREDUCE") != "0"):
            enable_allreduce = True
            os.environ["VLLM_USE_FLASHINFER_ALLREDUCE"] = "1"
            if tp_size >= 2:
                print(f"\n✓ AllReduce Fusion ENABLED for {model_name} (TP={tp_size})")
                if not args.enable_allreduce:
                    print("  (automatically enabled for TP >= 2, set VLLM_USE_FLASHINFER_ALLREDUCE=0 to disable)")
        else:
            enable_allreduce = False
            if tp_size >= 2:
                print(f"\n○ AllReduce Fusion DISABLED for {model_name} (set VLLM_USE_FLASHINFER_ALLREDUCE=1 to enable)")
        
        # Check if we have enough GPUs
        if gpu_count < config["min_tp"]:
            print(f"\n⚠ Skipping {model_name}: requires {config['min_tp']} GPUs, "
                  f"but only {gpu_count} available")
            results[model_name] = RESULT_SKIPPED
            continue
        
        # Select model ID and quantization based on --fp8 flag
        result_name = f"{model_name}-fp8" if args.fp8 else model_name
        
        if args.fp8:
            # Check if model has FP8 variant
            if config.get("model_id_fp8") is None:
                print(f"\n⚠ Skipping {result_name}: Model does not have an FP8 variant")
                if config.get("is_mxfp4"):
                    print(f"   Note: {model_name} uses MXFP4 quantization instead of FP8")
                results[result_name] = RESULT_SKIPPED
                continue
            
            # Skip Qwen FP8 - FlashInfer FP8 MoE requires CUDA 12.7+ but the
            # FlashInfer package is compiled against CUDA 12.6
            if model_name == "qwen":
                print(f"\n⚠ Skipping {result_name}: FlashInfer FP8 MoE kernel requires "
                      "CUDA 12.7+ (FlashInfer package compiled against older CUDA)")
                results[result_name] = RESULT_SKIPPED
                continue
            
            model_id = config["model_id_fp8"]
            
            # Only pass quantization arg if the model needs it
            # (some models auto-detect from config, e.g., compressed-tensors)
            quantization = "fp8" if config.get("fp8_needs_quant_arg", True) else None
        else:
            model_id = config["model_id"]
            quantization = None
        
        # Check GPU architecture for architecture-specific env vars
        device_capability = torch.cuda.get_device_capability(0)
        is_blackwell = device_capability[0] >= 10  # SM 100+
        
        # Set model-specific environment variables
        model_env_vars = config.get("env_vars", {})
        for env_name, env_value in model_env_vars.items():
            os.environ[env_name] = env_value
            print(f"  Setting {env_name}={env_value}")
        
        # Set Blackwell-only environment variables (e.g., FlashInfer MXFP4 MoE)
        if is_blackwell:
            blackwell_env_vars = config.get("env_vars_blackwell_only", {})
            for env_name, env_value in blackwell_env_vars.items():
                os.environ[env_name] = env_value
                print(f"  Setting {env_name}={env_value} (Blackwell only)")
        elif config.get("env_vars_blackwell_only"):
            print(f"  Note: Skipping FlashInfer MXFP4 MoE (requires Blackwell/SM100+)")
            print(f"        Using Marlin/Triton MXFP4 MoE instead (Hopper/SM90)")
            # Explicitly disable FlashInfer MXFP4 MoE on non-Blackwell to use Marlin/Triton
            # Note: VLLM_USE_FLASHINFER_MOE_MXFP4_BF16 is the correct env var name
            os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"] = "0"
            os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "0"
            os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"] = "0"
        
        # Check if FlashInfer attention should be skipped for attention sinks
        # GPT-OSS uses attention sinks:
        # - On Hopper (SM90): FlashInfer doesn't support sinks, use FlashAttention3
        # - On Blackwell (SM100): FlashInfer supports sinks via TRTLLM attention
        original_attn_backend = os.environ.get("VLLM_ATTENTION_BACKEND")
        if config.get("skip_flashinfer_attention_hopper_only", False) and not is_blackwell:
            # Override attention backend on Hopper - FlashInfer doesn't support 
            # attention sinks without TRTLLM (which requires SM100).
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
            print(f"  Note: Using FLASH_ATTN for attention (model uses attention sinks)")
            print(f"        FlashInfer supports sinks only on Blackwell via TRTLLM")
        elif config.get("skip_flashinfer_attention_hopper_only", False) and is_blackwell:
            print(f"  Note: Using FlashInfer attention with TRTLLM (supports sinks on Blackwell)")
        
        # Run test
        result, skip_reason = run_inference_test(model_id, tp_size, quantization, enable_allreduce)
        
        # Restore attention backend for next test
        if original_attn_backend is not None:
            os.environ["VLLM_ATTENTION_BACKEND"] = original_attn_backend
        elif "VLLM_ATTENTION_BACKEND" in os.environ:
            del os.environ["VLLM_ATTENTION_BACKEND"]
        
        if result == RESULT_SKIPPED and skip_reason:
            print(f"\n⚠ {result_name}: SKIPPED - {skip_reason}")
        
        results[result_name] = result
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for model_name, result in results.items():
        if result == RESULT_PASS:
            print(f"✓ {model_name}: {result}")
        elif result == RESULT_SKIPPED:
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

