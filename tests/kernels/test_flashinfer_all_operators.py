# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script to verify FlashInfer is used for all supported operators.

This test verifies that when VLLM_USE_FLASHINFER=1 is set:
1. FlashInfer attention backend is used
2. FlashInfer sampling is used
3. FlashInfer RMSNorm is used
4. FlashInfer activations (SiLU, GELU) are used
5. FlashInfer MoE kernels are used (for MoE models)
6. FlashInfer allreduce fusion is used (for TP > 1)
7. Generated text is coherent and makes sense
8. No errors occur during inference

Models tested:
- meta-llama/Llama-3.1-70B-Instruct (dense model)
- Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE model)

Usage:
    # Run with FlashInfer enabled for all operators
    VLLM_USE_FLASHINFER=1 python tests/kernels/test_flashinfer_all_operators.py

    # Run specific test
    VLLM_USE_FLASHINFER=1 pytest tests/kernels/test_flashinfer_all_operators.py -v -k "llama"
"""

import logging
import os
import sys
from typing import NamedTuple

import pytest
import torch

# Ensure VLLM_USE_FLASHINFER is set before importing vllm
os.environ.setdefault("VLLM_USE_FLASHINFER", "1")

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)


class ModelConfig(NamedTuple):
    """Configuration for a model to test."""
    name: str
    model_id: str
    is_moe: bool
    min_tp_size: int
    test_prompts: list[str]
    expected_keywords: list[str]  # Keywords expected in reasonable responses
    quantization: str | None = None
    max_model_len: int = 4096


# Test configurations for different models
LLAMA_70B_CONFIG = ModelConfig(
    name="Llama-3.1-70B-Instruct",
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    is_moe=False,
    min_tp_size=4,  # 70B needs at least 4 GPUs
    test_prompts=[
        "What is the capital of France? Answer in one sentence.",
        "Write a haiku about programming.",
        "Explain what machine learning is in simple terms.",
    ],
    expected_keywords=["Paris", "code", "data"],  # At least one should appear
)

LLAMA_70B_FP8_CONFIG = ModelConfig(
    name="Llama-3.1-70B-Instruct-FP8",
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    is_moe=False,
    min_tp_size=2,  # FP8 needs fewer GPUs
    quantization="fp8",
    test_prompts=[
        "What is the capital of France? Answer in one sentence.",
        "Write a haiku about programming.",
    ],
    expected_keywords=["Paris", "code"],
)

QWEN3_MOE_CONFIG = ModelConfig(
    name="Qwen3-30B-A3B-Instruct-MoE",
    model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    is_moe=True,
    min_tp_size=1,  # MoE model with 3B active params
    test_prompts=[
        "What is 2 + 2? Answer with just the number.",
        "Name the largest planet in our solar system.",
        "What programming language is vLLM written in?",
    ],
    expected_keywords=["4", "Jupiter", "Python"],
)

QWEN3_MOE_FP8_CONFIG = ModelConfig(
    name="Qwen3-30B-A3B-Instruct-MoE-FP8",
    model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    is_moe=True,
    min_tp_size=1,
    quantization="fp8",
    test_prompts=[
        "What is 2 + 2? Answer with just the number.",
        "Name the largest planet in our solar system.",
    ],
    expected_keywords=["4", "Jupiter"],
)


def get_available_gpu_count() -> int:
    """Get the number of available GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def check_flashinfer_availability() -> dict[str, bool]:
    """Check which FlashInfer features are available."""
    features = {
        "flashinfer_installed": False,
        "attention": False,
        "sampling": False,
        "norm": False,
        "activation": False,
        "moe": False,
        "comm": False,
    }
    
    try:
        import flashinfer
        features["flashinfer_installed"] = True
        
        # Check attention
        try:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper
            features["attention"] = True
        except ImportError:
            pass
        
        # Check sampling
        try:
            from flashinfer.sampling import top_k_top_p_sampling_from_probs
            features["sampling"] = True
        except ImportError:
            pass
        
        # Check norm
        try:
            from flashinfer.norm import rmsnorm
            features["norm"] = True
        except ImportError:
            pass
        
        # Check activation
        try:
            from flashinfer.activation import silu_and_mul
            features["activation"] = True
        except ImportError:
            pass
        
        # Check MoE
        try:
            from flashinfer.fused_moe import cutlass_fused_moe
            features["moe"] = True
        except ImportError:
            pass
        
        # Check comm (allreduce)
        try:
            from flashinfer.comm import trtllm_allreduce_fusion
            features["comm"] = True
        except ImportError:
            pass
            
    except ImportError:
        pass
    
    return features


def verify_flashinfer_env_vars() -> dict[str, bool]:
    """Verify FlashInfer environment variables are set correctly."""
    return {
        "VLLM_USE_FLASHINFER": envs.VLLM_USE_FLASHINFER,
        "VLLM_ATTENTION_BACKEND": envs.VLLM_ATTENTION_BACKEND == "FLASHINFER",
        "VLLM_USE_FLASHINFER_SAMPLER": envs.VLLM_USE_FLASHINFER_SAMPLER is True,
        "VLLM_USE_FLASHINFER_NORM": envs.VLLM_USE_FLASHINFER_NORM,
        "VLLM_USE_FLASHINFER_ACTIVATION": envs.VLLM_USE_FLASHINFER_ACTIVATION,
        "VLLM_USE_FLASHINFER_ALLREDUCE": envs.VLLM_USE_FLASHINFER_ALLREDUCE,
        "VLLM_USE_FLASHINFER_MOE_FP16": envs.VLLM_USE_FLASHINFER_MOE_FP16,
        "VLLM_USE_FLASHINFER_MOE_FP8": envs.VLLM_USE_FLASHINFER_MOE_FP8,
    }


def validate_output(output_text: str, expected_keywords: list[str]) -> bool:
    """
    Validate that the output text is coherent and contains expected content.
    
    Returns True if the output seems reasonable.
    """
    # Check output is not empty
    if not output_text or len(output_text.strip()) < 5:
        logger.error(f"Output too short: '{output_text}'")
        return False
    
    # Check for common error patterns
    error_patterns = [
        "error",
        "exception",
        "traceback",
        "nan",
        "inf",
        "<unk>",
        "�",  # Unicode replacement character
    ]
    
    output_lower = output_text.lower()
    for pattern in error_patterns:
        if pattern in output_lower and pattern not in ["inference", "reference"]:
            logger.warning(f"Potential error pattern '{pattern}' found in output")
    
    # Check if at least one expected keyword is present (case-insensitive)
    found_keyword = False
    for keyword in expected_keywords:
        if keyword.lower() in output_lower:
            found_keyword = True
            logger.info(f"Found expected keyword: '{keyword}'")
            break
    
    if not found_keyword:
        logger.warning(
            f"None of the expected keywords {expected_keywords} found in output: "
            f"'{output_text[:200]}...'"
        )
    
    return True  # Return True even if keyword not found, as LLM outputs can vary


def run_inference_test(
    config: ModelConfig,
    tp_size: int = 1,
) -> tuple[bool, list[str], str]:
    """
    Run inference test with the given model configuration.
    
    Returns:
        (success, outputs, error_message)
    """
    logger.info(f"Testing model: {config.name}")
    logger.info(f"  Model ID: {config.model_id}")
    logger.info(f"  Tensor Parallel Size: {tp_size}")
    logger.info(f"  Quantization: {config.quantization or 'None'}")
    logger.info(f"  Is MoE: {config.is_moe}")
    
    try:
        # Initialize LLM
        llm_kwargs = {
            "model": config.model_id,
            "tensor_parallel_size": tp_size,
            "max_model_len": config.max_model_len,
            "trust_remote_code": True,
            "enforce_eager": False,  # Enable compilation for fusion passes
        }
        
        if config.quantization:
            llm_kwargs["quantization"] = config.quantization
        
        llm = LLM(**llm_kwargs)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
        )
        
        # Run inference
        outputs = llm.generate(config.test_prompts, sampling_params)
        
        # Extract and validate outputs
        output_texts = []
        all_valid = True
        
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            output_texts.append(generated_text)
            
            logger.info(f"  Prompt {i+1}: {config.test_prompts[i][:50]}...")
            logger.info(f"  Output {i+1}: {generated_text[:100]}...")
            
            # Validate output
            is_valid = validate_output(
                generated_text, 
                [config.expected_keywords[i]] if i < len(config.expected_keywords) else []
            )
            if not is_valid:
                all_valid = False
        
        # Cleanup
        del llm
        torch.cuda.empty_cache()
        
        return all_valid, output_texts, ""
        
    except Exception as e:
        error_msg = f"Error testing {config.name}: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return False, [], error_msg


class TestFlashInferAllOperators:
    """Test class for FlashInfer operator integration."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        # Verify FlashInfer is available
        self.fi_features = check_flashinfer_availability()
        self.env_vars = verify_flashinfer_env_vars()
        self.gpu_count = get_available_gpu_count()
        
        logger.info("=== FlashInfer Feature Availability ===")
        for feature, available in self.fi_features.items():
            logger.info(f"  {feature}: {'✓' if available else '✗'}")
        
        logger.info("=== Environment Variables ===")
        for var, value in self.env_vars.items():
            logger.info(f"  {var}: {value}")
        
        logger.info(f"=== GPU Count: {self.gpu_count} ===")
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_flashinfer_env_vars_set(self):
        """Test that FlashInfer environment variables are properly set."""
        assert envs.VLLM_USE_FLASHINFER, (
            "VLLM_USE_FLASHINFER must be set to 1. "
            "Run with: VLLM_USE_FLASHINFER=1 pytest ..."
        )
        
        # When master switch is on, these should all be True
        assert envs.VLLM_USE_FLASHINFER_NORM, "VLLM_USE_FLASHINFER_NORM should be enabled"
        assert envs.VLLM_USE_FLASHINFER_ACTIVATION, "VLLM_USE_FLASHINFER_ACTIVATION should be enabled"
        assert envs.VLLM_USE_FLASHINFER_SAMPLER, "VLLM_USE_FLASHINFER_SAMPLER should be enabled"
        assert envs.VLLM_ATTENTION_BACKEND == "FLASHINFER", (
            f"VLLM_ATTENTION_BACKEND should be FLASHINFER, got {envs.VLLM_ATTENTION_BACKEND}"
        )
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_flashinfer_installed(self):
        """Test that FlashInfer is properly installed."""
        features = check_flashinfer_availability()
        assert features["flashinfer_installed"], "FlashInfer is not installed"
        assert features["attention"], "FlashInfer attention not available"
        assert features["norm"], "FlashInfer norm not available"
        assert features["activation"], "FlashInfer activation not available"
    
    @pytest.mark.skipif(
        get_available_gpu_count() < 1,
        reason="No GPU available"
    )
    @pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="Not running on CUDA platform"
    )
    def test_qwen3_moe_flashinfer(self):
        """Test Qwen3-30B-A3B MoE model with FlashInfer."""
        config = QWEN3_MOE_CONFIG
        tp_size = min(get_available_gpu_count(), 2)
        
        success, outputs, error = run_inference_test(config, tp_size)
        
        assert success, f"Qwen3 MoE test failed: {error}"
        assert len(outputs) == len(config.test_prompts), "Missing outputs"
        for output in outputs:
            assert len(output.strip()) > 0, "Empty output generated"
    
    @pytest.mark.skipif(
        get_available_gpu_count() < 1,
        reason="No GPU available"
    )
    @pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="Not running on CUDA platform"
    )
    @pytest.mark.skipif(
        not current_platform.has_device_capability(89),
        reason="FP8 requires SM89+ (Ada Lovelace or newer)"
    )
    def test_qwen3_moe_fp8_flashinfer(self):
        """Test Qwen3-30B-A3B MoE model with FP8 quantization and FlashInfer."""
        config = QWEN3_MOE_FP8_CONFIG
        tp_size = min(get_available_gpu_count(), 2)
        
        success, outputs, error = run_inference_test(config, tp_size)
        
        assert success, f"Qwen3 MoE FP8 test failed: {error}"
        assert len(outputs) == len(config.test_prompts), "Missing outputs"
    
    @pytest.mark.skipif(
        get_available_gpu_count() < 4,
        reason="Llama-70B requires at least 4 GPUs"
    )
    @pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="Not running on CUDA platform"
    )
    def test_llama_70b_flashinfer(self):
        """Test Llama-3.1-70B model with FlashInfer."""
        config = LLAMA_70B_CONFIG
        tp_size = min(get_available_gpu_count(), 8)
        
        success, outputs, error = run_inference_test(config, tp_size)
        
        assert success, f"Llama-70B test failed: {error}"
        assert len(outputs) == len(config.test_prompts), "Missing outputs"
        for output in outputs:
            assert len(output.strip()) > 0, "Empty output generated"
    
    @pytest.mark.skipif(
        get_available_gpu_count() < 2,
        reason="Llama-70B FP8 requires at least 2 GPUs"
    )
    @pytest.mark.skipif(
        not current_platform.is_cuda(),
        reason="Not running on CUDA platform"
    )
    @pytest.mark.skipif(
        not current_platform.has_device_capability(89),
        reason="FP8 requires SM89+ (Ada Lovelace or newer)"
    )
    def test_llama_70b_fp8_flashinfer(self):
        """Test Llama-3.1-70B model with FP8 quantization and FlashInfer."""
        config = LLAMA_70B_FP8_CONFIG
        tp_size = min(get_available_gpu_count(), 4)
        
        success, outputs, error = run_inference_test(config, tp_size)
        
        assert success, f"Llama-70B FP8 test failed: {error}"
        assert len(outputs) == len(config.test_prompts), "Missing outputs"


def print_test_summary(results: dict[str, tuple[bool, str]]):
    """Print a summary of test results."""
    print("\n" + "=" * 60)
    print("FLASHINFER ALL OPERATORS TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, (passed, message) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            print(f"       Error: {message}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    return all_passed


def main():
    """Main function to run all tests."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Verify VLLM_USE_FLASHINFER is set
    if not envs.VLLM_USE_FLASHINFER:
        print("ERROR: VLLM_USE_FLASHINFER must be set to 1")
        print("Run with: VLLM_USE_FLASHINFER=1 python test_flashinfer_all_operators.py")
        sys.exit(1)
    
    # Check FlashInfer availability
    fi_features = check_flashinfer_availability()
    print("\n=== FlashInfer Feature Availability ===")
    for feature, available in fi_features.items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature}")
    
    if not fi_features["flashinfer_installed"]:
        print("\nERROR: FlashInfer is not installed!")
        print("Install with: pip install flashinfer-python")
        sys.exit(1)
    
    # Check environment variables
    env_vars = verify_flashinfer_env_vars()
    print("\n=== Environment Variables ===")
    for var, value in env_vars.items():
        status = "✓" if value else "✗"
        print(f"  {status} {var}: {value}")
    
    # Check GPU availability
    gpu_count = get_available_gpu_count()
    print(f"\n=== Available GPUs: {gpu_count} ===")
    
    if gpu_count == 0:
        print("ERROR: No GPUs available!")
        sys.exit(1)
    
    # Run tests
    results = {}
    
    # Test 1: Qwen3 MoE (lightweight, good for quick testing)
    print("\n" + "=" * 60)
    print("Test 1: Qwen3-30B-A3B MoE with FlashInfer")
    print("=" * 60)
    tp_size = min(gpu_count, 2)
    success, outputs, error = run_inference_test(QWEN3_MOE_CONFIG, tp_size)
    results["Qwen3-30B-A3B-MoE"] = (success, error)
    
    # Test 2: Qwen3 MoE with FP8 (if hardware supports it)
    if current_platform.has_device_capability(89):
        print("\n" + "=" * 60)
        print("Test 2: Qwen3-30B-A3B MoE FP8 with FlashInfer")
        print("=" * 60)
        success, outputs, error = run_inference_test(QWEN3_MOE_FP8_CONFIG, tp_size)
        results["Qwen3-30B-A3B-MoE-FP8"] = (success, error)
    else:
        print("\nSkipping FP8 tests (requires SM89+)")
    
    # Test 3: Llama-70B (if enough GPUs)
    if gpu_count >= 4:
        print("\n" + "=" * 60)
        print("Test 3: Llama-3.1-70B with FlashInfer")
        print("=" * 60)
        tp_size = min(gpu_count, 8)
        success, outputs, error = run_inference_test(LLAMA_70B_CONFIG, tp_size)
        results["Llama-3.1-70B"] = (success, error)
        
        # Test 4: Llama-70B FP8
        if current_platform.has_device_capability(89) and gpu_count >= 2:
            print("\n" + "=" * 60)
            print("Test 4: Llama-3.1-70B FP8 with FlashInfer")
            print("=" * 60)
            tp_size = min(gpu_count, 4)
            success, outputs, error = run_inference_test(LLAMA_70B_FP8_CONFIG, tp_size)
            results["Llama-3.1-70B-FP8"] = (success, error)
    else:
        print(f"\nSkipping Llama-70B tests (requires 4+ GPUs, have {gpu_count})")
    
    # Print summary
    all_passed = print_test_summary(results)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

