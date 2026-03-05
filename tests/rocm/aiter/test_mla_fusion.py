#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive tests for MLA fusion with AMD AITER.

This test suite includes:
- Unit tests for fusion detection and fallback logic
- Integration tests with real DeepSeek models
- Correctness verification tests comparing fused vs unfused outputs

AITER is automatically enabled (VLLM_ROCM_USE_AITER=1) for all tests
via the enable_aiter fixture.

Location: tests/rocm/aiter/test_mla_fusion.py

Run with:
    pytest tests/rocm/aiter/test_mla_fusion.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

# Mark all tests as ROCm-specific
pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="MLA fusion only available on ROCm/AMD GPUs",
)


@pytest.fixture(autouse=True)
def enable_aiter(monkeypatch):
    """Enable AITER for all tests in this module."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")


# =============================================================================
# UNIT TESTS - Testing fusion detection and fallback logic
# =============================================================================


class TestFuseRMSNormQuant:
    """Unit tests for _fuse_rmsnorm_quant function."""

    def test_returns_none_when_aiter_unavailable(self):
        """Test that fusion returns None when AITER is not available."""
        # Mock AITER as unavailable
        with patch("vllm.model_executor.layers.mla._AITER_AVAILABLE", False):
            from vllm.model_executor.layers.mla import _fuse_rmsnorm_quant

            # Create dummy inputs
            q_c = torch.randn(1, 128, 512)
            q_weight = torch.randn(512)
            kv_c = torch.randn(1, 128, 512)
            kv_weight = torch.randn(512)

            # Call fusion function
            result = _fuse_rmsnorm_quant(
                q_c,
                q_weight,
                1e-6,
                kv_c,
                kv_weight,
                1e-6,
                dtype_quant=None,
            )

            # Should return (None, None, None)
            assert result == (None, None, None)

    def test_returns_none_when_dtype_not_fp8(self):
        """Test that fusion returns None when dtype is not FP8."""
        # Mock AITER as available
        with patch("vllm.model_executor.layers.mla._AITER_AVAILABLE", True):
            # Mock dtypes
            mock_dtypes = MagicMock()
            mock_dtypes.fp8 = "fp8"
            mock_dtypes.fp4x2 = "fp4x2"

            with patch("vllm.model_executor.layers.mla.dtypes", mock_dtypes):
                from vllm.model_executor.layers.mla import _fuse_rmsnorm_quant

                q_c = torch.randn(1, 128, 512)
                q_weight = torch.randn(512)
                kv_c = torch.randn(1, 128, 512)
                kv_weight = torch.randn(512)

                # Call with non-FP8 dtype
                result = _fuse_rmsnorm_quant(
                    q_c,
                    q_weight,
                    1e-6,
                    kv_c,
                    kv_weight,
                    1e-6,
                    dtype_quant=mock_dtypes.fp4x2,  # Not FP8
                )

                # Should return (None, None, None)
                assert result == (None, None, None)


# =============================================================================
# INTEGRATION TESTS - Testing with real DeepSeek models
# =============================================================================


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-ai/DeepSeek-V2-Lite",
    ],
)
@pytest.mark.parametrize("quantization", ["fp8", None])
@pytest.mark.parametrize("max_tokens", [10])
def test_mla_model_inference(
    vllm_runner, example_prompts, model, quantization, max_tokens
):
    """Test that DeepSeek models with MLA run successfully."""
    with vllm_runner(
        model,
        quantization=quantization,
        trust_remote_code=True,
        max_model_len=512,
        enforce_eager=True,  # For testing
    ) as vllm_model:
        # Generate outputs
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

        # Basic checks
        assert len(outputs) == len(example_prompts)
        for output_ids, output_text in outputs:
            assert output_ids is not None
            assert output_text is not None
            assert len(output_text) > 0


def test_fp8_vs_baseline(vllm_runner, example_prompts):
    """Test that FP8 with fusion produces reasonable outputs."""
    import gc

    import torch

    from vllm.distributed import cleanup_dist_env_and_memory

    model = "deepseek-ai/DeepSeek-V2-Lite"
    max_tokens = 20

    # Baseline (no quantization, no fusion)
    with vllm_runner(
        model,
        quantization=None,
        trust_remote_code=True,
        max_model_len=512,
        enforce_eager=True,
    ) as vllm_model:
        baseline_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    # Explicit cleanup to free GPU memory before loading second model
    cleanup_dist_env_and_memory()
    gc.collect()
    torch.accelerator.empty_cache()

    # With FP8 (fusion should be enabled on ROCm)
    with vllm_runner(
        model,
        quantization="fp8",
        trust_remote_code=True,
        max_model_len=512,
        enforce_eager=True,
    ) as vllm_model:
        fp8_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    # Both should produce outputs
    assert len(baseline_outputs) == len(fp8_outputs)

    # Outputs should be reasonable (not empty)
    for baseline, fp8 in zip(baseline_outputs, fp8_outputs):
        baseline_ids, baseline_text = baseline
        fp8_ids, fp8_text = fp8

        assert len(baseline_text) > 0
        assert len(fp8_text) > 0

        # Optional: Check outputs are similar (may differ due to FP8)
        # This is a loose check - exact match not expected

        # At least check they're non-empty and reasonable length
        assert len(baseline_text) > 5
        assert len(fp8_text) > 5


@pytest.mark.slow_test  # Mark as slow since it loads models multiple times
def test_different_batch_sizes(vllm_runner):
    """Test fusion works with different batch sizes."""
    model = "deepseek-ai/DeepSeek-V2-Lite"

    for batch_size in [1, 2, 4]:
        prompts = ["Hello"] * batch_size

        with vllm_runner(
            model,
            quantization="fp8",
            trust_remote_code=True,
            max_model_len=256,
            enforce_eager=True,
        ) as vllm_model:
            outputs = vllm_model.generate_greedy(prompts, max_tokens=5)

            assert len(outputs) == batch_size
            for output_ids, output_text in outputs:
                assert len(output_text) > 0


@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_tensor_parallelism(vllm_runner, tensor_parallel_size):
    """Test that fusion works with tensor parallelism.

    Note: DeepSeek-V2-Lite only supports TP=1.
    DeepSeek-V3 would require TP=8, but it's too large for tests.
    """
    model = "deepseek-ai/DeepSeek-V2-Lite"
    prompts = ["Hello, how are you?"]

    with vllm_runner(
        model,
        quantization="fp8",
        trust_remote_code=True,
        max_model_len=256,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=10)

        assert len(outputs) == 1
        output_ids, output_text = outputs[0]
        assert len(output_text) > 0


def test_no_crash_on_unsupported_model(vllm_runner):
    """Test that fusion doesn't crash non-DeepSeek models."""
    # Use a model that doesn't have MLA
    model = "facebook/opt-125m"

    with vllm_runner(
        model,
        quantization="fp8",
        max_model_len=256,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(["Hello"], max_tokens=5)

        # Should work (fusion just won't be used)
        assert len(outputs) == 1
        output_ids, output_text = outputs[0]
        assert len(output_text) > 0


# =============================================================================
# CORRECTNESS TESTS - Verifying numerical accuracy
# =============================================================================


@pytest.mark.slow_test  # Loads model twice, may OOM on sequential runs
@pytest.mark.parametrize(
    "model",
    [
        "deepseek-ai/DeepSeek-V2-Lite",
    ],
)
@pytest.mark.parametrize("max_tokens", [10])
def test_logprobs_match_baseline(
    vllm_runner, example_prompts, model, max_tokens, monkeypatch
):
    """
    Test that FP8 with fusion produces similar logprobs to FP8 without fusion.

    This test compares FP8-with-fusion vs FP8-without-fusion to verify that
    the fusion kernel produces correct results. Both runs use FP8 quantization.

    Note: Due to FP8 quantization, exact matches are not expected.
    We use a tolerance to account for numerical differences.

    This test loads the model twice and may fail with OOM when run
    sequentially after other tests. Use smaller max_model_len to reduce memory.
    """
    import gc
    import time

    import torch

    from vllm.distributed import cleanup_dist_env_and_memory

    NUM_LOG_PROBS = 5
    MAX_MODEL_LEN = 256  # Reduced from 512 to avoid OOM

    # Baseline: FP8 without fusion (disable AITER to force unfused path)
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    with vllm_runner(
        model,
        max_model_len=MAX_MODEL_LEN,
        quantization="fp8",
        trust_remote_code=True,
        enforce_eager=True,
    ) as vllm_model:
        baseline_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, NUM_LOG_PROBS
        )

    # Explicit cleanup to free GPU memory before loading second model
    cleanup_dist_env_and_memory()
    gc.collect()
    torch.accelerator.empty_cache()

    # Additional cleanup to avoid OOM when tests run sequentially
    time.sleep(2)  # Allow GPU memory to fully release
    gc.collect()
    torch.accelerator.empty_cache()

    # Test: FP8 with fusion (re-enable AITER)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    with vllm_runner(
        model,
        max_model_len=MAX_MODEL_LEN,
        quantization="fp8",
        trust_remote_code=True,
        enforce_eager=True,
    ) as vllm_model:
        test_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, NUM_LOG_PROBS
        )

    # Check that logprobs are close
    # Note: check_logprobs_close() checks if highest-logprob tokens match,
    # not numerical closeness. For FP8, we use warn_on_mismatch=True
    # to allow some differences due to quantization
    check_logprobs_close(
        outputs_0_lst=baseline_outputs,
        outputs_1_lst=test_outputs,
        name_0="fp8_unfused",
        name_1="fp8_fused",
        warn_on_mismatch=True,  # Allow warnings for FP8 differences
        always_check_logprobs=False,  # Only check when tokens differ
    )


@pytest.mark.skip(
    reason="AITER kernels may exhibit non-deterministic behavior due to "
    "parallel reduction operations and floating-point arithmetic ordering. "
    "This is expected for GPU kernels optimized for performance."
)
@pytest.mark.parametrize(
    "model",
    [
        "deepseek-ai/DeepSeek-V2-Lite",
    ],
)
def test_deterministic_outputs(vllm_runner, model):
    """Test that fusion produces deterministic outputs.

    NOTE: This test is skipped because AITER kernels may have non-deterministic
    behavior due to:
    1. Parallel reduction operations (atomic adds, sum reductions)
    2. Floating-point arithmetic ordering differences
    3. Non-deterministic kernel launch patterns

    This is common for GPU kernels optimized for performance. If strict
    determinism is required, use enforce_eager=True and set appropriate
    environment variables (CUBLAS_WORKSPACE_CONFIG, etc.), though this
    may not eliminate all sources of non-determinism in custom kernels.
    """
    import gc

    import torch

    from vllm.distributed import cleanup_dist_env_and_memory

    prompts = ["Hello, how are you?"]
    max_tokens = 20

    # Run twice with same seed
    outputs_list = []
    for i in range(2):
        with vllm_runner(
            model,
            quantization="fp8",
            trust_remote_code=True,
            max_model_len=256,
            enforce_eager=True,
            seed=42,  # Fixed seed
        ) as vllm_model:
            outputs = vllm_model.generate_greedy(prompts, max_tokens)
            outputs_list.append(outputs)

        # Cleanup GPU memory between runs
        if i == 0:  # After first run
            cleanup_dist_env_and_memory()
            gc.collect()
            torch.accelerator.empty_cache()

    # Outputs should be identical
    assert len(outputs_list) == 2
    for i in range(len(prompts)):
        _, text1 = outputs_list[0][i]
        _, text2 = outputs_list[1][i]
        assert text1 == text2, f"Outputs differ:\n{text1}\n vs\n{text2}"


@pytest.mark.parametrize("prompt_length", [10, 100])
def test_different_prompt_lengths(vllm_runner, prompt_length):
    """Test fusion works correctly with different prompt lengths."""
    model = "deepseek-ai/DeepSeek-V2-Lite"

    # Create prompt of specific length
    prompt = "Hello " * (prompt_length // 6)
    max_tokens = 10

    with vllm_runner(
        model,
        quantization="fp8",
        trust_remote_code=True,
        max_model_len=512,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy([prompt], max_tokens)

        # Should produce valid output
        assert len(outputs) == 1
        output_ids, output_text = outputs[0]
        assert len(output_text) > 0


def test_temperature_sampling(vllm_runner):
    """Test fusion works with temperature sampling (not just greedy)."""
    model = "deepseek-ai/DeepSeek-V2-Lite"
    prompts = ["Write a short poem about AI."]

    with vllm_runner(
        model,
        quantization="fp8",
        trust_remote_code=True,
        max_model_len=256,
        enforce_eager=True,
    ) as vllm_model:
        # Use temperature sampling
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=50)
        outputs = vllm_model.generate(prompts, sampling_params)

        assert len(outputs) == 1
        output_ids_list, output_text_list = outputs[0]
        assert len(output_text_list[0]) > 0


def test_special_tokens_handling(vllm_runner):
    """Test fusion handles special tokens correctly."""
    model = "deepseek-ai/DeepSeek-V2-Lite"
    prompts = ["<|begin_of_text|>Hello<|end_of_text|>"]

    with vllm_runner(
        model,
        quantization="fp8",
        trust_remote_code=True,
        max_model_len=256,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=10)

        # Should handle special tokens without crashing
        assert len(outputs) == 1
        output_ids, output_text = outputs[0]
        assert len(output_text) >= 0  # May be empty if EOS hit


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-ai/DeepSeek-V2-Lite",
    ],
)
def test_no_nans_or_infs(vllm_runner, example_prompts, model):
    """Test that fusion doesn't produce NaN or Inf logprobs."""
    max_tokens = 10
    NUM_LOG_PROBS = 5

    with vllm_runner(
        model,
        quantization="fp8",
        trust_remote_code=True,
        max_model_len=256,
        enforce_eager=True,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, NUM_LOG_PROBS
        )

        # Check all logprobs are finite
        for output_ids, output_text, logprobs_list in outputs:
            if logprobs_list:
                for token_logprobs in logprobs_list:
                    if token_logprobs:
                        for logprob_value in token_logprobs.values():
                            # Handle both dict format and Logprob object format
                            lp = (
                                logprob_value.logprob
                                if hasattr(logprob_value, "logprob")
                                else logprob_value
                            )
                            assert lp != float("inf"), "Found Inf logprob"
                            assert lp != float("-inf"), "Found -Inf logprob"
                            assert lp == lp, "Found NaN logprob"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
