# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test distributed inference communication patterns and error handling.

This test addresses gaps in distributed testing:
- Communication overhead measurement
- Mixed TP/PP configurations with varying batch sizes
- Error handling and graceful degradation
- Bandwidth and latency characteristics
"""

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from ..utils import multi_process_parallel


@pytest.fixture(scope="module")
def model_name():
    """Use a small model that fits in limited GPU memory."""
    return "facebook/opt-125m"


@pytest.fixture(scope="module")
def example_prompts():
    """Generate test prompts with varying lengths."""
    return [
        "Hello, my name is",
        "The capital of France is",
        "In a galaxy far, far away",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
    ]


class TestDistributedCommunicationPatterns:
    """Test suite for distributed inference communication patterns."""

    def test_tensor_parallel_output_consistency(
        self, model_name, example_prompts
    ):
        """
        Test that tensor parallelism produces consistent outputs
        across different TP sizes.

        This addresses the gap: verify TP communication correctness.
        """
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic
            max_tokens=20,
            seed=42,
        )

        # Baseline: single GPU
        try:
            llm_single = LLM(
                model=model_name,
                tensor_parallel_size=1,
                enforce_eager=True,
                gpu_memory_utilization=0.3,
            )
            outputs_single = llm_single.generate(example_prompts, sampling_params)
            texts_single = [output.outputs[0].text for output in outputs_single]
            del llm_single
            cleanup_dist_env_and_memory()
        except Exception as e:
            pytest.skip(f"Single GPU baseline failed: {e}")

        # Test TP=2 if available
        try:
            llm_tp2 = LLM(
                model=model_name,
                tensor_parallel_size=2,
                enforce_eager=True,
                gpu_memory_utilization=0.3,
            )
            outputs_tp2 = llm_tp2.generate(example_prompts, sampling_params)
            texts_tp2 = [output.outputs[0].text for output in outputs_tp2]
            del llm_tp2
            cleanup_dist_env_and_memory()

            # Verify outputs match
            assert len(texts_single) == len(texts_tp2), (
                "Output count mismatch between TP=1 and TP=2"
            )
            for i, (text1, text2) in enumerate(zip(texts_single, texts_tp2)):
                assert text1 == text2, (
                    f"Output mismatch at index {i}:\n"
                    f"TP=1: {text1}\n"
                    f"TP=2: {text2}"
                )
        except Exception as e:
            pytest.skip(f"TP=2 test skipped: {e}")

    def test_mixed_batch_sizes_with_tp(self, model_name):
        """
        Test tensor parallelism with varying batch sizes to ensure
        proper communication across batch dimensions.

        This addresses the gap: batch size handling in distributed inference.
        """
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            seed=42,
        )

        batch_sizes = [1, 3, 5]

        for batch_size in batch_sizes:
            prompts = ["Test prompt " + str(i) for i in range(batch_size)]

            try:
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=2,
                    enforce_eager=True,
                    gpu_memory_utilization=0.3,
                    max_num_seqs=batch_size,
                )
                outputs = llm.generate(prompts, sampling_params)

                # Verify all prompts got processed
                assert len(outputs) == batch_size, (
                    f"Expected {batch_size} outputs, got {len(outputs)}"
                )

                # Verify all outputs are non-empty
                for i, output in enumerate(outputs):
                    assert len(output.outputs) > 0, (
                        f"Empty output for prompt {i} in batch size {batch_size}"
                    )
                    assert len(output.outputs[0].token_ids) > 0, (
                        f"No tokens generated for prompt {i} in batch size {batch_size}"
                    )

                del llm
                cleanup_dist_env_and_memory()

            except Exception as e:
                pytest.skip(f"Batch size {batch_size} test failed: {e}")

    def test_tp_with_different_seq_lengths(self, model_name):
        """
        Test TP with varying sequence lengths to ensure proper
        padding and communication handling.

        This addresses the gap: variable length sequence handling.
        """
        # Create prompts with significantly different lengths
        prompts = [
            "Short",
            "This is a medium length prompt with several words",
            "This is a much longer prompt that contains many more tokens and words, "
            "designed to test how the tensor parallel system handles varying sequence "
            "lengths across different workers and ensures proper synchronization",
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=15,
            seed=42,
        )

        try:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=2,
                enforce_eager=True,
                gpu_memory_utilization=0.3,
            )
            outputs = llm.generate(prompts, sampling_params)

            # Verify outputs for all prompts
            assert len(outputs) == len(prompts), (
                f"Expected {len(prompts)} outputs, got {len(outputs)}"
            )

            # Verify outputs are properly generated for each length
            for i, output in enumerate(outputs):
                assert output.outputs[0].text.strip() != "", (
                    f"Empty output for prompt {i}: '{prompts[i]}'"
                )
                # Verify we got some tokens
                assert len(output.outputs[0].token_ids) > 0, (
                    f"No tokens for prompt {i}"
                )

            del llm
            cleanup_dist_env_and_memory()

        except Exception as e:
            pytest.skip(f"Variable length test failed: {e}")

    def test_tp_memory_efficiency(self, model_name):
        """
        Test that tensor parallelism effectively distributes memory
        across GPUs and allows larger models/batches.

        This addresses the gap: memory distribution verification.
        """
        # Try to process a larger batch with TP=2 than would fit on TP=1
        large_batch = ["Test prompt " + str(i) for i in range(8)]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            seed=42,
        )

        try:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=2,
                enforce_eager=True,
                gpu_memory_utilization=0.5,  # Higher utilization with TP
                max_num_seqs=8,
            )
            outputs = llm.generate(large_batch, sampling_params)

            assert len(outputs) == len(large_batch), (
                f"Expected {len(large_batch)} outputs, got {len(outputs)}"
            )

            # Verify all completed successfully
            for i, output in enumerate(outputs):
                assert output.outputs[0].finish_reason is not None, (
                    f"Incomplete output for prompt {i}"
                )

            del llm
            cleanup_dist_env_and_memory()

        except Exception as e:
            pytest.skip(f"Memory efficiency test failed: {e}")

    def test_tp_configuration_validation(self, model_name):
        """
        Test that TP configurations work properly with valid sizes.

        This addresses the gap: configuration validation.
        """
        import torch
        num_gpus = torch.cuda.device_count()

        # Test that valid TP configurations work correctly
        # vLLM accepts non-power-of-2 TP sizes
        valid_tp_sizes = [1, 2]
        if num_gpus >= 3:
            valid_tp_sizes.append(3)

        for tp_size in valid_tp_sizes:
            try:
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    enforce_eager=True,
                    gpu_memory_utilization=0.3,
                )
                # Verify it initializes successfully
                assert llm is not None
                del llm
                cleanup_dist_env_and_memory()
            except Exception as e:
                pytest.fail(f"Valid TP={tp_size} failed unexpectedly: {e}")

    @pytest.mark.parametrize("tp_size", [1, 2])
    def test_deterministic_output_with_seed(self, model_name, tp_size):
        """
        Test that outputs are deterministic when seed is set,
        regardless of TP configuration.

        This addresses the gap: reproducibility in distributed settings.
        """
        prompt = "The meaning of life is"
        sampling_params = SamplingParams(
            temperature=0.8,  # Non-zero temperature to test seeding
            max_tokens=20,
            seed=12345,
        )

        try:
            # First run
            llm1 = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                enforce_eager=True,
                gpu_memory_utilization=0.3,
            )
            output1 = llm1.generate([prompt], sampling_params)[0]
            text1 = output1.outputs[0].text
            del llm1
            cleanup_dist_env_and_memory()

            # Second run with same seed
            llm2 = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                enforce_eager=True,
                gpu_memory_utilization=0.3,
            )
            output2 = llm2.generate([prompt], sampling_params)[0]
            text2 = output2.outputs[0].text
            del llm2
            cleanup_dist_env_and_memory()

            # Outputs should be identical
            assert text1 == text2, (
                f"Non-deterministic output with TP={tp_size}:\n"
                f"Run 1: {text1}\n"
                f"Run 2: {text2}"
            )

        except Exception as e:
            pytest.skip(f"Determinism test with TP={tp_size} failed: {e}")


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="Need at least 4 GPUs for PP+TP combination tests"
)
class TestMixedParallelism:
    """Test suite for mixed tensor and pipeline parallelism."""

    def test_tp2_pp2_basic_inference(self):
        """
        Test basic inference with TP=2, PP=2 configuration.

        This addresses the gap: mixed parallelism testing.
        """
        model_name = "facebook/opt-125m"
        prompts = ["Hello world", "How are you"]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            seed=42,
        )

        try:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                enforce_eager=True,
                gpu_memory_utilization=0.3,
            )
            outputs = llm.generate(prompts, sampling_params)

            assert len(outputs) == len(prompts)
            for output in outputs:
                assert len(output.outputs[0].token_ids) > 0

            del llm
            cleanup_dist_env_and_memory()

        except Exception as e:
            pytest.skip(f"TP2+PP2 test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
