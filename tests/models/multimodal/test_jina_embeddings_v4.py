# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import time
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import torch
from PIL import Image

from vllm import LLM
from vllm.config import PoolerConfig
from vllm.inputs.data import TextPrompt
from vllm.sequence import SequenceData

model_name = "jinaai/jina-embeddings-v4-vllm-retrieval"

# Vision token IDs
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653


@pytest.fixture(scope="module")
def model():
    """Initialize model once for all tests."""
    return LLM(
        model=model_name,
        task="embed",
        override_pooler_config=PoolerConfig(pooling_type="ALL",
                                            normalize=False),
        dtype="float16",
        max_model_len=2048,
    )


def extract_embeddings(output):
    """Extract embeddings based on token type."""
    if VISION_START_TOKEN_ID in output.prompt_token_ids:
        # Extract vision tokens only
        img_start = output.prompt_token_ids.index(VISION_START_TOKEN_ID)
        img_end = output.prompt_token_ids.index(VISION_END_TOKEN_ID)
        embeddings = output.outputs.data[img_start:img_end + 1]
    else:
        # Use all tokens for text
        embeddings = output.outputs.data

    # Mean pool and normalize
    pooled = embeddings.mean(dim=0, dtype=torch.float32)
    return torch.nn.functional.normalize(pooled, dim=-1)


class TestBasicFunctionality:
    """Test basic embedding generation functionality."""

    def test_text_only_embeddings(self, model):
        """Test text-only embedding generation."""
        prompts = [
            TextPrompt(prompt="Query: What is machine learning?"),
            TextPrompt(prompt="Passage: Machine learning is a subset of "
                       "artificial intelligence.")
        ]

        outputs = model.encode(prompts)
        embeddings = [extract_embeddings(output) for output in outputs]

        # Check embeddings are normalized
        for emb in embeddings:
            assert torch.allclose(emb.norm(), torch.tensor(1.0), atol=1e-3)

        # Check similarity is reasonable
        similarity = torch.dot(embeddings[0], embeddings[1]).item()
        assert 0.0 <= similarity <= 1.0

    def test_image_embeddings(self, model):
        """Test image embedding generation."""
        # Create a dummy image
        image = Image.new('RGB', (224, 224), color='red')

        prompt = TextPrompt(
            prompt="<|im_start|>user\n<|vision_start|><|image_pad|>"
            "<|vision_end|>Describe the image.<|im_end|>\n",
            multi_modal_data={"image": image},
        )

        outputs = model.encode([prompt])
        embedding = extract_embeddings(outputs[0])

        # Check embedding is normalized
        assert torch.allclose(embedding.norm(), torch.tensor(1.0), atol=1e-3)

        # Check dimension
        assert embedding.shape[
            0] == model.llm_engine.model_config.hf_config.hidden_size

    def test_mixed_batch(self, model):
        """Test mixed text and image batch processing."""
        image = Image.new('RGB', (224, 224), color='blue')

        prompts = [
            TextPrompt(prompt="Query: blue color"),
            TextPrompt(
                prompt="<|im_start|>user\n<|vision_start|><|image_pad|>"
                "<|vision_end|>Describe the image.<|im_end|>\n",
                multi_modal_data={"image": image},
            ),
            TextPrompt(prompt="Passage: The sky is blue.")
        ]

        outputs = model.encode(prompts)
        embeddings = [extract_embeddings(output) for output in outputs]

        # All embeddings should be normalized
        for emb in embeddings:
            assert torch.allclose(emb.norm(), torch.tensor(1.0), atol=1e-3)

        # Text query about blue should have some similarity to blue image
        text_image_sim = torch.dot(embeddings[0], embeddings[1]).item()
        assert text_image_sim > 0.0  # Should have positive similarity


class TestThreadSafety:
    """Test thread safety of the pooling implementation."""

    def test_concurrent_requests(self, model):
        """Test handling of concurrent embedding requests."""
        num_threads = 4
        requests_per_thread = 5

        def process_request(thread_id):
            results = []
            for i in range(requests_per_thread):
                prompt = TextPrompt(
                    prompt=f"Query from thread {thread_id}, request {i}")
                outputs = model.encode([prompt])
                embedding = extract_embeddings(outputs[0])
                results.append(embedding)
            return results

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(process_request, i) for i in range(num_threads)
            ]

            all_results = []
            for future in as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)

        # Verify all embeddings are valid
        assert len(all_results) == num_threads * requests_per_thread
        for emb in all_results:
            assert torch.allclose(emb.norm(), torch.tensor(1.0), atol=1e-3)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input_handling(self, model):
        """Test handling of empty inputs."""
        # This should not crash but return empty outputs
        outputs = model.encode([])
        assert len(outputs) == 0

    def test_very_long_sequence(self, model):
        """Test handling of sequences near max length."""
        # Create a long text that approaches max_model_len
        long_text = " ".join(["word"] * 1000)
        prompt = TextPrompt(prompt=f"Query: {long_text}")

        # Should handle gracefully without crashing
        outputs = model.encode([prompt])
        embedding = extract_embeddings(outputs[0])
        assert torch.allclose(embedding.norm(), torch.tensor(1.0), atol=1e-3)

    def test_invalid_image_format(self, model):
        """Test handling of invalid image inputs."""
        # Create an invalid image (too small)
        tiny_image = Image.new('RGB', (1, 1), color='red')

        prompt = TextPrompt(
            prompt="<|im_start|>user\n<|vision_start|><|image_pad|>"
            "<|vision_end|>Describe the image.<|im_end|>\n",
            multi_modal_data={"image": tiny_image},
        )

        # Should handle gracefully
        try:
            outputs = model.encode([prompt])
            # If it doesn't crash, check output is valid
            if outputs:
                embedding = extract_embeddings(outputs[0])
                assert embedding.shape[
                    0] == model.llm_engine.model_config.hf_config.hidden_size
        except Exception as e:
            # Should provide meaningful error message
            assert "image" in str(e).lower() or "size" in str(e).lower()


class TestMemoryManagement:
    """Test memory management and cleanup."""

    def test_memory_cleanup(self, model):
        """Test that memory is properly cleaned up after processing."""
        # Get initial memory usage
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()

        # Process multiple large batches
        for _ in range(5):
            prompts = [
                TextPrompt(prompt=f"Query: test {i}") for i in range(10)
            ]
            outputs = model.encode(prompts)
            del outputs
            gc.collect()

        # Check memory usage hasn't grown significantly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            # Allow some growth but not excessive
            assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth


class TestPerformance:
    """Test performance characteristics."""

    def test_pooling_performance(self, model):
        """Test that custom pooling is performant."""
        # Create test prompts
        text_prompts = [
            TextPrompt(prompt=f"Query: test {i}") for i in range(10)
        ]

        # Time text-only pooling
        start_time = time.time()
        text_outputs = model.encode(text_prompts)
        text_time = time.time() - start_time

        # Create image prompts
        image = Image.new('RGB', (224, 224), color='green')
        image_prompts = [
            TextPrompt(
                prompt="<|im_start|>user\n<|vision_start|><|image_pad|>"
                "<|vision_end|>Describe.<|im_end|>\n",
                multi_modal_data={"image": image},
            ) for _ in range(10)
        ]

        # Time vision pooling
        start_time = time.time()
        image_outputs = model.encode(image_prompts)
        image_time = time.time() - start_time

        # Vision pooling should not be significantly slower
        # (allowing 2x slower due to additional processing)
        assert image_time < text_time * 2.0

        # Verify outputs are valid
        for output in text_outputs + image_outputs:
            embedding = extract_embeddings(output)
            assert torch.allclose(embedding.norm(),
                                  torch.tensor(1.0),
                                  atol=1e-3)


class TestPoolingMetadataIntegration:
    """Test proper integration with PoolingMetadata."""

    def test_seq_data_access(self):
        """Test that token IDs are properly accessible via seq_data."""
        # Create mock sequence data
        prompt_tokens = array('l', [
            101, 102, VISION_START_TOKEN_ID, VISION_START_TOKEN_ID,
            VISION_END_TOKEN_ID, 104
        ])
        seq_data = SequenceData(prompt_tokens)

        # Verify prompt_token_ids_array property works
        assert hasattr(seq_data, 'prompt_token_ids_array')
        retrieved_tokens = seq_data.prompt_token_ids_array
        assert list(retrieved_tokens) == list(prompt_tokens)

        # Verify vision tokens can be detected
        token_tensor = torch.tensor(list(retrieved_tokens))
        vision_mask = ((token_tensor >= VISION_START_TOKEN_ID) &
                       (token_tensor <= VISION_END_TOKEN_ID))
        assert vision_mask.any()
        assert vision_mask.sum() == 3  # Start, middle, end tokens


class TestAccuracyValidation:
    """Test accuracy against expected behavior."""

    @pytest.mark.parametrize("text", [
        "Short text",
        "A much longer text that contains multiple sentences for testing",
        "特殊字符测试 🚀 emoji test", "Numbers 12345 and symbols !@#$%"
    ])
    def test_text_embedding_consistency(self, model, text):
        """Test that same text produces consistent embeddings."""
        prompt = TextPrompt(prompt=f"Query: {text}")

        # Generate embeddings multiple times
        embeddings = []
        for _ in range(3):
            outputs = model.encode([prompt])
            emb = extract_embeddings(outputs[0])
            embeddings.append(emb)

        # All should be identical
        for i in range(1, len(embeddings)):
            assert torch.allclose(embeddings[0], embeddings[i], atol=1e-5)

    def test_vision_only_pooling(self, model):
        """Test that vision pooling extracts only vision tokens."""
        # Create an image with known characteristics
        image = Image.new('RGB', (224, 224), color='red')

        # Two prompts with same image but different text
        prompt1 = TextPrompt(
            prompt="<|im_start|>user\n<|vision_start|><|image_pad|>"
            "<|vision_end|>Red image<|im_end|>\n",
            multi_modal_data={"image": image},
        )
        prompt2 = TextPrompt(
            prompt="<|im_start|>user\n<|vision_start|><|image_pad|>"
            "<|vision_end|>Blue sky green grass<|im_end|>\n",
            multi_modal_data={"image": image},
        )

        outputs = model.encode([prompt1, prompt2])
        emb1 = extract_embeddings(outputs[0])
        emb2 = extract_embeddings(outputs[1])

        # Since both use the same image and vision-only pooling,
        # embeddings should be very similar despite different text
        similarity = torch.dot(emb1, emb2).item()
        assert similarity > 0.99  # Should be nearly identical
