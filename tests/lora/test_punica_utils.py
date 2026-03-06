# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.lora.punica_wrapper.utils module."""

import time

import pytest
import torch

from vllm.lora.layers import LoRAMapping
from vllm.lora.punica_wrapper.utils import convert_mapping


class TestConvertMapping:
    """Tests for the convert_mapping function."""

    @pytest.fixture
    def device(self):
        """Get the device to use for tests."""
        return torch.device("cpu")

    def test_basic_mapping(self, device):
        """Test basic LoRA mapping conversion."""
        # Create a simple mapping with 2 LoRAs
        # index_mapping maps each token position to a LoRA id
        # prompt_mapping maps each request to a LoRA id
        mapping = LoRAMapping(
            index_mapping=(1, 1, 2, 2),  # 4 tokens, 2 from each LoRA
            prompt_mapping=(1, 2),  # 2 requests
        )
        # lora_index_to_id maps slot indices to LoRA ids
        # None means empty slot, actual LoRA ids are stored at their indices
        lora_index_to_id = [None, 1, 2, None]  # LoRA 1 at slot 1, LoRA 2 at slot 2

        base_indices, sampler_indices, _, _, indices_len = convert_mapping(
            mapping=mapping,
            lora_index_to_id=lora_index_to_id,
            max_loras=4,
            vocab_size=32000,
            extra_vocab_size=256,
            device=device,
        )

        # base_indices should map each token to the correct LoRA slot index
        assert base_indices.tolist() == [1, 1, 2, 2]
        # sampler_indices should map each request to LoRA slot index
        assert sampler_indices.tolist() == [1, 2]

    def test_no_lora_tokens(self, device):
        """Test mapping with tokens that don't use LoRA (id <= 0)."""
        mapping = LoRAMapping(
            index_mapping=(0, 1, 0, 2),  # 0 means no LoRA
            prompt_mapping=(0, 1),
        )
        lora_index_to_id = [None, 1, 2]

        base_indices, sampler_indices, _, _, _ = convert_mapping(
            mapping=mapping,
            lora_index_to_id=lora_index_to_id,
            max_loras=3,
            vocab_size=32000,
            extra_vocab_size=256,
            device=device,
        )

        # Tokens with id=0 should get lora_idx=-1
        assert base_indices.tolist() == [-1, 1, -1, 2]
        assert sampler_indices.tolist() == [-1, 1]

    def test_many_loras_performance(self, device):
        """Test that performance is acceptable with many LoRAs.

        This test verifies the O(1) lookup optimization by comparing
        execution time with a large number of LoRAs and mappings.
        """
        num_loras = 100
        num_tokens = 1000

        # Create mapping with many tokens using various LoRAs
        index_mapping = tuple((i % num_loras) + 1 for i in range(num_tokens))
        prompt_mapping = tuple((i % num_loras) + 1 for i in range(num_tokens // 10))

        mapping = LoRAMapping(
            index_mapping=index_mapping,
            prompt_mapping=prompt_mapping,
        )

        # Create lora_index_to_id with many LoRAs
        lora_index_to_id: list[int | None] = [None] * (num_loras + 1)
        for i in range(1, num_loras + 1):
            lora_index_to_id[i] = i

        # Time the conversion
        start_time = time.perf_counter()
        for _ in range(10):  # Run multiple times for more stable measurement
            convert_mapping(
                mapping=mapping,
                lora_index_to_id=lora_index_to_id,
                max_loras=num_loras + 1,
                vocab_size=32000,
                extra_vocab_size=256,
                device=device,
            )
        elapsed = time.perf_counter() - start_time

        # With O(1) dict lookup, this should complete quickly
        # With O(n) list.index(), this would be much slower
        # Allow generous 1 second for 10 iterations on slow CI machines
        assert elapsed < 1.0, (
            f"convert_mapping took {elapsed:.3f}s for 10 iterations, "
            "which suggests O(n) lookup is being used instead of O(1)"
        )

    def test_output_tensor_shapes(self, device):
        """Test that output tensors have correct shapes."""
        mapping = LoRAMapping(
            index_mapping=(1, 1, 1, 2, 2),
            prompt_mapping=(1, 2),
        )
        lora_index_to_id = [None, 1, 2]

        (
            base_indices,
            sampler_indices,
            sampler_indices_padded,
            embeddings_indices,
            indices_len,
        ) = convert_mapping(
            mapping=mapping,
            lora_index_to_id=lora_index_to_id,
            max_loras=3,
            vocab_size=32000,
            extra_vocab_size=256,
            device=device,
        )

        assert base_indices.shape == (5,)
        assert sampler_indices.shape == (2,)
        assert sampler_indices_padded.shape == (2,)
        assert embeddings_indices.shape == (2, 5)
        assert indices_len == [5, 2, 2, 5]
