# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.utils import opcheck
from vllm._custom_ops import (
    apply_repetition_penalties_cuda,
    apply_repetition_penalties_torch,
)
from vllm.platforms import current_platform

NUM_SEQS = [1, 2, 3, 4, 8, 13, 17, 32, 37, 256, 1023, 1024, 1025]
# [stress, stress, stress, Qwen, llama 4]
VOCAB_SIZES = [17, 256, 1019, 151936, 202048]
REPETITION_PENALTY_VALUES = [1.05]
SEEDS = [0]
DTYPES = [torch.float32, torch.float16]


@pytest.mark.parametrize("num_seqs", NUM_SEQS)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
@pytest.mark.parametrize("repetition_penalty", REPETITION_PENALTY_VALUES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test for checking CUDA kernel"
)
@torch.inference_mode()
def test_apply_repetition_penalties(
    num_seqs: int,
    vocab_size: int,
    repetition_penalty: float,
    dtype: torch.dtype,
    seed: int,
) -> None:
    """
    Test the apply_repetition_penalties custom op
    against a reference implementation.
    """
    current_platform.seed_everything(seed)
    torch.set_default_device("cuda:0")

    # Create test data
    logits = torch.randn(num_seqs, vocab_size, dtype=dtype)

    # Create masks with some random tokens marked as repeated
    prompt_mask = torch.zeros(num_seqs, vocab_size, dtype=torch.bool)
    output_mask = torch.zeros(num_seqs, vocab_size, dtype=torch.bool)

    # Mark some tokens as repeated in prompt and output
    prompt_indices = torch.randint(0, vocab_size, (num_seqs, max(1, vocab_size // 200)))
    output_indices = torch.randint(0, vocab_size, (num_seqs, max(1, vocab_size // 200)))

    for i in range(num_seqs):
        prompt_mask[i, prompt_indices[i]] = True
        output_mask[i, output_indices[i]] = True

    # Create repetition penalties tensor
    repetition_penalties = torch.full((num_seqs,), repetition_penalty, dtype=dtype)

    # Run all three implementations
    logits_torch = logits.clone()
    logits_cuda = logits.clone()

    apply_repetition_penalties_torch(
        logits_torch, prompt_mask, output_mask, repetition_penalties
    )
    apply_repetition_penalties_cuda(
        logits_cuda, prompt_mask, output_mask, repetition_penalties
    )

    # Compare all outputs to reference
    torch.testing.assert_close(logits_torch, logits_cuda, rtol=1e-3, atol=1e-3)

    # Test the operator by applying the opcheck utility
    opcheck(
        torch.ops._C.apply_repetition_penalties_,
        (logits.clone(), prompt_mask, output_mask, repetition_penalties),
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test for checking CUDA kernel"
)
@torch.inference_mode()
def test_apply_repetition_penalties_zero_seqs() -> None:
    """
    Test the apply_repetition_penalties custom op with num_seqs=0
    against a reference implementation.
    """
    num_seqs = 0
    vocab_size = 17
    repetition_penalty = 1.05
    dtype = torch.float32
    seed = 0

    current_platform.seed_everything(seed)
    torch.set_default_device("cuda:0")

    # Create test data
    logits = torch.randn(num_seqs, vocab_size, dtype=dtype)

    # Create masks with some random tokens marked as repeated
    prompt_mask = torch.zeros(num_seqs, vocab_size, dtype=torch.bool)
    output_mask = torch.zeros(num_seqs, vocab_size, dtype=torch.bool)

    # No tokens to mark as repeated since num_seqs=0

    # Create repetition penalties tensor
    repetition_penalties = torch.full((num_seqs,), repetition_penalty, dtype=dtype)

    # Run all three implementations
    logits_torch = logits.clone()
    logits_cuda = logits.clone()

    apply_repetition_penalties_torch(
        logits_torch, prompt_mask, output_mask, repetition_penalties
    )
    apply_repetition_penalties_cuda(
        logits_cuda, prompt_mask, output_mask, repetition_penalties
    )

    # Compare all outputs to reference
    torch.testing.assert_close(logits_torch, logits_cuda, rtol=1e-3, atol=1e-3)

    # Test the operator by applying the opcheck utility
    opcheck(
        torch.ops._C.apply_repetition_penalties_,
        (logits.clone(), prompt_mask, output_mask, repetition_penalties),
    )
