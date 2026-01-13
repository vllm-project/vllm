# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import Generator

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_and_top_p,
    apply_top_k_top_p,
)

DEVICE = current_platform.device_type

BATCH_SIZE = 1024
VOCAB_SIZE = 128 * 1024


@pytest.fixture(autouse=True)
def reset_default_device():
    """
    Explicitly set the default device, which can affect subsequent tests.
    Adding this fixture helps avoid this problem.
    """
    original_device = torch.get_default_device()
    yield
    torch.set_default_device(original_device)


def test_topk_impl_equivalence():
    torch.set_default_device(DEVICE)
    generator = Generator(device=DEVICE).manual_seed(33)

    logits = torch.rand((BATCH_SIZE, VOCAB_SIZE), generator=generator)

    # Random top-k values between 1 and 9.
    k = torch.randint(1, 10, (BATCH_SIZE,), generator=generator)

    # Set k=vocab_size for ~50% of requests in the batch (top-k disabled).
    k.masked_fill_(
        torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=bool), VOCAB_SIZE
    )

    # Top-k only implementation
    result1 = apply_top_k_top_p(logits=logits.clone(), k=k, p=None)

    # Top-p + top-k
    no_op_top_p = torch.tensor([1.0])
    result2 = apply_top_k_top_p(logits=logits.clone(), k=k, p=no_op_top_p)

    assert torch.allclose(result1, result2)


def test_flashinfer_sampler():
    """
    This test verifies that the FlashInfer top-k and top-p sampling
    implementation produces the same results as the Python implementation.

    NOTE: FlashInfer did not directly expose an interface for fused top-k and
    top-p prob renorm (it did provide fused sampling but we cannot compare
    sampling results due to randomness), so we will compare the probability
    renormed consequently by top-k and then top-p of FlashInfer implementation.
    """
    try:
        from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

        is_flashinfer_available = True
    except ImportError:
        is_flashinfer_available = False

    FLASHINFER_ENABLED = current_platform.is_cuda() and is_flashinfer_available

    if not FLASHINFER_ENABLED:
        pytest.skip("FlashInfer not installed or not available on this platform.")

    torch.set_default_device(DEVICE)
    generator = Generator(device=DEVICE).manual_seed(42)

    # Generate random logits
    logits = torch.rand((BATCH_SIZE, VOCAB_SIZE), generator=generator)

    # Generate various top-k and top-p values
    k_values = torch.randint(1, 1000, (BATCH_SIZE,), generator=generator)
    p_values = (
        torch.rand((BATCH_SIZE,), generator=generator) * 0.5 + 0.5
    )  # range in [0.5, 1.0]

    # Sometimes disable top-k (k=vocab_size)
    k_values.masked_fill_(
        torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=torch.bool),
        VOCAB_SIZE,
    )

    # Sometimes disable top-p (p=1.0)
    p_values.masked_fill_(
        torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=torch.bool), 1.0
    )

    python_logits = apply_top_k_top_p(
        logits=logits.clone(),
        k=k_values,
        p=p_values,
    )
    python_probs = torch.softmax(python_logits, dim=-1)

    # FlashInfer only exposed renorm interfaces for probs so convert first
    flashinfer_probs = torch.softmax(logits.clone(), dim=-1)
    flashinfer_probs = top_k_renorm_probs(
        probs=flashinfer_probs,
        top_k=k_values,
    )
    flashinfer_probs = top_p_renorm_probs(
        probs=flashinfer_probs,
        top_p=p_values,
    )

    # Compare the results
    assert torch.allclose(python_probs, flashinfer_probs, atol=2e-2), (
        "FlashInfer and Python sampling implementations do not match!"
    )


def test_apply_top_k_and_top_p_correctness():
    """
    Test that apply_top_k_and_top_p correctly masks logits.
    Verifies that:
    1. Only the top-k values remain (others are -inf)
    2. Top-p filtering further reduces candidates
    3. At least one token is always kept
    """
    torch.set_default_device(DEVICE)

    # Small example for manual verification.
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # [1, 5]
    k = torch.tensor([3])  # Keep top 3.
    p = torch.tensor([0.9])  # Keep top 90% probability mass.

    result = apply_top_k_and_top_p(logits.clone(), k=k, p=p)

    # Top-3 values are 5.0, 4.0, 3.0 (indices 4, 3, 2).
    # After top-k mask, indices 0, 1 should be -inf.
    assert result[0, 0] == float("-inf"), "Index 0 should be masked by top-k"
    assert result[0, 1] == float("-inf"), "Index 1 should be masked by top-k"

    # Top-p with p=0.9 masks value 3.0 (its cumsum prob 0.09 <= 1-p=0.1).
    assert result[0, 2] == float("-inf"), (
        "Index 2 (value 3.0) should be masked by top-p"
    )

    # Values 4.0 and 5.0 should be preserved.
    assert result[0, 3] == 4.0, "Index 3 (value 4.0) should be preserved"
    assert result[0, 4] == 5.0, "Index 4 (value 5.0) should be preserved"

    # Verify exactly 2 tokens remain.
    num_valid = (result != float("-inf")).sum().item()
    assert num_valid == 2, f"Expected 2 valid tokens, got {num_valid}"


def test_apply_top_k_and_top_p_equivalence():
    """
    Test that apply_top_k_and_top_p matches the general sorting path.
    """
    torch.set_default_device(DEVICE)
    generator = Generator(device=DEVICE).manual_seed(42)

    batch_size = 256
    vocab_size = 8 * 1024

    logits = torch.rand((batch_size, vocab_size), generator=generator)
    k = torch.randint(1, 500, (batch_size,), generator=generator)
    p = torch.rand((batch_size,), generator=generator) * 0.5 + 0.5

    # Optimized path: all k < vocab_size triggers apply_top_k_and_top_p.
    result = apply_top_k_top_p(logits.clone(), k=k, p=p)

    # General path: set k[0]=vocab_size to force full-sort fallback.
    k_general = k.clone()
    k_general[0] = vocab_size
    expected = apply_top_k_top_p(logits.clone(), k=k_general, p=p)

    # Compare rows 1+ (row 0 has different k values).
    assert torch.equal(result[1:], expected[1:]), (
        "Optimized and general paths don't match!"
    )
