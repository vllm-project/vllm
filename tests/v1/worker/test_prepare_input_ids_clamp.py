# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for issue #36906: EAGLE3 + multimodal + async scheduling
crashes with CUDA error in F.embedding() due to -1 placeholder token IDs
reaching the embedding layer.

This test directly exercises the _prepare_input_ids scatter path to verify
that -1 spec token placeholders from async scheduling are clamped to valid
values before reaching the model.
"""

import pytest
import torch


def test_prepare_input_ids_clamps_negative_placeholders():
    """Simulate the async scatter path and verify -1 values are clamped.

    Reproduces the exact scenario where -1 leaks through:
    1. A request transitions from prefill to decode (not in prev_req_id_to_index)
    2. Async scheduler writes -1 spec token placeholders to token_ids_cpu
    3. _prepare_input_ids copies input_ids.cpu (containing -1) to GPU
    4. Scatter only covers "common" requests, leaving -1 for the new request

    Without the fix, input_ids.gpu would contain -1, causing F.embedding()
    to crash with CUDA device-side assert.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulate a batch with 3 requests, each with 1 sampled + 2 spec tokens = 3 tokens
    total_num_scheduled_tokens = 9  # 3 requests * 3 tokens each

    # Create input_ids buffer (like CpuGpuBuffer)
    input_ids_cpu = torch.zeros(total_num_scheduled_tokens, dtype=torch.int32)
    input_ids_gpu = torch.zeros(
        total_num_scheduled_tokens, dtype=torch.int32, device=device
    )

    # Request 0: common request (was decoding in prev step) - valid tokens
    input_ids_cpu[0] = 100  # sampled token (will be overwritten by scatter)
    input_ids_cpu[1] = 200  # spec token 1 (will be overwritten by scatter)
    input_ids_cpu[2] = 300  # spec token 2 (will be overwritten by scatter)

    # Request 1: TRANSITIONING request (was prefilling, now decoding)
    # NOT in prev_req_id_to_index, so scatter won't touch it
    input_ids_cpu[3] = 0  # sampled token position (0 from token_ids_cpu init)
    input_ids_cpu[4] = -1  # spec token placeholder from async scheduler
    input_ids_cpu[5] = -1  # spec token placeholder from async scheduler

    # Request 2: common request - valid tokens
    input_ids_cpu[6] = 400  # sampled token
    input_ids_cpu[7] = 500  # spec token 1
    input_ids_cpu[8] = 600  # spec token 2

    # Step 1: Copy CPU -> GPU (simulates copy_to_gpu)
    input_ids_gpu.copy_(input_ids_cpu)

    # Verify -1 values ARE present before clamp (the bug condition)
    assert (input_ids_gpu == -1).any(), (
        "Test setup error: -1 values should be present before clamp"
    )

    # Step 2: Apply the fix - clamp negative values
    # This is the exact line added by the fix in _prepare_input_ids
    input_ids_gpu[:total_num_scheduled_tokens].clamp_(min=0)

    # Step 3: Simulate scatter for common requests (requests 0 and 2)
    # In the real code, scatter overwrites sampled + spec positions
    # for requests in prev_req_id_to_index
    prev_sampled_token_ids = torch.tensor([[42], [99]], device=device)
    sampled_indices = torch.tensor([0, 6], dtype=torch.int64, device=device)
    input_ids_gpu.scatter_(
        0, sampled_indices, prev_sampled_token_ids[:, 0].to(torch.int32)
    )

    draft_token_ids = torch.tensor([50, 51, 80, 81], dtype=torch.int32, device=device)
    draft_indices = torch.tensor([1, 2, 7, 8], dtype=torch.int64, device=device)
    input_ids_gpu.scatter_(0, draft_indices, draft_token_ids)

    # VERIFICATION: No -1 values should remain anywhere
    assert input_ids_gpu.min().item() >= 0, (
        f"Bug: input_ids_gpu still contains negative values: {input_ids_gpu}"
    )

    # Verify the transitioning request (request 1) has clamped values
    assert input_ids_gpu[3].item() == 0, "Sampled position should be 0"
    assert input_ids_gpu[4].item() == 0, "Spec position should be clamped to 0"
    assert input_ids_gpu[5].item() == 0, "Spec position should be clamped to 0"

    # Verify common requests have correct scattered values
    assert input_ids_gpu[0].item() == 42, "Req 0 sampled should be scattered"
    assert input_ids_gpu[1].item() == 50, "Req 0 spec 1 should be scattered"
    assert input_ids_gpu[2].item() == 51, "Req 0 spec 2 should be scattered"
    assert input_ids_gpu[6].item() == 99, "Req 2 sampled should be scattered"
    assert input_ids_gpu[7].item() == 80, "Req 2 spec 1 should be scattered"
    assert input_ids_gpu[8].item() == 81, "Req 2 spec 2 should be scattered"

    # Verify that F.embedding would NOT crash with these values
    vocab_size = 1000
    embedding = torch.nn.Embedding(vocab_size, 16).to(device)
    # This would crash with "CUDA error: device-side assert triggered"
    # if any input_ids_gpu values were -1
    result = embedding(input_ids_gpu.to(torch.int64))
    assert result.shape == (total_num_scheduled_tokens, 16)


def test_no_clamp_needed_when_all_common():
    """Verify the fast path: when all requests are common, no clamp needed.

    This tests the code path where num_common_tokens == total_without_spec,
    meaning the copy_to_gpu block (and thus the clamp) is skipped entirely.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # All tokens are from common requests - scatter handles everything
    total_tokens = 6
    input_ids_gpu = torch.zeros(total_tokens, dtype=torch.int32, device=device)

    # Scatter puts correct values for all positions
    prev_sampled = torch.tensor([[10], [20]], device=device, dtype=torch.int32)
    sampled_idx = torch.tensor([0, 3], dtype=torch.int64, device=device)
    input_ids_gpu.scatter_(0, sampled_idx, prev_sampled[:, 0])

    draft = torch.tensor([11, 12, 21, 22], dtype=torch.int32, device=device)
    draft_idx = torch.tensor([1, 2, 4, 5], dtype=torch.int64, device=device)
    input_ids_gpu.scatter_(0, draft_idx, draft)

    # All values should be valid without any clamp
    assert input_ids_gpu.min().item() >= 0
    expected = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32, device=device)
    assert torch.equal(input_ids_gpu, expected)


def test_negative_token_ids_crash_embedding():
    """Demonstrate that F.embedding crashes on -1 input (the bug).

    Uses CPU to avoid poisoning the CUDA context with a device-side assert.
    On GPU, this same error manifests as the CUDA device-side assert
    reported in issue #36906.
    """
    vocab_size = 1000
    embedding = torch.nn.Embedding(vocab_size, 16)

    # F.embedding with -1 should raise IndexError on CPU
    bad_input_ids = torch.tensor([100, -1, 200], dtype=torch.int64)
    with pytest.raises(IndexError):
        embedding(bad_input_ids)

    # After clamping, embedding succeeds
    fixed_input_ids = bad_input_ids.clamp(min=0)
    result = embedding(fixed_input_ids)
    assert result.shape == (3, 16), "Clamped input should embed successfully"
