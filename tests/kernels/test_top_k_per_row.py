# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

# Test parameters
NUM_ROWS = [1, 32, 2050]
TOP_K_VALUES = [2048]
BATCH_SIZE = [1, 2, 2048]
NEXT_N = [1, 2, 8]
DATA_GENERATION = ["random", "10LSBits"]


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    data_generation: str,
) -> torch.Tensor:
    """Create random logits tensor for testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Generate logits with some structure to make testing more meaningful
    if data_generation == "random":
        logits = torch.randn(
            row_starts.shape[0], max(row_ends), dtype=dtype, device="cuda"
        )
    elif data_generation == "10LSBits":
        top_22_bits_mask = 0xFFFFFC00
        last_10_bits_mask = 0x000003FF
        fixed_top_22_bits = 0x3F900000
        # Generate random bits for the last 10 bits
        random_bottom_bits = torch.randint(
            0,
            2**10,
            (row_starts.shape[0], max(row_ends)),
            dtype=torch.int32,
            device="cuda",
        )
        # Combine: fixed top 22 bits with random last 10 bits
        logits_bits = (fixed_top_22_bits & top_22_bits_mask) | (
            random_bottom_bits & last_10_bits_mask
        )
        logits = logits_bits.view(dtype)

    for i, end in enumerate(row_ends):
        logits[i, end:] = float("-inf")
    return logits


def create_row_boundaries(
    seq_len: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create row start and end indices for testing."""
    row_starts = torch.zeros(seq_len, dtype=torch.int32, device="cuda")
    row_ends = torch.arange(1, seq_len + 1, device="cuda", dtype=torch.int32)
    return row_starts, row_ends


def compare_top_k_results(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from CUDA top_k_per_row with torch.topk.
    Both results should be sorted and contain the same top-k elements.
    """
    num_rows = cuda_indices.shape[0]

    for row_idx in range(num_rows):
        # Get valid elements using row boundaries
        row_start = row_starts[row_idx].item()
        row_end = row_ends[row_idx].item()
        row_length = row_end - row_start
        num_valid = min(top_k, row_length)
        cuda_row_indices = cuda_indices[row_idx][:num_valid].cpu()
        torch_row_indices = torch_indices[row_idx][:num_valid].cpu()

        # Compare the sets of indices first
        cuda_set = set(cuda_row_indices.tolist())
        torch_set = set(torch_row_indices.tolist())
        if cuda_set == torch_set:
            continue

        # Any difference in elements, compare the values
        logits_row = logits[row_idx]
        cuda_row_values = [logits_row[i] for i in cuda_row_indices]
        torch_row_values = [logits_row[i] for i in torch_row_indices]

        cuda_only_values, torch_only_values = [], []
        for idx in cuda_set - torch_set:
            cuda_pos = (cuda_row_indices == idx).nonzero(as_tuple=True)[0]
            cuda_only_values.append(cuda_row_values[cuda_pos[0]])

        for idx in torch_set - cuda_set:
            torch_pos = (torch_row_indices == idx).nonzero(as_tuple=True)[0]
            torch_only_values.append(torch_row_values[torch_pos[0]])

        if len(cuda_only_values) != len(torch_only_values):
            return False
        if not torch.allclose(
            torch.tensor(cuda_only_values),
            torch.tensor(torch_only_values),
            rtol=tolerance,
            atol=tolerance,
        ):
            return False

    return True


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_top_k_per_row(
    num_rows: int,
    top_k: int,
) -> None:
    """
    Test top_k_per_row.
    """
    torch.set_default_device("cuda:0")

    # Create test data
    vocab_size = 20000
    row_starts, row_ends = create_row_boundaries(num_rows, vocab_size)
    logits = create_random_logits(row_starts, row_ends, torch.float32, 42, "random")

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    # Run CUDA implementation
    torch.ops._C.top_k_per_row(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
    )

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, top_k
    ), "CUDA top_k_per_row results don't match torch.topk"


@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("next_n", NEXT_N)
@pytest.mark.parametrize("data_generation", DATA_GENERATION)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_top_k_per_row_decode(
    top_k: int,
    batch_size: int,
    next_n: int,
    data_generation: str,
) -> None:
    """
    Test top_k_per_row with seq_lens tensor.
    """
    torch.set_default_device("cuda:0")

    # Create test data
    num_rows = batch_size * next_n
    vocab_size = 20000
    seq_lens = torch.randint(
        vocab_size, (batch_size,), dtype=torch.int32, device="cuda"
    )
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_rows, device="cuda") // next_n
    next_n_offset = torch.arange(num_rows, device="cuda") % next_n
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, data_generation
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    # Run CUDA implementation
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
    )

    torch.cuda.synchronize()

    # Run reference implementation
    torch_indices = logits.topk(min(top_k, max(row_ends)), dim=-1)[1]
    mask_lo = torch_indices >= 0
    mask_hi = (torch_indices - (row_ends - row_starts)[:, None]) < 0
    mask = mask_lo & mask_hi
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, top_k
    ), "CUDA top_k_per_row results don't match torch.topk"
