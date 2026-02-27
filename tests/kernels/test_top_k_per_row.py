# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Test parameters
NUM_ROWS = [1, 32, 2050]
TOP_K_VALUES = [2048, 3000]
BATCH_SIZE = [1, 2, 2048]
NEXT_N = [1, 8]
DATA_GENERATION = ["random", "10LSBits"]


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    clean_logits: bool,
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

    if clean_logits:
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


def validate_topk_against_reference(
    logits: torch.Tensor,
    cuda_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    kernel_name: str,
) -> None:
    """
    Validate CUDA top-k results against PyTorch reference implementation.

    Args:
        logits: Input logits tensor
        cuda_indices: CUDA kernel output indices
        row_starts: Row start positions
        row_ends: Row end positions
        top_k: Number of top elements to select
        kernel_name: Name of the kernel being tested (for error messages)
    """
    num_rows = cuda_indices.shape[0]
    torch_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    for i in range(num_rows):
        row_end = int(row_ends[i])
        k_i = min(top_k, row_end)
        idx = logits[i, :row_end].topk(k_i, dim=-1)[1]
        torch_indices[i, :k_i] = idx

    assert compare_top_k_results(
        logits, cuda_indices, torch_indices, row_starts, row_ends, top_k
    ), f"{kernel_name} results don't match torch.topk"


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("clean_logits", [True, False])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_top_k_per_row(
    num_rows: int,
    top_k: int,
    clean_logits: bool,
) -> None:
    """
    Test top_k_per_row.
    """
    set_random_seed(0)
    torch.set_default_device("cuda:0")

    # Create test data
    vocab_size = 20000
    row_starts, row_ends = create_row_boundaries(num_rows, vocab_size)
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, clean_logits, "random"
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    # Run CUDA implementation
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )

    # Run reference implementation
    torch_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    for i in range(num_rows):
        row_end = int(row_ends[i])
        k_i = min(top_k, row_end)
        idx = logits[i, :row_end].topk(k_i, dim=-1)[1]
        torch_indices[i, :k_i] = idx

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, top_k
    ), "CUDA top_k_per_row_prefill results don't match torch.topk"


def _run_top_k_per_row_decode_test(
    top_k: int,
    batch_size: int,
    next_n: int,
    vocab_size: int,
    clean_logits: bool,
    data_generation: str,
) -> None:
    """
    Helper function to run top_k_per_row_decode test with given parameters.
    """
    torch.set_default_device("cuda:0")

    # Create test data
    num_rows = batch_size * next_n
    seq_lens = torch.randint(
        low=next_n,
        high=vocab_size,
        size=(batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_rows, device="cuda") // next_n
    next_n_offset = torch.arange(num_rows, device="cuda") % next_n
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, clean_logits, data_generation
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
        top_k,
    )

    torch.cuda.synchronize()

    # Run reference implementation
    torch_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    for i in range(num_rows):
        row_end = int(row_ends[i])
        k_i = min(top_k, row_end)
        idx = logits[i, :row_end].topk(k_i, dim=-1)[1]
        torch_indices[i, :k_i] = idx

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, top_k
    ), "CUDA top_k_per_row_decode results don't match torch.topk"


@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("next_n", NEXT_N)
@pytest.mark.parametrize("clean_logits", [True, False])
@pytest.mark.parametrize("data_generation", DATA_GENERATION)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_top_k_per_row_decode(
    top_k: int,
    batch_size: int,
    next_n: int,
    clean_logits: bool,
    data_generation: str,
) -> None:
    """
    Test top_k_per_row with seq_lens tensor.
    """
    set_random_seed(0)
    vocab_size = 20000
    _run_top_k_per_row_decode_test(
        top_k, batch_size, next_n, vocab_size, clean_logits, data_generation
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("clean_logits", [True, False])
@torch.inference_mode()
def test_top_k_per_row_decode_large_vocab_size(clean_logits: bool) -> None:
    """
    Test top_k_per_row_decode with large vocabulary size.
    """
    set_random_seed(0)
    top_k = 2048
    batch_size = 2
    next_n = 2
    vocab_size = 300000
    data_generation = "random"
    _run_top_k_per_row_decode_test(
        top_k, batch_size, next_n, vocab_size, clean_logits, data_generation
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("clean_logits", [True, False])
@torch.inference_mode()
def test_deepseek_hybrid_topk(clean_logits: bool) -> None:
    torch.set_default_device("cuda:0")

    top_k = 2048

    # Test case 1: Short sequences (< 8192)
    batch_size_short = 4
    next_n = 1
    num_rows_short = batch_size_short * next_n

    # Create sequences with max length < 8192
    seq_lens_short = torch.randint(
        4000, 8000, (batch_size_short,), dtype=torch.int32, device="cuda"
    )

    row_starts_short = torch.zeros(num_rows_short, dtype=torch.int32, device="cuda")
    row_indices_short = torch.arange(num_rows_short, device="cuda") // next_n
    next_n_offset_short = torch.arange(num_rows_short, device="cuda") % next_n
    row_ends_short = (
        seq_lens_short[row_indices_short] - next_n + next_n_offset_short + 1
    )

    logits_short = create_random_logits(
        row_starts_short, row_ends_short, torch.float32, 42, clean_logits, "random"
    )

    indices_vllm = torch.empty(
        (num_rows_short, top_k), dtype=torch.int32, device="cuda"
    )

    # Use vllm's kernel for short sequences
    torch.ops._C.top_k_per_row_decode(
        logits_short,
        next_n,
        seq_lens_short,
        indices_vllm,
        num_rows_short,
        logits_short.stride(0),
        logits_short.stride(1),
        top_k,
    )

    # Test case 2: Long sequences (>= 8192) - should use large_context_topk kernel
    batch_size_long = 4
    num_rows_long = batch_size_long * next_n

    # Create sequences with max length >= 8192
    seq_lens_long = torch.randint(
        8192, 16384, (batch_size_long,), dtype=torch.int32, device="cuda"
    )

    row_starts_long = torch.zeros(num_rows_long, dtype=torch.int32, device="cuda")
    row_indices_long = torch.arange(num_rows_long, device="cuda") // next_n
    next_n_offset_long = torch.arange(num_rows_long, device="cuda") % next_n
    row_ends_long = seq_lens_long[row_indices_long] - next_n + next_n_offset_long + 1

    logits_long = create_random_logits(
        row_starts_long, row_ends_long, torch.float32, 43, clean_logits, "random"
    )

    indices = torch.empty((num_rows_long, top_k), dtype=torch.int32, device="cuda")

    # Use large_context_topk kernel for long sequences
    if next_n == 1:
        lengths = seq_lens_long
    else:
        offsets = torch.arange(next_n, device=logits_long.device, dtype=torch.int32)
        lengths = (seq_lens_long.unsqueeze(1) - next_n + 1 + offsets).flatten()

    torch.ops._C.large_context_topk(
        logits_long,
        indices,
        lengths,
        None,
    )

    torch_indices_short = torch.empty(
        (num_rows_short, top_k), dtype=torch.int32, device="cuda"
    )
    for i in range(num_rows_short):
        row_end = int(row_ends_short[i])
        k_i = min(top_k, row_end)
        idx = logits_short[i, :row_end].topk(k_i, dim=-1)[1]
        torch_indices_short[i, :k_i] = idx

    assert compare_top_k_results(
        logits_short,
        indices_vllm,
        torch_indices_short,
        row_starts_short,
        row_ends_short,
        top_k,
    ), "top_k_per_row_decode kernel (short sequences) doesn't match torch.topk"

    torch_indices_long = torch.empty(
        (num_rows_long, top_k), dtype=torch.int32, device="cuda"
    )
    for i in range(num_rows_long):
        row_end = int(row_ends_long[i])
        k_i = min(top_k, row_end)
        idx = logits_long[i, :row_end].topk(k_i, dim=-1)[1]
        torch_indices_long[i, :k_i] = idx

    assert compare_top_k_results(
        logits_long, indices, torch_indices_long, row_starts_long, row_ends_long, top_k
    ), "large_context_topk kernel (long sequences) doesn't match torch.topk"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("kernel_name", ["radix_topk", "large_context_topk"])
@pytest.mark.parametrize(
    "seq_len_range,test_id",
    [
        pytest.param((4000, 8000), "short_sequences", id="short"),
        pytest.param((32000, 163840), "long_sequences", id="long"),
    ],
)
@pytest.mark.parametrize("clean_logits", [True, False])
@pytest.mark.parametrize("top_k", [2048])
@pytest.mark.parametrize("next_n", [1, 4])
@torch.inference_mode()
def test_deepseek_radix_topk(
    kernel_name: str,
    seq_len_range: tuple[int, int],
    test_id: str,
    clean_logits: bool,
    top_k: int,
    next_n: int,
) -> None:
    """
    Test top-k kernels with varying sequence lengths and speculative decoding.
    Tests both radix_topk and large_context_topk kernels.
    Supports speculative decoding with next_n > 1.
    """
    set_random_seed(42 if test_id == "short_sequences" else 43)
    torch.set_default_device("cuda:0")

    batch_size = 4
    num_rows = batch_size * next_n

    seq_lens = torch.randint(
        seq_len_range[0],
        seq_len_range[1],
        (batch_size,),
        dtype=torch.int32,
        device="cuda",
    )

    # Compute row boundaries for speculative decoding
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_rows, device="cuda") // next_n
    next_n_offset = torch.arange(num_rows, device="cuda") % next_n
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1

    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, clean_logits, "random"
    )

    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    if next_n == 1:
        lengths = seq_lens
    else:
        offsets = torch.arange(next_n, device=logits.device, dtype=torch.int32)
        lengths = (seq_lens.unsqueeze(1) - next_n + 1 + offsets).flatten()

    if kernel_name == "radix_topk":
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
        torch.ops._C.radix_topk(logits, lengths, indices, workspace, top_k)
    elif kernel_name == "large_context_topk":
        torch.ops._C.large_context_topk(logits, indices, lengths, None)
    else:
        raise ValueError(f"Unknown kernel_name: {kernel_name}")

    validate_topk_against_reference(
        logits, indices, row_starts, row_ends, top_k, f"{kernel_name} ({test_id})"
    )


def run_radix_topk_test(
    batch_size: int,
    seq_lens: list[int],
    top_k: int,
    data_type: str = "random",
    seed: int = 42,
    kernel_name: str = "radix_topk",
) -> None:
    """
    Helper to run top-k kernel test with given parameters.

    Args:
        batch_size: Number of rows/sequences
        seq_lens: List of sequence lengths (one per row)
        top_k: Number of top elements to select
        data_type: Type of test data to generate
        seed: Random seed for reproducibility
        kernel_name: Which kernel to test ("radix_topk"
                     or "large_context_topk")
    """
    torch.set_default_device("cuda:0")
    set_random_seed(seed)

    # Create test data
    num_rows = batch_size
    max_len = max(seq_lens)
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    if data_type == "random":
        logits = torch.randn(num_rows, max_len, dtype=torch.float32, device="cuda")
    elif data_type == "sorted_asc":
        # Each row gets its own ascending sequence based on its length
        logits = torch.empty(num_rows, max_len, dtype=torch.float32, device="cuda")
        for i, length in enumerate(seq_lens):
            logits[i, :length] = torch.arange(
                length, dtype=torch.float32, device="cuda"
            )
            if length < max_len:
                logits[i, length:] = float("-inf")
    elif data_type == "sorted_desc":
        # Each row gets its own descending sequence based on its length
        logits = torch.empty(num_rows, max_len, dtype=torch.float32, device="cuda")
        for i, length in enumerate(seq_lens):
            logits[i, :length] = torch.arange(
                length, 0, -1, dtype=torch.float32, device="cuda"
            )
            if length < max_len:
                logits[i, length:] = float("-inf")
    elif data_type == "all_same":
        logits = torch.ones(num_rows, max_len, dtype=torch.float32, device="cuda")
        for i, length in enumerate(seq_lens):
            if length < max_len:
                logits[i, length:] = float("-inf")
    elif data_type == "many_ties":
        # Only 10 unique values, many duplicates
        logits = torch.randint(0, 10, (num_rows, max_len), device="cuda").float() / 10.0
        for i, length in enumerate(seq_lens):
            if length < max_len:
                logits[i, length:] = float("-inf")
    elif data_type == "small_differences":
        # Very small differences to test float precision
        base = torch.randn(num_rows, max_len, dtype=torch.float32, device="cuda")
        noise = (
            torch.randn(num_rows, max_len, dtype=torch.float32, device="cuda") * 1e-6
        )
        logits = base + noise
        for i, length in enumerate(seq_lens):
            if length < max_len:
                logits[i, length:] = float("-inf")
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    # Create output tensor
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    if kernel_name == "radix_topk":
        workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
        torch.ops._C.radix_topk(logits, lengths, indices, workspace, top_k)
    elif kernel_name == "large_context_topk":
        torch.ops._C.large_context_topk(logits, indices, lengths, None)
    else:
        raise ValueError(f"Unknown kernel_name: {kernel_name}")

    torch.cuda.synchronize()

    torch_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    for i in range(num_rows):
        length = seq_lens[i]
        k_i = min(top_k, length)
        if k_i > 0:
            idx = logits[i, :length].topk(k_i, dim=-1)[1]
            torch_indices[i, :k_i] = idx
            if k_i < top_k:
                torch_indices[i, k_i:] = -1
        else:
            torch_indices[i, :] = -1

    # Compare results
    for i in range(num_rows):
        length = seq_lens[i]
        k_i = min(top_k, length)

        if k_i == 0:
            continue

        cuda_row = indices[i, :k_i].cpu()
        torch_row = torch_indices[i, :k_i].cpu()

        # Filter out -1 padding values from cuda_row
        valid_mask = cuda_row >= 0
        cuda_row = cuda_row[valid_mask]

        # Compare sets (order may differ for ties)
        cuda_set = set(cuda_row.tolist())
        torch_set = set(torch_row.tolist())

        if cuda_set == torch_set:
            continue

        # If sets differ, check if it's due to equal values (ties)
        cuda_vals = logits[i, cuda_row].cpu()
        torch_vals = logits[i, torch_row].cpu()

        # Check that min CUDA value >= max of values NOT in top-k
        if k_i < length:
            non_topk_indices = torch.tensor(
                list(set(range(length)) - cuda_set), dtype=torch.int32
            )
            if len(non_topk_indices) > 0:
                non_topk_vals = logits[i, non_topk_indices].cpu()
                min_cuda_val = cuda_vals.min()
                max_non_topk = non_topk_vals.max()

                # Allow small tolerance for floating point errors
                assert min_cuda_val >= max_non_topk - 1e-4, (
                    f"Row {i}: CUDA top-k contains values smaller than non-top-k. "
                    f"Min CUDA: {min_cuda_val}, Max non-top-k: {max_non_topk}, "
                    f"Length: {length}, k: {k_i}, CUDA indices: {sorted(cuda_set)[:10]}..., "  # noqa: E501
                    f"Expected indices: {sorted(torch_set)[:10]}..."
                )

        # For ties, verify the values are close
        assert torch.allclose(
            cuda_vals.sort(descending=True)[0],
            torch_vals.sort(descending=True)[0],
            rtol=1e-4,
            atol=1e-4,
        ), f"""Row {i}: Top-k values don't match.
            CUDA: {cuda_vals.sort(descending=True)[0][:10]},
            Torch: {torch_vals.sort(descending=True)[0][:10]}"""


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("kernel_name", ["radix_topk", "large_context_topk"])
@pytest.mark.parametrize(
    "test_config",
    [
        # ==================== CATEGORY: Sequence Length Edge Cases ====================
        pytest.param(
            {"seq_lens": [1, 10, 100, 2048], "top_k": 2048, "data_type": "random"},
            id="seq_len_edge_very_small_to_medium",
        ),
        pytest.param(
            {
                "seq_lens": [2049, 2100, 2500, 3000],
                "top_k": 2048,
                "data_type": "random",
            },
            id="seq_len_edge_above_k",
        ),
        pytest.param(
            {"seq_lens": [8000, 16384, 20000], "top_k": 2048, "data_type": "random"},
            id="algo_transition_filtered_radix",
        ),
        # ==================== CATEGORY: Data Distributions ====================
        pytest.param(
            {"seq_lens": [5000, 10000], "top_k": 2048, "data_type": "sorted_asc"},
            id="data_sorted_ascending",
        ),
        pytest.param(
            {"seq_lens": [5000, 10000], "top_k": 2048, "data_type": "sorted_desc"},
            id="data_sorted_descending",
        ),
        pytest.param(
            {"seq_lens": [5000, 10000], "top_k": 2048, "data_type": "all_same"},
            id="data_all_same",
        ),
        pytest.param(
            {"seq_lens": [5000, 10000], "top_k": 2048, "data_type": "many_ties"},
            id="data_many_ties",
        ),
        pytest.param(
            {
                "seq_lens": [5000, 10000],
                "top_k": 2048,
                "data_type": "small_differences",
            },
            id="data_float_precision",
        ),
        # ==================== CATEGORY: Alignment / Vectorization ====================
        pytest.param(
            {
                "seq_lens": [2055, 2056, 2057, 2063],
                "top_k": 2048,
                "data_type": "random",
            },
            id="align_vec_boundaries_low",
        ),
        pytest.param(
            {
                "seq_lens": [4095, 4096, 4097, 4102],
                "top_k": 2048,
                "data_type": "random",
            },
            id="align_4k_boundary",
        ),
        pytest.param(
            {
                "seq_lens": [8191, 8192, 8193, 8198],
                "top_k": 2048,
                "data_type": "random",
            },
            id="align_8k_boundary",
        ),
        pytest.param(
            {
                "seq_lens": [16383, 16384, 16385, 16390],
                "top_k": 2048,
                "data_type": "random",
            },
            id="align_16k_boundary",
        ),
    ],
)
@torch.inference_mode()
def test_radix_topk_correctness(kernel_name: str, test_config: dict) -> None:
    """
    Comprehensive correctness tests covering:
    - Sequence length edge cases (trivial, boundary, varied)
    - Small top_k values (1, 2, 10, 256)
    - Very small sequences (< 100 elements)
    - Mixed sequence lengths in same batch
    - Data distributions (sorted, ties, precision)
    - Memory alignment / vectorization boundaries

    Tests both radix_topk and large_context_topk kernels.
    """
    run_radix_topk_test(
        batch_size=len(test_config["seq_lens"]),
        seq_lens=test_config["seq_lens"],
        top_k=test_config["top_k"],
        data_type=test_config.get("data_type", "random"),
        kernel_name=kernel_name,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("kernel_name", ["radix_topk", "large_context_topk"])
@pytest.mark.parametrize(
    "test_config",
    [
        # ==================== CATEGORY: Batch Size Scalability ====================
        pytest.param(
            {"batch_size": 1, "seq_len": 5000, "top_k": 2048},
            id="batch_1",
        ),
        pytest.param(
            {"batch_size": 4, "seq_len": 5000, "top_k": 2048},
            id="batch_4",
        ),
        pytest.param(
            {"batch_size": 32, "seq_len": 5000, "top_k": 2048},
            id="batch_32",
        ),
        pytest.param(
            {"batch_size": 256, "seq_len": 5000, "top_k": 2048},
            id="batch_256",
        ),
        # ==================== CATEGORY: Single-CTA vs Multi-CTA ====================
        pytest.param(
            {"batch_size": 2, "seq_len": 4096, "top_k": 2048},
            id="single_cta_4k",
        ),
        pytest.param(
            {"batch_size": 2, "seq_len": 8192, "top_k": 2048},
            id="single_cta_8k",
        ),
        pytest.param(
            {"batch_size": 2, "seq_len": 163840, "top_k": 2048},
            id="multi_cta_163840_dsv3_max",
        ),
        # ==================== CATEGORY: Extreme Cases ====================
        pytest.param(
            {"batch_size": 512, "seq_len": 5000, "top_k": 2048},
            id="extreme_large_batch",
        ),
        pytest.param(
            {"batch_size": 2, "seq_len": 163840, "top_k": 2048},
            id="extreme_dsv3_max_context",
        ),
    ],
)
@torch.inference_mode()
def test_radix_topk_algorithm_paths(kernel_name: str, test_config: dict) -> None:
    """
    Test different algorithm execution paths (capped at 163840 for DeepSeek V3.2):
    - Batch size scalability (1, 4, 32, 256, 1024)
    - FilteredTopK vs RadixTopK selection (8K to 163K)
    - Single-CTA vs Multi-CTA execution
    - Extreme configurations (large batch, max context length)

    Tests both radix_topk and large_context_topk kernels.
    """
    run_radix_topk_test(
        batch_size=test_config["batch_size"],
        seq_lens=[test_config["seq_len"]] * test_config["batch_size"],
        top_k=test_config["top_k"],
        kernel_name=kernel_name,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("kernel_name", ["radix_topk", "large_context_topk"])
@torch.inference_mode()
def test_radix_topk_stress(kernel_name: str) -> None:
    """
    Stress test with random configurations to catch edge cases.
    Capped at 163840 (DeepSeek V3.2 max context) for realistic testing.

    Tests both radix_topk and large_context_topk kernels.
    """
    torch.set_default_device("cuda:0")
    top_k = 2048

    for seed in range(3):
        set_random_seed(seed)

        # Random batch size (limited for speed)
        batch_size = torch.randint(1, 32, (1,)).item()

        # Random sequence lengths capped at DeepSeek V3.2 max context
        seq_lens = torch.randint(100, 163840, (batch_size,)).tolist()

        run_radix_topk_test(
            batch_size=batch_size,
            seq_lens=seq_lens,
            top_k=top_k,
            seed=seed,
            kernel_name=kernel_name,
        )
