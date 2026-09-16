# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Oracle tests for the HiSparse dense->ragged top-k adapter on ROCm.

HiSparse's residency resolver returns a dense ``[tokens, topk]`` tensor padded
with ``-1``; the AITER sparse decode kernel consumes a ragged index list plus a
``paged_kv_indptr``. ``compact_topk_to_ragged_triton`` bridges the two. It is
the one piece of genuinely new logic in the ROCm HiSparse wiring and a silent
a wrong flattening reads the wrong KV rows without crashing, so it gets its own
reference comparison.
"""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "ROCm AITER sparse MLA ragged adapter test requires ROCm.",
        allow_module_level=True,
    )

from vllm._aiter_ops import is_aiter_found_and_supported

if not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm AITER sparse MLA ragged adapter test requires a supported AITER "
        "installation.",
        allow_module_level=True,
    )

from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    compact_topk_to_ragged_triton,
)

DEVICE = current_platform.device_type


def _reference_ragged(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a ``-1``-padded dense top-k the way the kernel should."""
    counts = (dense >= 0).sum(axis=1)
    indptr = np.zeros(dense.shape[0] + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    flat = np.concatenate([row[row >= 0] for row in dense]) if dense.size else dense
    return indptr, flat.astype(np.int32)


def _run(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dense_gpu = torch.from_numpy(dense).to(DEVICE)
    expected_indptr, expected_flat = _reference_ragged(dense)
    indptr = torch.from_numpy(expected_indptr).to(DEVICE)
    # Poison the destination so any row the kernel fails to write is visible.
    out = torch.full(
        (max(int(expected_indptr[-1]), 1),), -7, dtype=torch.int32, device=DEVICE
    )
    compact_topk_to_ragged_triton(dense_gpu, indptr, out)
    return out[: int(expected_indptr[-1])].cpu().numpy(), expected_flat


@pytest.mark.parametrize("width", [1, 4, 16, 64, 65, 128])
def test_compact_ragged_full_rows(width):
    """Fully valid rows flatten to a contiguous copy at every top-k width."""
    dense = np.arange(3 * width, dtype=np.int32).reshape(3, width)
    actual, expected = _run(dense)
    np.testing.assert_array_equal(actual, expected)


def test_compact_ragged_drops_interior_padding():
    """A ``-1`` in the middle of a row must drop out, not displace a valid id."""
    dense = np.array(
        [
            [10, -1, 11, -1, 12, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [20, 21, 22, 23, 24, 25, 26, 27],
            [-1, 30, -1, -1, -1, -1, -1, 31],
        ],
        dtype=np.int32,
    )
    actual, expected = _run(dense)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        actual, [10, 11, 12, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31]
    )


@pytest.mark.parametrize("width", [8, 33, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compact_ragged_random_rows(width, seed):
    """Randomly ragged rows of differing length match the NumPy reference."""
    rng = np.random.default_rng(seed)
    num_tokens = 17
    dense = rng.integers(0, 4096, size=(num_tokens, width)).astype(np.int32)
    keep = rng.integers(0, width + 1, size=num_tokens)
    for token, num_valid in enumerate(keep):
        dense[token, num_valid:] = -1
        # Shuffle so the valid entries are not always a leading prefix.
        rng.shuffle(dense[token])
    actual, expected = _run(dense)
    np.testing.assert_array_equal(actual, expected)


def test_compact_ragged_leaves_tail_untouched():
    """The kernel writes exactly indptr[-1] entries and nothing beyond."""
    dense = np.array([[5, -1, 6, -1]], dtype=np.int32)
    dense_gpu = torch.from_numpy(dense).to(DEVICE)
    indptr = torch.tensor([0, 2], dtype=torch.int32, device=DEVICE)
    out = torch.full((8,), -7, dtype=torch.int32, device=DEVICE)
    compact_topk_to_ragged_triton(dense_gpu, indptr, out)
    np.testing.assert_array_equal(out.cpu().numpy(), [5, 6, -7, -7, -7, -7, -7, -7])
