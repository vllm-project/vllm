# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The kernel is adapted from SGLang (Apache-2.0):
# https://github.com/sgl-project/sglang/blob/656331295619a4d0267b40b7da0b5cd5b549fcbc/python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh  # noqa: E501
"""The candidate consumers' top-k and its remap with the ``topk_bf16_small``
kernel (``sparse_topk_small.cu``).

A candidate consumer scores only its candidate blocks (DeepGEMM's paged sparse
MQA logits): row ``r`` of the BF16 sparse logits scores, in column
``j < ends[r]``, the token at request-local position
``blocks[r, j >> s] << s | (j & (2^s - 1))``, where ``blocks`` holds the row's
sparse block ids of ``2^s`` tokens each. One CTA of 512 threads a row
holds the row in registers, finds the exact k-th largest value with two radix
histograms of the raw BF16 bytes and writes the selected columns through the
block table in ascending order: every column above the k-th value, then the
lowest-numbered ones equal to it. A row of at most ``k`` columns gets all of
them, and -1 pads the rest of the ``k`` slots (also one slot per positive NaN,
which is never selected). It waits on its predecessor grid before reading the
logits and lets its dependents launch once its picks are staged.

Torch builds the op into its extension cache at warm-up (``build``) for each
device's architecture; the TP ranks share one build.
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("sparse_topk_small.cu")

MAX_COLUMNS = 16384
MAX_K = 2048


@cache
def build(device_index: int) -> Callable[..., None]:
    """Build (or load the cached build of) the op for the device's
    architecture."""
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability(device_index)
    arch = f"{major}{minor}a"
    load(
        name=f"sd_sparse_topk_small_sm{arch}",
        sources=[str(_SOURCE)],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
        is_python_module=False,
    )
    return torch.ops.sd_sparse_topk_small.topk


def supports_sparse_topk_small(width: int, k: int, block_size: int) -> bool:
    """Whether `sparse_topk_small` takes rows of ``width`` logits for ``k`` and
    ``block_size``-token sparse blocks."""
    return (
        0 < width <= MAX_COLUMNS
        and width % 8 == 0
        and 0 < k <= MAX_K
        and block_size & (block_size - 1) == 0
    )


def sparse_topk_small(
    logits: torch.Tensor,
    ends: torch.Tensor,
    blocks: torch.Tensor,
    block_size: int,
    out: torch.Tensor,
) -> None:
    """Launch ``topk_bf16_small`` on the current stream.

    Args:
        logits: [rows, width] BF16 sparse logits (16-byte aligned rows), row
            ``r`` valid below ``ends[r]`` (see `supports_sparse_topk_small`).
        ends: Contiguous [rows] int32 valid column counts.
        blocks: [rows, width // block_size] int32 sparse block ids (unit
            column stride), the request-local blocks backing the columns.
        block_size: Tokens a sparse block.
        out: [rows, k] int32 (unit column stride) for the request-local
            positions.

    """
    rows, width = logits.shape
    assert supports_sparse_topk_small(width, out.shape[1], block_size)
    assert logits.dtype == torch.bfloat16 and logits.stride(1) == 1
    assert logits.stride(0) % 8 == 0 and logits.data_ptr() % 16 == 0
    assert ends.dtype == torch.int32 and ends.is_contiguous() and ends.numel() == rows
    assert blocks.dtype == torch.int32 and blocks.stride(1) == 1
    assert blocks.shape[0] >= rows and blocks.shape[1] * block_size >= width
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape[0] == rows
    build(logits.device.index)(logits, ends, blocks, out, block_size.bit_length() - 1)
