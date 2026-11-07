# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless CSF encoding of MXFP4 (E8M0) expert block scales for b12x.

Each scale matrix is stored as a compact fixed plane plus a short list of
exception words, which b12x consumes as resident compressed expert scales.
"""

import numpy as np
import torch

# MXFP4 (E8M0) scales: b12x consumes canonical planes per expert. Each 16-row
# slab holds one base byte per row, then every row's one-bit selectors
# (``ceil(cols / 8)`` bytes per row, low bit first); a set bit selects
# ``base + 1`` over ``base``. Exceptions are sorted uint32 words: low 24 bits a
# flat ``row * cols + col`` position, high 8 bits the exact scale byte.
MXFP4_CSF_ROWS = 16


def encode_mxfp4_csf_scales(
    scales: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Encode ``[experts, rows, cols]`` E8M0 block scales as b12x MXFP4 planes.

    A row's base is the lowest byte whose two-value window ``{base, base + 1}``
    holds the most of the row's bytes, which minimizes exceptions. Returns each
    expert's fixed plane (uint8) and exception words (uint32) on the CPU.
    """
    if scales.dim() != 3 or scales.element_size() != 1:
        raise ValueError(
            f"Expected [experts, rows, cols] byte scales, got {scales.shape}"
        )
    experts, rows, cols = scales.shape
    if (
        experts == 0
        or rows == 0
        or rows % MXFP4_CSF_ROWS
        or not 1 <= cols <= 255
        or rows * cols > 1 << 24
    ):
        raise ValueError(
            "MXFP4 CSF needs nonempty experts, positive rows divisible by 16, "
            "1..255 columns and at most "
            f"2**24 scales per matrix, got {rows}x{cols}"
        )
    raw = scales.contiguous().view(torch.uint8).cpu().numpy()
    row_bytes = -(-cols // 8)
    fixed: list[torch.Tensor] = []
    exceptions: list[torch.Tensor] = []
    for expert in range(experts):
        values = raw[expert]
        keys = (np.arange(rows, dtype=np.int64)[:, None] << 8) | values
        counts = np.bincount(keys.reshape(-1), minlength=rows << 8).reshape(rows, 256)
        bases = (counts[:, :255] + counts[:, 1:]).argmax(1).astype(np.uint8)
        offsets = values.astype(np.int16) - bases[:, None]
        selected = offsets == 1
        outliers = (offsets != 0) & ~selected
        bits = np.packbits(selected, axis=1, bitorder="little")
        assert bits.shape[1] == row_bytes
        slabs = np.concatenate(
            (
                bases.reshape(-1, MXFP4_CSF_ROWS),
                bits.reshape(-1, MXFP4_CSF_ROWS * row_bytes),
            ),
            axis=1,
        )
        fixed.append(torch.from_numpy(np.ascontiguousarray(slabs)))
        positions = np.flatnonzero(outliers.reshape(-1)).astype(np.uint32)
        words = (values.reshape(-1)[positions].astype(np.uint32) << 24) | positions
        exceptions.append(torch.from_numpy(words.astype(np.uint32)))
    return tuple(fixed), tuple(exceptions)
