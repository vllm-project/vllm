# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the shared DSA candidate mask (CUDA path).

``apply_candidate_mask`` poisons each row's ``[start, end)`` extent only; the
columns outside it (padding plus other requests' packed column ranges) are
left untouched instead of being sanitized to ``-inf``. This is the same
contract the ROCm strided variant already relies on (see
``test_rocm_aiter_candidate_mask.py``): every consumer of the masked logits
bounds its scan by the same row extents, ``top_k_per_row_prefill`` takes
``rowStart/rowEnd`` from the cu_seqlen tensors. So the tests assert agreement
with a full-width poisoned reference over ``[0, end)`` and that the workspace
outside each row's extent is left untouched.
"""

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-specific tests"
)

BLOCK_SIZE = 64


def _reference_mask(
    logits: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    candidates: torch.Tensor,
    block: int,
    row_repeat: int,
) -> torch.Tensor:
    """Full-width torch reference of the original poisoning mask."""
    out = logits.clone()
    rows, width = logits.shape
    nblocks = (width + block - 1) // block
    for r in range(rows):
        b = r // row_repeat
        s, e = int(starts[b]), min(int(ends[b]), width)
        cand = [c for c in candidates[r].tolist() if c >= 0]
        edge = any(s + c * block >= width for c in cand)
        keep_blocks = {c for c in cand if s + c * block < width}
        cols = torch.arange(s, e, device=logits.device)
        blocks = torch.div(cols - s, block, rounding_mode="floor")
        keep = torch.isin(blocks, torch.tensor(list(keep_blocks) or [-1],
                                                device=logits.device))
        out[r, cols] = torch.where(keep, out[r, cols], float("-inf"))
        if edge and width - 1 >= s and width - 1 < e:
            out[r, width - 1] = logits[r, width - 1]
    return out


def _make_case(rows, width, row_repeat, seed, dead_tail=True):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, width, dtype=torch.float32,
                         generator=g).cuda()
    groups = rows // row_repeat
    if dead_tail:
        # Packed layout: each group owns a short span of a wide workspace.
        lens = torch.randint(1, max(2, width // 4), (groups,), generator=g)
        offs = torch.randint(0, max(1, width // 4), (groups,), generator=g)
    else:
        lens = torch.full((groups,), width)
        offs = torch.zeros(groups, dtype=torch.long)
    starts = offs.to(torch.int32).cuda()
    ends = (offs + lens).clamp(max=width).to(torch.int32).cuda()
    nblocks = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    candidates = torch.randint(
        0, nblocks, (rows, 8), generator=g, dtype=torch.int32)
    # Sprinkle -1 padding and an occasional clamped (edge) candidate.
    drop = torch.rand(rows, 8, generator=g) < 0.3
    edge_rows = torch.rand(rows, generator=g) < 0.25
    extra = torch.randint(0, 4, (rows,), generator=g).to(torch.int32)
    candidates = candidates.masked_fill(drop, -1)
    candidates[:, 0] = torch.where(edge_rows, nblocks - 1 + extra,
                                   candidates[:, 0])
    return logits, starts, ends, candidates.cuda()


@pytest.mark.parametrize("rows,width,row_repeat", [
    (1, 4096, 1),
    (7, 16384, 1),
    (126, 16384, 6),
    (66, 262144, 6),
])
@pytest.mark.parametrize("seed", [0, 1])
def test_mask_agrees_within_extent(rows, width, row_repeat, seed):
    logits, starts, ends, candidates = _make_case(rows, width, row_repeat,
                                                  seed)
    ref = _reference_mask(logits, starts, ends, candidates, BLOCK_SIZE,
                          row_repeat)
    out = logits.clone()
    apply_candidate_mask(out, starts, ends, candidates, BLOCK_SIZE,
                         row_repeat)
    for r in range(rows):
        s, e = int(starts[r // row_repeat]), min(int(ends[r // row_repeat]),
                                                 width)
        assert torch.equal(out[r, s:e], ref[r, s:e]), f"row {r} extent differs"
        # Beyond the extent the kernel must leave the workspace untouched.
        assert torch.equal(out[r, e:], logits[r, e:]), f"row {r} tail touched"
        if s > 0:
            assert torch.equal(out[r, :s], logits[r, :s]), (
                f"row {r} head touched")
