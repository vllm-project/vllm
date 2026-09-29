# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 row-offset math in Model Runner V2 kernels (issue #57030).

Kernels that index a 2-D tensor as `row_idx * stride + col` must promote the
row index to int64 before the multiply. With int32 indices (e.g. row ids
loaded from int32 staging buffers or explicit-dtype h2d mappings) the product
wraps once `row_idx * stride >= 2**31`, producing a negative (or re-wrapped
positive) offset: an out-of-bounds access or a silent write to the wrong row.

Realistic trigger: `RequestState.all_token_ids` is (max_num_reqs,
max_model_len) int32; at 1M context any row index >= 2148 overflows, so the
first request faults on engines with max_num_reqs >= 2149.

Each case below allocates a >2**31-element (>= 8 GiB) int32 tensor and is
skipped when the device does not have enough free memory.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.sample.penalties import bincount

# (max_model_len, row_under_test): row * max_model_len > 2**31.
STRIDE = 1_000_000
ROW = 2559
NUM_ROWS = ROW + 1  # (2560, 1M) int32 = 10 GiB
NEEDED_BYTES = NUM_ROWS * STRIDE * 4


def _has_memory() -> bool:
    if not current_platform.is_cuda_alike():
        return False
    free, _ = torch.accelerator.get_memory_info(0)
    # Slack for the auxiliary buffers the kernels allocate.
    return free >= NEEDED_BYTES + 2**30


pytestmark = pytest.mark.skipif(
    not _has_memory(),
    reason=f"requires >= {NEEDED_BYTES / 2**30:.0f} GiB free device memory",
)


def test_staged_write_high_row():
    """`_apply_write_kernel` must write row 2559 of a 1M-column tensor.

    int32 `row_idx * row_stride` wraps to a negative offset here, faulting
    (or landing on a wrapped row) without the int64 promotion.
    """
    device = torch.device("cuda:0")
    staged = StagedWriteTensor((NUM_ROWS, STRIDE), torch.int32, device)
    content = list(range(9000, 9016))
    staged.stage_write(ROW, 0, content)
    staged.apply_write()
    torch.accelerator.synchronize()

    assert staged.gpu[ROW, : len(content)].tolist() == content
    # A wrapped offset would also clobber row 0 (2559*1e6 - 2**32 = -1.74e9
    # wraps inside-out; the re-wrapped positive case lands at row 0 col 32704
    # for other shapes). Row 0 must stay untouched.
    assert staged.gpu[0, : len(content)].tolist() == [0] * len(content)
    del staged


def test_bincount_high_row():
    """`_bincount_kernel` reads all_token_ids at `req_state_idx * stride`.

    The penalties path builds `idx_mapping` with an explicit int32 dtype, so
    the row index must be promoted in-kernel.
    """
    device = torch.device("cuda:0")
    vocab = 8192
    prompt_len, prefill_len = 1024, 2048

    all_token_ids = torch.zeros(NUM_ROWS, STRIDE, dtype=torch.int32, device=device)
    all_token_ids[ROW, :prefill_len] = torch.arange(
        prefill_len, dtype=torch.int32, device=device
    )
    req_lens = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    req_lens[ROW] = prompt_len
    prefill_lens = torch.zeros(NUM_ROWS, dtype=torch.int32, device=device)
    prefill_lens[ROW] = prefill_len
    idx_mapping = torch.tensor([ROW], dtype=torch.int32, device=device)
    prompt_bin_mask = torch.zeros(
        NUM_ROWS, cdiv(vocab, 32), dtype=torch.int32, device=device
    )
    output_bin_counts = torch.zeros(NUM_ROWS, vocab, dtype=torch.int32, device=device)

    bincount(
        idx_mapping,
        all_token_ids,
        req_lens,
        prefill_lens,
        prompt_bin_mask,
        output_bin_counts,
        max_prefill_len=prefill_len,
    )
    torch.accelerator.synchronize()

    # Prompt tokens [0, prompt_len) set one bit each in the request's row.
    words = prompt_bin_mask[ROW, : cdiv(prompt_len, 32)]
    bits = torch.stack([(words >> i) & 1 for i in range(32)], 1).reshape(-1)
    assert bool((bits[:prompt_len] == 1).all().item())
    # Output tokens [prompt_len, prefill_len) counted once each.
    assert bool((output_bin_counts[ROW, prompt_len:prefill_len] == 1).all().item())
    assert output_bin_counts[ROW, :prompt_len].sum().item() == 0
    assert output_bin_counts[ROW, prefill_len:].sum().item() == 0
