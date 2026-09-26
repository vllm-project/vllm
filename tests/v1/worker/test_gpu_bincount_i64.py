# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in the penalty bincount kernel.

`_bincount_kernel` reads `all_token_ids` at `req_state_idx * all_token_ids_stride`.
That stride is `max_model_len`, and `req_state_idx` runs up to `max_num_reqs - 1`.
Once `max_num_reqs * max_model_len` passes 2**31 - 1, an int32 product wraps
negative and the kernel reads far before the buffer.

The test places one request at the last row of a [65536, 32769] int32 buffer.
That row starts at element 2147516415, past 2**31 - 1 (2147483647). The buffer
cannot be smaller: to keep the row inside the buffer, the offset has to leave
int32 at the same moment. This case allocates ~8.6 GiB of GPU memory.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.penalties import bincount

ROWS = 65536
COLS = 32769  # ROWS * COLS > 2**31
VOCAB_SIZE = 1024
BLOCK_SIZE = 1024
TOKEN = 7
FILLER_TOKEN = 3


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_bincount_row_offset_beyond_int32():
    device = torch.device("cuda:0")

    # Every row holds the filler token. The last row holds the token under test,
    # so a read from the wrong row shows up as a wrong count.
    all_token_ids = torch.zeros((ROWS, COLS), dtype=torch.int32, device=device)
    all_token_ids[:, :BLOCK_SIZE] = FILLER_TOKEN
    all_token_ids[ROWS - 1, :BLOCK_SIZE] = TOKEN

    # One request, placed at the highest row.
    idx_mapping = torch.tensor([ROWS - 1], dtype=torch.int32, device=device)
    prompt_len = torch.zeros(ROWS, dtype=torch.int32, device=device)
    prefill_len = torch.zeros(ROWS, dtype=torch.int32, device=device)
    prefill_len[ROWS - 1] = BLOCK_SIZE
    prompt_bin_mask = torch.zeros(
        (ROWS, (VOCAB_SIZE + 31) // 32), dtype=torch.int32, device=device
    )
    output_bin_counts = torch.zeros(
        (ROWS, VOCAB_SIZE), dtype=torch.int32, device=device
    )

    bincount(
        idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        BLOCK_SIZE,
    )
    torch.accelerator.synchronize()

    assert output_bin_counts[ROWS - 1, TOKEN].item() == BLOCK_SIZE
    assert output_bin_counts[ROWS - 1, FILLER_TOKEN].item() == 0
