# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft's base logits on gfx942.

With one request, the draft's base logits are a GEMM of 5 rows, one for each
draft step, over the draft's lm_head shard (32320 x 5120 at TP 4). vLLM's
dispatch (rocm_unquantized_gemm) sends a GEMM of up to 5 rows to wvSplitK. On
MI325X, wvSplitK takes 295 us for these 5 rows, which reads the shard at
1.1 TB/s, against 71 us for 4 rows, and the torch GEMM takes 91 us for 5 or 6
rows. So here the 5 rows get a sixth row of zeros, which makes vLLM's dispatch
take the torch GEMM, and the logits of the sixth row are dropped. That takes
96 us. Every other GEMM keeps vLLM's dispatch.

It runs on gfx942 with --specialized-decode (``ops.enabled()``).
tests/kernels/models/amd/test_dsv41_gfx942_draft_lm_head.py checks it.
"""

import torch

from vllm.models.deepseek_v41.amd.ops import enabled

# The draft's rows with one request and 5 draft steps. It is also the most
# rows that vLLM's dispatch sends to wvSplitK.
ROWS = 5


def compute_logits(logits_processor, lm_head, normed: torch.Tensor):
    """The base logits ``logits_processor(lm_head, normed)``. On gfx942 with
    --specialized-decode, 5 rows run as 6 rows with a row of zeros and come back as
    5. The processor returns None on a rank that does not get the gathered
    logits, and so does this function then."""
    if normed.shape[0] != ROWS or not enabled():
        return logits_processor(lm_head, normed)
    # One row of zeros after the last row.
    padded = torch.nn.functional.pad(normed, (0, 0, 0, 1))
    logits = logits_processor(lm_head, padded)
    return None if logits is None else logits[:ROWS]
