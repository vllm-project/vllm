# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the xgrammar bitmask kernel, which the V1 dummy sampler run never
reaches. V2 warmup applies a real grammar bitmask instead."""

from typing import TYPE_CHECKING

import torch

from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

# Row counts covering Triton's ==1 / generic / multiple-of-16 int classes.
_ROW_CLASS_REPS = (1, 2, 16)


@torch.inference_mode()
def grammar_bitmask_warmup(logits: torch.Tensor, max_num_rows: int) -> None:
    """Compile the bitmask kernel for logits shaped like ``logits``.

    Args:
        logits: Logits as produced by ``compute_logits`` (dtype and vocab
            width must match serving).
        max_num_rows: Max logits rows sampled in one step, including
            speculative tokens.

    """
    if not logits.is_cuda:
        return
    vocab_size = logits.shape[-1]
    bitmask_width = (vocab_size + 31) // 32
    for num_rows in _ROW_CLASS_REPS:
        if num_rows > max_num_rows:
            continue
        rows = logits[:1].expand(num_rows, vocab_size).clone()
        bitmask = torch.full(
            (num_rows, bitmask_width), -1, dtype=torch.int32, device=logits.device
        )
        indices = torch.arange(num_rows, dtype=torch.int32, device=logits.device)
        # `apply_grammar_bitmask` drops the indices when every row is
        # structured, which compiles a separate specialization.
        xgr.apply_token_bitmask_inplace(rows, bitmask, indices=None)
        xgr.apply_token_bitmask_inplace(
            rows,
            bitmask,
            indices=indices,  # type: ignore[arg-type]
        )
