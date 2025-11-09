# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton


def apply_grammar_bitmask(
    logits: torch.Tensor,
    req_ids: list[str],
    grammar_req_ids: list[str],
    grammar_bitmask: np.ndarray,
) -> None:
    grammar_req_id_to_idx = {req_id: i for i, req_id in enumerate(grammar_req_ids)}
    # logits -> bitmask mapping
    mapping = [grammar_req_id_to_idx.get(req_id, -1) for req_id in req_ids]
    # TODO(woosuk): Make the below non-blocking.
    mapping_tensor = torch.tensor(mapping, dtype=torch.int32, device=logits.device)
    assert grammar_bitmask.dtype == np.int32
    grammar_bitmask = torch.from_numpy(grammar_bitmask).to(logits.device)

    vocab_size = logits.shape[-1]
    BLOCK_SIZE = 8192
    grid = (logits.shape[0], tl.cdiv(vocab_size, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        grammar_bitmask,
        grammar_bitmask.stride(0),
        mapping_tensor,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    bitmask_ptr,
    bitmask_stride,
    mapping_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    logits_idx = tl.program_id(0)
    bitmask_idx = tl.load(mapping_ptr + logits_idx)
    if bitmask_idx == -1:
        # No bitmask to apply.
        return

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )
