# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


class StructuredOutputsWorker:
    def __init__(self, device: torch.device):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        # Copy the bitmask.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(grammar_bitmask, device=self.device)

        # Construct bitmask -> logits mapping
        mapping: list[int] = []
        req_ids = input_batch.req_ids
        cu_num_logits = input_batch.cu_num_logits_np.tolist()
        req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
        for grammar_req_id in grammar_req_ids:
            req_idx = req_id_to_idx[grammar_req_id]
            logits_start_idx = cu_num_logits[req_idx]
            logits_end_idx = cu_num_logits[req_idx + 1]
            mapping.extend(range(logits_start_idx, logits_end_idx))

        # Copy the mapping.
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=True
            )
            logits_indices = logits_indices.to(device=self.device, non_blocking=True)

        # Make sure the kernel is executed after the copy is finished.
        torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

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
