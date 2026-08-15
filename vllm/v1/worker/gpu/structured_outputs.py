# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


class StructuredOutputsWorker:
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        mask_stride: int,
    ):
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = torch.cuda.Stream()
        self.mask_stride = mask_stride

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        num_grammar_reqs = len(grammar_req_ids)
        assert grammar_bitmask.shape[0] == num_grammar_reqs * self.mask_stride

        # Asynchronously copy the bitmask to GPU.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        req_id_to_idx = {
            req_id: req_idx for req_idx, req_id in enumerate(input_batch.req_ids)
        }
        req_indices = [req_id_to_idx[req_id] for req_id in grammar_req_ids]

        # Asynchronously copy the request indices to GPU.
        with torch.cuda.stream(self.copy_stream):
            req_indices_tensor = torch.tensor(
                req_indices, dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
            )
            req_indices_tensor = self.logits_indices[:num_grammar_reqs].copy_(
                req_indices_tensor, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_grammar_reqs, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            req_indices_tensor,
            input_batch.cu_num_logits,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            MASK_STRIDE=self.mask_stride,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    req_indices_ptr,
    cu_num_logits_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    MASK_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    grammar_idx = tl.program_id(0)
    req_idx = tl.load(req_indices_ptr + grammar_idx)
    logits_start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_req_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_start_idx

    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for position_idx in range(MASK_STRIDE):
        position_is_active = position_idx < num_req_logits
        bitmask_idx = grammar_idx * MASK_STRIDE + position_idx
        packed_bitmask = tl.load(
            bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
            mask=position_is_active & (bitmask_offset < bitmask_stride),
            other=0,
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE)

        logits_idx = logits_start_idx + position_idx
        tl.store(
            logits_ptr + logits_idx * logits_stride + block_offset,
            -float("inf"),
            mask=(position_is_active & bitmask & (block_offset < vocab_size)),
        )
