# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch

xgr = LazyLoader("xgr", globals(), "xgrammar")


def _build_grammar_mapping(
    req_ids: list[str],
    grammar_req_ids: list[str],
    cu_num_logits_np: np.ndarray,
    num_draft_tokens_per_req: np.ndarray | None,
    num_bonus_tokens: int,
    mask_stride: int,
) -> list[int]:
    mapping: list[int] = []
    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        if num_draft_tokens_per_req is None:
            num_positions = int(
                cu_num_logits_np[req_idx + 1] - cu_num_logits_np[req_idx]
            )
        else:
            # Grammar masks follow the scheduled layout even when adaptive
            # verification compacts the actual CPU logit offsets to bonus-only.
            num_positions = int(num_draft_tokens_per_req[req_idx]) + num_bonus_tokens
        mapping.extend(
            req_idx * mask_stride + position for position in range(num_positions)
        )
    return mapping


class StructuredOutputsWorker:
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        mask_stride: int,
        num_bonus_tokens: int,
        bitmask_backend: str = "auto",
    ):
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.xgrammar_bitmask = (
            torch.empty(
                (max_num_logits, cdiv(vocab_size, 32)),
                dtype=torch.int32,
                device=device,
            )
            if bitmask_backend != "auto"
            else None
        )
        self.bitmask_backend = bitmask_backend
        self.device = device
        self.copy_stream = torch.cuda.Stream()
        self.mask_stride = mask_stride
        self.num_bonus_tokens = num_bonus_tokens

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        # Asynchronously copy the bitmask to GPU.
        with torch.cuda.stream(self.copy_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Construct bitmask -> logits mapping
        # Key by (request, position) rather than absolute logit index:
        # adaptive verification finalizes per-request logit offsets on
        # device, so the kernel resolves them from the GPU cu_num_logits.
        mapping = _build_grammar_mapping(
            input_batch.req_ids,
            grammar_req_ids,
            input_batch.cu_num_logits_np,
            input_batch.num_draft_tokens_per_req,
            self.num_bonus_tokens,
            self.mask_stride,
        )

        # Asynchronously copy the mapping to GPU.
        with torch.cuda.stream(self.copy_stream):
            logits_indices = torch.tensor(
                mapping, dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        if self.bitmask_backend == "auto":
            _apply_grammar_bitmask_kernel[grid](
                logits,
                logits.stride(0),
                logits_indices,
                input_batch.cu_num_logits,
                bitmask,
                bitmask.stride(0),
                vocab_size,
                MASK_STRIDE=self.mask_stride,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            self._apply_xgrammar_bitmask(
                logits, bitmask, logits_indices, input_batch.cu_num_logits
            )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        self.copy_stream.wait_stream(current_stream)

    def _apply_xgrammar_bitmask(
        self,
        logits: torch.Tensor,
        bitmask: torch.Tensor,
        mapping: torch.Tensor,
        cu_num_logits: torch.Tensor,
    ) -> None:
        """Apply a selected XGrammar backend using the V2 request-to-logit map."""
        assert self.xgrammar_bitmask is not None

        request_indices = torch.div(
            mapping, self.mask_stride, rounding_mode="floor"
        ).to(torch.long)
        position_indices = torch.remainder(mapping, self.mask_stride)
        logit_starts = cu_num_logits[request_indices]
        num_request_logits = cu_num_logits[request_indices + 1] - logit_starts
        valid = position_indices < num_request_logits
        logit_indices = (logit_starts + position_indices)[valid]

        # XGrammar indexes the bitmask with the output-logit indices. Build a
        # dense, aligned view because V2's source bitmasks are request-ordered.
        aligned_bitmask = self.xgrammar_bitmask[: logits.shape[0]]
        aligned_bitmask.fill_(-1)
        aligned_bitmask.index_copy_(
            0,
            logit_indices.to(torch.long),
            bitmask[valid],
        )
        xgr.apply_token_bitmask_inplace(
            logits,
            aligned_bitmask,
            indices=logit_indices,
            backend=self.bitmask_backend,
        )


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    cu_num_logits_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    MASK_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    mapping_idx = tl.load(logits_indices_ptr + bitmask_idx)
    req_idx = mapping_idx // MASK_STRIDE
    position_idx = mapping_idx % MASK_STRIDE
    logits_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_req_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_idx
    logits_idx += position_idx
    position_is_active = position_idx < num_req_logits

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
        mask=position_is_active & bitmask & (block_offset < vocab_size),
    )
