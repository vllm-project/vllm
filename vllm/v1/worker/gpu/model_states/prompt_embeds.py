# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch

TOKEN_BLOCK = 16


class _HostRows(NamedTuple):
    """A request's uncomputed prompt_embeds rows, in pinned host memory."""

    embeds: torch.Tensor
    is_token_id: torch.Tensor | None
    """uint8 flags of a mixed prompt's token-id rows, or None."""


class PromptEmbedsState:
    """User-provided prompt embeddings, kept in pinned host memory.

    Each step copies only the rows it computes into a GPU staging buffer of
    `max_num_rows` rows, and points a per-request pointer table (UVA) at them,
    so a single triton kernel overlays all of them onto `inputs_embeds`.
    The staging buffer is allocated on first use, after KV cache sizing, which
    sets its memory aside.

    A request is sent only the embeddings of positions not computed yet, and
    they are dropped after the step that computes its last prompt position; a
    request rolled back below them is re-sent.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.pin_memory = is_pin_memory_available()

        # req_id -> the request's rows in host memory.
        self.host_tensors: dict[str, _HostRows] = {}
        # Indexed by req_state index: prompt position of the first host row
        # (rows below it were computed before the request was sent) and the end
        # prompt position, 0 once the request's rows are dropped.
        self.host_offsets = np.zeros(max_num_reqs, dtype=np.int32)
        self.host_ends = np.zeros(max_num_reqs, dtype=np.int32)

        # The most rows held on the GPU at once, which KV cache planning sets
        # aside: one step's rows, staged.
        self.max_num_rows = max_num_rows
        # GPU staging buffers for one step's rows and their token-id flags,
        # allocated together by _get_staged_rows on first use.
        self._staged: tuple[torch.Tensor, torch.Tensor] | None = None

        # Indexed by req_state index and rewritten each step for the requests in
        # the batch: the kernel reads staged row `num_computed + t - offset`
        # from `embeds_ptrs` up to prompt position `embeds_lens`.
        self.embeds_ptrs = UvaBackedTensor(max_num_reqs, dtype=torch.uint64)
        self.mask_ptrs = UvaBackedTensor(max_num_reqs, dtype=torch.uint64)
        self.embeds_offsets = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.embeds_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        self.host_tensors.pop(new_req_data.req_id, None)
        prompt_embeds = new_req_data.prompt_embeds
        if prompt_embeds is None:
            self.host_ends[req_index] = 0
            return

        # Pinned, so the per-step copies are DMA transfers.
        embeds = torch.empty(
            prompt_embeds.shape, dtype=self.dtype, pin_memory=self.pin_memory
        ).copy_(prompt_embeds)
        is_token_ids = new_req_data.prompt_is_token_ids
        mask = None
        if is_token_ids is not None:
            mask = torch.tensor(
                is_token_ids, dtype=torch.uint8, pin_memory=self.pin_memory
            )
        self.host_tensors[new_req_data.req_id] = _HostRows(embeds, mask)
        offset = new_req_data.prompt_embeds_offset
        assert offset <= new_req_data.num_computed_tokens
        self.host_offsets[req_index] = offset
        self.host_ends[req_index] = offset + embeds.shape[0]

    def remove_request(self, req_id: str) -> None:
        # Pinned memory read by an in-flight copy is not reused before the copy
        # completes, so this is safe right after issuing it.
        self.host_tensors.pop(req_id, None)

    def _get_staged_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The GPU staging buffers for one step's rows and their token-id
        flags, copied there from pinned host memory."""
        if self._staged is None:
            self._staged = (
                torch.empty(
                    self.max_num_rows,
                    self.hidden_size,
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.empty(self.max_num_rows, dtype=torch.uint8, device=self.device),
            )
        return self._staged

    def apply(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> None:
        """Overlay prompt embeddings onto `inputs_embeds` for the batch."""
        if not self.host_tensors:
            # No request holds prompt embeddings left to compute.
            return
        start, end = self._prompt_embeds_positions(input_batch)
        staged = np.flatnonzero(end > start)
        if len(staged) == 0:
            # No request in this batch computes a position with a prompt
            # embedding.
            return
        self._stage_rows(input_batch, start, end, staged)

        # The kernel reads the staged rows as inputs_embeds' dtype.
        assert inputs_embeds.dtype == self.dtype
        max_query_len = int(input_batch.num_scheduled_tokens.max())
        grid = (input_batch.num_reqs, triton.cdiv(max_query_len, TOKEN_BLOCK))
        _apply_prompt_embeds_kernel[grid](
            inputs_embeds,
            inputs_embeds.stride(0),
            self.embeds_ptrs.gpu,
            self.mask_ptrs.gpu,
            self.embeds_offsets.gpu,
            self.embeds_lens.gpu,
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            num_computed_tokens,
            self.hidden_size,
            TOKEN_BLOCK=TOKEN_BLOCK,
            BLOCK_SIZE=1024,
        )
        self._drop_computed(input_batch, end)

    def _drop_computed(self, input_batch: InputBatch, end: np.ndarray) -> None:
        num_reqs = input_batch.num_reqs
        idx_mapping = input_batch.idx_mapping_np[:num_reqs]
        host_ends = self.host_ends[idx_mapping]
        for i in np.flatnonzero((host_ends > 0) & (end >= host_ends)).tolist():
            self.remove_request(input_batch.req_ids[i])
            self.host_ends[idx_mapping[i]] = 0

    def _prompt_embeds_positions(
        self, input_batch: InputBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        """The prompt positions `[start, end)` each request in the batch
        computes this step that have a prompt embedding; empty for the others."""
        num_reqs = input_batch.num_reqs
        idx_mapping = input_batch.idx_mapping_np[:num_reqs]
        start = input_batch.num_computed_tokens_np[:num_reqs]
        end = np.minimum(
            start + input_batch.num_scheduled_tokens[:num_reqs],
            self.host_ends[idx_mapping],
        )
        return start, end

    def _stage_rows(
        self,
        input_batch: InputBatch,
        start: np.ndarray,
        end: np.ndarray,
        staged: np.ndarray,
    ) -> None:
        """Copy the prompt positions `[start, end)` of the `staged` requests from
        host memory into the staging buffer, and point their table entries at
        them. The other requests in the batch get no rows this step."""
        idx_mapping = input_batch.idx_mapping_np[: input_batch.num_reqs]
        assert int((end - start)[staged].sum()) <= self.max_num_rows
        # On the compute stream, so the previous step's overlay has read the
        # staging buffer before this copy overwrites it.
        staged_rows, staged_is_token_id = self._get_staged_rows()
        row_bytes = staged_rows.stride(0) * staged_rows.element_size()
        self.embeds_lens.np[idx_mapping] = 0
        dst = 0
        for i in staged.tolist():
            req_index = idx_mapping[i]
            embeds, mask = self.host_tensors[input_batch.req_ids[i]]
            num_rows = int(end[i] - start[i])
            src = int(start[i] - self.host_offsets[req_index])
            staged_rows[dst : dst + num_rows].copy_(
                embeds[src : src + num_rows], non_blocking=True
            )
            self.embeds_ptrs.np[req_index] = staged_rows.data_ptr() + dst * row_bytes
            self.mask_ptrs.np[req_index] = 0
            if mask is not None:
                staged_is_token_id[dst : dst + num_rows].copy_(
                    mask[src : src + num_rows], non_blocking=True
                )
                self.mask_ptrs.np[req_index] = staged_is_token_id.data_ptr() + dst
            self.embeds_offsets.np[req_index] = start[i]
            self.embeds_lens.np[req_index] = end[i]
            dst += num_rows
        self.embeds_ptrs.copy_to_uva()
        self.mask_ptrs.copy_to_uva()
        self.embeds_offsets.copy_to_uva()
        self.embeds_lens.copy_to_uva()


@triton.jit
def _apply_prompt_embeds_kernel(
    inputs_embeds_ptr,
    inputs_embeds_stride,
    embeds_ptrs_ptr,  # uint64 [max_num_reqs], device pointers (0-len = unused)
    mask_ptrs_ptr,  # uint64 [max_num_reqs], 0 = no is-token-ids mask
    embeds_offsets_ptr,  # int32 [max_num_reqs], prompt position of row 0
    embeds_lens_ptr,  # int32 [max_num_reqs], end prompt position
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    hidden_size: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    embeds_len = tl.load(embeds_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= embeds_len:
        # No prompt embeds for this request, or they are fully consumed.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    num_rows = tl.minimum(query_end - query_start, embeds_len - num_computed)

    t_start = tl.program_id(1) * TOKEN_BLOCK
    if t_start >= num_rows:
        return

    src_ptr = tl.load(embeds_ptrs_ptr + req_state_idx).to(
        tl.pointer_type(inputs_embeds_ptr.dtype.element_ty)
    )
    offset = tl.load(embeds_offsets_ptr + req_state_idx)
    mask_int = tl.load(mask_ptrs_ptr + req_state_idx)
    mask_ptr = mask_int.to(tl.pointer_type(tl.int8))

    for t_offset in tl.static_range(TOKEN_BLOCK):
        t = t_start + t_offset
        if t < num_rows:
            src_row = (num_computed + t - offset).to(tl.int64)
            is_token_id = 0
            if mask_int != 0:
                is_token_id = tl.load(mask_ptr + src_row).to(tl.int32)
            if is_token_id == 0:
                dst_row = (query_start + t).to(tl.int64)
                for h in tl.range(0, hidden_size, BLOCK_SIZE):
                    offs = h + tl.arange(0, BLOCK_SIZE)
                    h_mask = offs < hidden_size
                    row = tl.load(src_ptr + src_row * hidden_size + offs, mask=h_mask)
                    tl.store(
                        inputs_embeds_ptr + dst_row * inputs_embeds_stride + offs,
                        row,
                        mask=h_mask,
                    )
