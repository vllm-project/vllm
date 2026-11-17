# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch

TOKEN_BLOCK = 16


class PromptEmbedsState:
    """GPU-side state for user-provided prompt embeddings.

    Each request's embeddings are copied to the GPU once at `add_request`,
    off the per-step hot path. A per-request pointer table (UVA) then lets a
    single triton kernel overlay all scheduled prompt-embeds rows onto
    `inputs_embeds` each step, with no python loops or per-request H2D copies.

    A request's embeddings are dropped after the step that computes its last
    prompt position, unless `keep_until_finished`. The scheduler admits
    requests so that the rows held here stay within
    `max_num_prompt_embeds_tokens`, which memory profiling reserves.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        keep_until_finished: bool = False,
    ):
        self.hidden_size = hidden_size
        self.keep_until_finished = keep_until_finished
        self.dtype = dtype
        self.device = device

        # req_id -> (embeds, is_token_ids mask or None). Holds the references
        # that keep the pointer table below valid.
        self.gpu_tensors: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        # Rows held in gpu_tensors, bounded by the scheduler's budget.
        self.max_num_rows = max_num_rows
        self.num_rows = 0

        # Indexed by req_state index. Stale entries after removal are
        # harmless: add_request rewrites all fields for every index it claims.
        self.embeds_ptrs = UvaBackedTensor(max_num_reqs, dtype=torch.uint64)
        self.mask_ptrs = UvaBackedTensor(max_num_reqs, dtype=torch.uint64)
        self.embeds_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        self._release(new_req_data.req_id)
        prompt_embeds = new_req_data.prompt_embeds
        if prompt_embeds is None:
            self.embeds_lens.np[req_index] = 0
            return

        # The scheduler releases rows when it schedules the step that drops
        # them, this holds only while steps run in order and each removes
        # requests before adding new ones.
        num_rows = prompt_embeds.shape[0]
        assert self.num_rows + num_rows <= self.max_num_rows, (
            f"prompt_embeds rows exceed max_num_prompt_embeds_tokens "
            f"({self.num_rows} held + {num_rows} > {self.max_num_rows})"
        )
        self.num_rows += num_rows
        embeds = async_tensor_h2d(prompt_embeds, device=self.device, dtype=self.dtype)
        embeds = embeds.contiguous()
        is_token_ids = new_req_data.prompt_is_token_ids
        mask = None
        if is_token_ids is not None:
            mask = async_tensor_h2d(is_token_ids, device=self.device, dtype=torch.uint8)
        self.gpu_tensors[new_req_data.req_id] = (embeds, mask)
        self.embeds_ptrs.np[req_index] = embeds.data_ptr()
        self.mask_ptrs.np[req_index] = 0 if mask is None else mask.data_ptr()
        self.embeds_lens.np[req_index] = embeds.shape[0]

    def remove_request(self, req_id: str) -> None:
        self._release(req_id)

    def _release(self, req_id: str) -> None:
        tensors = self.gpu_tensors.pop(req_id, None)
        if tensors is not None:
            self.num_rows -= tensors[0].shape[0]

    def apply_staged_writes(self) -> None:
        self.embeds_ptrs.copy_to_uva()
        self.mask_ptrs.copy_to_uva()
        self.embeds_lens.copy_to_uva()

    def apply(
        self,
        input_batch: InputBatch,
        num_computed_tokens: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> None:
        """Overlay prompt embeddings onto `inputs_embeds` for the batch."""
        if not self.gpu_tensors:
            return
        # The kernel reinterprets raw source pointers as inputs_embeds' dtype.
        assert inputs_embeds.dtype == self.dtype
        num_reqs = input_batch.num_reqs
        max_query_len = int(input_batch.num_scheduled_tokens.max())
        grid = (num_reqs, triton.cdiv(max_query_len, TOKEN_BLOCK))
        _apply_prompt_embeds_kernel[grid](
            inputs_embeds,
            inputs_embeds.stride(0),
            self.embeds_ptrs.gpu,
            self.mask_ptrs.gpu,
            self.embeds_lens.gpu,
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            num_computed_tokens,
            self.hidden_size,
            TOKEN_BLOCK=TOKEN_BLOCK,
            BLOCK_SIZE=1024,
        )
        if not self.keep_until_finished:
            self._drop_computed(input_batch)

    def _drop_computed(self, input_batch: InputBatch) -> None:
        # Freeing is stream-ordered after the kernel above, which reads them.
        # The stale UVA length is harmless: once its rows are computed, the
        # kernel skips a request before reading its pointer.
        num_reqs = input_batch.num_reqs
        idx_mapping = input_batch.idx_mapping_np[:num_reqs]
        embeds_lens = self.embeds_lens.np[idx_mapping]
        end = (
            input_batch.num_computed_tokens_np[:num_reqs]
            + input_batch.num_scheduled_tokens[:num_reqs]
        )
        for i in np.flatnonzero((embeds_lens > 0) & (end >= embeds_lens)).tolist():
            self._release(input_batch.req_ids[i])
            self.embeds_lens.np[idx_mapping[i]] = 0


@triton.jit
def _apply_prompt_embeds_kernel(
    inputs_embeds_ptr,
    inputs_embeds_stride,
    embeds_ptrs_ptr,  # uint64 [max_num_reqs], device pointers (0-len = unused)
    mask_ptrs_ptr,  # uint64 [max_num_reqs], 0 = no is-token-ids mask
    embeds_lens_ptr,  # int32 [max_num_reqs]
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
    mask_int = tl.load(mask_ptrs_ptr + req_state_idx)
    mask_ptr = mask_int.to(tl.pointer_type(tl.int8))

    for t_offset in tl.static_range(TOKEN_BLOCK):
        t = t_start + t_offset
        if t < num_rows:
            src_row = (num_computed + t).to(tl.int64)
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
