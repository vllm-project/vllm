# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
    """

    def __init__(
        self,
        max_num_reqs: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

        # req_id -> (embeds, is_token_ids mask or None). Holds the references
        # that keep the pointer table below valid.
        self.gpu_tensors: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}

        # Indexed by req_state index. Stale entries after removal are
        # harmless: add_request rewrites all fields for every index it claims.
        self.embeds_ptrs = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self.mask_ptrs = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self.embeds_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        prompt_embeds = new_req_data.prompt_embeds
        if prompt_embeds is None:
            self.gpu_tensors.pop(new_req_data.req_id, None)
            self.embeds_lens.np[req_index] = 0
            return

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
        self.gpu_tensors.pop(req_id, None)

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


@triton.jit
def _apply_prompt_embeds_kernel(
    inputs_embeds_ptr,
    inputs_embeds_stride,
    embeds_ptrs_ptr,  # int64 [max_num_reqs], device pointers (0-len = unused)
    mask_ptrs_ptr,  # int64 [max_num_reqs], 0 = no is-token-ids mask
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
