# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.cpu.input_batch import InputBatch


class PromptEmbedsState:
    """Host-side state for user-provided prompt embeddings.

    Each request's embeddings are converted once at `add_request`, off the
    per-step hot path. `apply` then overlays the rows scheduled this step onto
    `inputs_embeds`, visiting only requests still inside their embeddings.
    """

    def __init__(self, max_num_reqs: int, dtype: torch.dtype) -> None:
        self.dtype = dtype

        # req_id -> (embeds, is_token_ids mask or None).
        self.tensors: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}

        # Indexed by req_state index. Stale entries after removal are
        # harmless: add_request rewrites the entry for every index it claims.
        self.embeds_lens = np.zeros(max_num_reqs, dtype=np.int32)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        prompt_embeds = new_req_data.prompt_embeds
        if prompt_embeds is None:
            self.tensors.pop(new_req_data.req_id, None)
            self.embeds_lens[req_index] = 0
            return

        embeds = prompt_embeds.to(dtype=self.dtype).contiguous()
        is_token_ids = new_req_data.prompt_is_token_ids
        mask = None
        if is_token_ids is not None:
            mask = torch.as_tensor(is_token_ids, dtype=torch.bool)
        self.tensors[new_req_data.req_id] = (embeds, mask)
        self.embeds_lens[req_index] = embeds.shape[0]

    def remove_request(self, req_id: str) -> None:
        self.tensors.pop(req_id, None)

    def apply(self, input_batch: InputBatch, inputs_embeds: torch.Tensor) -> None:
        """Overlay prompt embeddings onto `inputs_embeds` for the batch."""
        if not self.tensors:
            return
        assert inputs_embeds.dtype == self.dtype
        num_computed = input_batch.num_computed_tokens_np
        num_rows = np.minimum(
            input_batch.num_scheduled_tokens,
            self.embeds_lens[input_batch.idx_mapping_np] - num_computed,
        )
        query_start_loc = input_batch.query_start_loc_np
        # No rows for requests without prompt embeds, or with them fully consumed.
        for batch_idx in np.flatnonzero(num_rows > 0):
            embeds, mask = self.tensors[input_batch.req_ids[batch_idx]]
            src_start = int(num_computed[batch_idx])
            src = slice(src_start, src_start + int(num_rows[batch_idx]))
            dst_start = int(query_start_loc[batch_idx])
            dst = inputs_embeds[dst_start : dst_start + int(num_rows[batch_idx])]
            if mask is None:
                dst.copy_(embeds[src])
            else:
                # Mixed mode: token-id rows keep the base embedding.
                dst.copy_(torch.where(mask[src, None], dst, embeds[src]))
