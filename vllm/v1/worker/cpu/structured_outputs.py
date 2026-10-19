# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.structured_outputs import (
    StructuredOutputsWorker,
    _build_grammar_mapping,
)


class CPUStructuredOutputsWorker(StructuredOutputsWorker):
    """Applies grammar bitmasks with torch, straight from host memory."""

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return
        mapping = torch.tensor(
            _build_grammar_mapping(
                input_batch.req_ids,
                grammar_req_ids,
                input_batch.cu_num_logits_np,
                input_batch.num_draft_tokens_per_req,
                self.num_bonus_tokens,
                self.mask_stride,
            )
        )
        req, position = mapping // self.mask_stride, mapping % self.mask_stride
        cu_num_logits = input_batch.cu_num_logits.long()
        start = cu_num_logits[req]
        is_active = position < cu_num_logits[req + 1] - start
        rows = (start + position)[is_active]
        # Bit j of word i allows token 32 * i + j.
        words = torch.from_numpy(grammar_bitmask)[is_active]
        bits = (words.unsqueeze(-1) >> torch.arange(32, dtype=words.dtype)) & 1
        allowed = bits.view(len(rows), -1)[:, : logits.shape[-1]].bool()
        logits[rows] = logits[rows].masked_fill(~allowed, float("-inf"))
