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
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        mask_stride: int,
        num_bonus_tokens: int,
    ) -> None:
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

        mapping = _build_grammar_mapping(
            input_batch.req_ids,
            grammar_req_ids,
            input_batch.cu_num_logits_np,
            input_batch.num_draft_tokens_per_req,
            self.num_bonus_tokens,
            self.mask_stride,
        )
        assert grammar_bitmask.shape[0] == len(mapping)

        mapping_tensor = torch.tensor(mapping, dtype=torch.int64)
        req_indices = mapping_tensor // self.mask_stride
        positions = mapping_tensor % self.mask_stride
        starts = input_batch.cu_num_logits[req_indices].long()
        ends = input_batch.cu_num_logits[req_indices + 1].long()
        active = positions < ends - starts
        if not torch.any(active):
            return

        logit_rows = starts[active] + positions[active]
        packed = torch.from_numpy(grammar_bitmask)[active]
        token_ids = torch.arange(logits.shape[1], dtype=torch.int64)
        words = packed[:, token_ids // 32]
        allowed = ((words >> (token_ids % 32)) & 1).bool()

        masked_logits = logits.index_select(0, logit_rows)
        masked_logits.masked_fill_(~allowed, -torch.inf)
        logits.index_copy_(0, logit_rows, masked_logits)
