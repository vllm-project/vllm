# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.worker.cpu.input_batch import InputBatch


class CPUStructuredOutputsWorker:
    def __init__(
        self,
        max_num_logits: int,
        vocab_size: int,
        device: torch.device,
        mask_stride: int,
    ) -> None:
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

        # Each grammar request owns `mask_stride` bitmask rows. Key them by
        # (request, position); positions beyond cu_num_logits are skipped.
        req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
        mapping = [
            req_id_to_idx[req_id] * self.mask_stride + position
            for req_id in grammar_req_ids
            for position in range(self.mask_stride)
        ]
        assert grammar_bitmask.shape[0] == len(mapping)

        mapping_tensor = torch.tensor(mapping, dtype=torch.int64, device=logits.device)
        req_indices = mapping_tensor // self.mask_stride
        positions = mapping_tensor % self.mask_stride
        starts = input_batch.cu_num_logits[req_indices].long()
        ends = input_batch.cu_num_logits[req_indices + 1].long()
        active = positions < ends - starts
        if not torch.any(active):
            return

        logit_rows = starts[active] + positions[active]
        packed = (
            grammar_bitmask if torch.all(active) else grammar_bitmask[active.numpy()]
        )
        # Normalize byte order before unpacking from the least-significant bit.
        packed_bytes = np.ascontiguousarray(packed, dtype="<i4").view(np.uint8)
        disallowed = np.unpackbits(
            packed_bytes,
            axis=1,
            count=logits.shape[1],
            bitorder="little",
        )
        np.bitwise_xor(disallowed, 1, out=disallowed)

        masked_logits = logits.index_select(0, logit_rows)
        masked_logits.masked_fill_(
            torch.from_numpy(disallowed.view(np.bool_)), -torch.inf
        )
        logits.index_copy_(0, logit_rows, masked_logits)
