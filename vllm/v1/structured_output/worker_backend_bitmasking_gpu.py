# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import torch
import xgrammar as xgr

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.structured_output.backend_bitmasking import BitmaskSOBatchMetaData
from vllm.v1.structured_output.worker_backend import (
    StructuredOutputWorkerBackend)
from vllm.v1.worker.gpu_input_batch import InputBatch

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class BitmaskGPUBackend(StructuredOutputWorkerBackend):

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self._grammar_bitmask: Optional[torch.Tensor] = None

    @staticmethod
    def apply_grammar_bitmask(
        input_batch: InputBatch,
        device: torch.device,
        scheduler_output: SchedulerOutput,
        logits: torch.Tensor,
    ):
        meta = cast(BitmaskSOBatchMetaData,
                    scheduler_output.structured_output_meta)
        if meta.grammar_bitmask is None:
            return
        grammar_bitmask = meta.grammar_bitmask

        # We receive the structured output bitmask from the scheduler,
        # compacted to contain bitmasks only for structured output requests.
        # The order of the requests in the bitmask is not guaranteed to be the
        # same as the order of the requests in the gpu runner's batch. We need
        # to sort the bitmask to match the order of the requests used here.

        # Get the batch indices of the structured output requests.
        # Keep track of the number of speculative tokens scheduled for every
        # request in the batch, as the logit indices are offset by this amount.
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        seq = sorted(input_batch.req_id_to_index.items(), key=lambda x: x[1])
        for req_id, batch_index in seq:
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            if req_id in meta.structured_output_request_ids:
                struct_out_req_batch_indices[req_id] = logit_index

        out_indices = []

        # Reorder the bitmask to match the order of the requests in the batch.
        sorted_bitmask = np.zeros_like(grammar_bitmask,
                                       shape=(logits.shape[0],
                                              grammar_bitmask.shape[1]))
        cumulative_index = 0
        seq = sorted(meta.structured_output_request_ids.items(),
                     key=lambda x: x[1])
        for req_id, _ in seq:
            logit_index = struct_out_req_batch_indices[req_id]
            num_spec_tokens = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            for i in range(1 + num_spec_tokens):
                sorted_bitmask[logit_index + i] = \
                    grammar_bitmask[cumulative_index + i]
                out_indices.append(logit_index + i)
            cumulative_index += 1 + num_spec_tokens
        grammar_bitmask = sorted_bitmask

        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = torch.from_numpy(grammar_bitmask)

        xgr.apply_token_bitmask_inplace(
            logits,
            grammar_bitmask.to(device, non_blocking=True),
            indices=out_indices,
        )

    def filter_logits(
        self,
        input_batch: InputBatch,
        device: torch.device,
        scheduler_output: SchedulerOutput,
        logits: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        **kwargs,
    ) -> None:
        BitmaskGPUBackend.apply_grammar_bitmask(
            input_batch,
            device,
            scheduler_output,
            logits,
        )

    def supported_backends(self) -> list[str]:
        return ["xgrammar", "guidance"]
