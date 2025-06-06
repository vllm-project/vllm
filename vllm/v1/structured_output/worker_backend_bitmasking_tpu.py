# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import cdiv, is_pin_memory_available
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.structured_output.backend_bitmasking import BitmaskSOBatchMetaData
from vllm.v1.structured_output.worker_backend import (
    StructuredOutputWorkerBackend)
from vllm.v1.worker.gpu_input_batch import InputBatch

if TYPE_CHECKING:

    pass

logger = init_logger(__name__)


class BitmaskTPUBackend(StructuredOutputWorkerBackend):

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self._grammar_bitmask: Optional[torch.Tensor] = None
        self.max_num_reqs: Optional[int] = None
        self.tpu_vocab_size = self.vllm_config.model_config.get_vocab_size()
        self.pin_memory = is_pin_memory_available()
        self.require_structured_out_cpu = torch.Tensor()
        self.structured_decode_arange = torch.Tensor()
        self.grammar_bitmask_cpu = torch.Tensor()

    def init_tensors(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs),
            dtype=torch.bool,
            device="cpu",
            pin_memory=self.pin_memory)
        self.structured_decode_arange = torch.arange(
            0, 32, device="cpu", pin_memory=self.pin_memory)
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.tpu_vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory)

    def filter_logits(
        self,
        input_batch: InputBatch,
        device: torch.device,
        scheduler_output: SchedulerOutput,
        logits: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        **kwargs,
    ) -> None:
        if self.max_num_reqs is None:
            assert "max_num_reqs" in kwargs, "max_num_reqs must be provided"
            max_num_reqs = kwargs.get("max_num_reqs")
            assert isinstance(max_num_reqs, int), \
                "max_num_reqs must be an integer"
            self.init_tensors(max_num_reqs)

        require_struct_decoding, grammar_bitmask_padded, arange = \
        self.prepare_structured_decoding_input(logits,
                                            scheduler_output, input_batch)
        self.structured_decode(require_struct_decoding, grammar_bitmask_padded,
                               logits, arange)

    def prepare_structured_decoding_input(
        self, logits: torch.Tensor, scheduler_output: SchedulerOutput,
        input_batch: InputBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        meta = cast(BitmaskSOBatchMetaData,
                    scheduler_output.structured_output_meta)
        grammar_bitmask = meta.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the tpu runner is
        # ordering the requests in the batch. We need to match the order of
        # bitmask with the order of requests
        struct_out_indices: list[int] = []
        mask_indices: list[int] = []
        assert scheduler_output.structured_output_meta is not None
        for req_id in input_batch.req_ids:
            mask_index = scheduler_output.structured_output_meta.\
                structured_output_request_ids.get(req_id)
            if mask_index is None:
                continue
            batch_index = input_batch.req_id_to_index[req_id]
            struct_out_indices.append(batch_index)
            mask_indices.append(mask_index)
        self.grammar_bitmask_cpu[struct_out_indices] = torch.from_numpy(
            grammar_bitmask[mask_indices])
        # It's not guaranteed that all requests in this batch require
        # structured output, so create a bool tensor to represent
        # the requests that need structured output.
        struct_out_indices = torch.tensor(struct_out_indices, dtype=torch.long)
        self.require_structured_out_cpu[struct_out_indices] = True
        return self.require_structured_out_cpu[:num_reqs].to(logits.device), \
            self.grammar_bitmask_cpu[:num_reqs].to(logits.device), \
            self.structured_decode_arange.to(logits.device)

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def structured_decode(self, require_struct_decoding: torch.Tensor,
                          grammar_bitmask: torch.Tensor, logits: torch.Tensor,
                          arange: torch.Tensor):
        """Applies structured decoding by modifying logits in-place 
        where required.
        
        Args:
            require_struct_decoding: [B] boolean tensor indicating 
                which batch items need structured decoding
            grammar_bitmask: [B, vocab_size//32] packed bit tensor 
                containing valid token masks
            logits: [B, vocab_size] tensor to modify in-place
            arange: [32] tensor for bit unpacking, contains values [0..31]
        """
        assert (logits.shape[0] == grammar_bitmask.shape[0])

        # Unpack bits for all batch items at once
        unpacked_bitmask = (
            torch.bitwise_right_shift(
                grammar_bitmask[:, :, None],  # [B, vocab_size//32, 1]
                arange[None, None, :]  # [1, 1, 32]
            ) & 1) == 0  # Result: [B, vocab_size//32, 32]

        unpacked_bitmask = unpacked_bitmask.reshape(
            logits.shape[0], -1)[:, :self.tpu_vocab_size]  # [B, vocab_size]

        # Only apply mask where require_struct_decoding is True
        mask_to_apply = unpacked_bitmask & \
            require_struct_decoding[:,None]  # [B, vocab_size]

        # Apply mask in-place
        logits.masked_fill_(mask_to_apply, -float("inf"))

    def precompile(self, dummy_logits: torch.Tensor, **kwargs):
        if self.max_num_reqs is None:
            assert "max_num_reqs" in kwargs, "max_num_reqs must be provided"
            max_num_reqs = kwargs.get("max_num_reqs")
            assert isinstance(max_num_reqs, int), \
                "max_num_reqs must be an integer"
            self.init_tensors(max_num_reqs)

        num_reqs = dummy_logits.shape[0]
        dummy_require_struct_decoding = \
            self.require_structured_out_cpu[:num_reqs].to(dummy_logits.device)
        dummy_grammar_bitmask = \
            self.grammar_bitmask_cpu[:num_reqs].to(dummy_logits.device)
        # The first dimension of the dummy logits and 2 dummy tensors above
        # cannot be mark_dynamic because some operations in structured_decode
        # require them to be static.
        arange = self.structured_decode_arange.to(dummy_logits.device)
        self.structured_decode(dummy_require_struct_decoding,
                               dummy_grammar_bitmask, dummy_logits, arange)

    def supported_backends(self) -> list[str]:
        return ["xgrammar", "guidance"]
