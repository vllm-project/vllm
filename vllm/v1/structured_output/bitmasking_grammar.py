# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import numpy.typing as npt
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import LazyLoader, cdiv, is_pin_memory_available
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend, StructuredOutputBatchMetaData,
    StructuredOutputGrammar)
from vllm.v1.worker.gpu_input_batch import InputBatch

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.v1.request import Request

else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


@dataclass
class BitmaskSOBatchMetaData(StructuredOutputBatchMetaData):
    """
    This class is used to store the bitmask for structured output requests.
    It is used to pass the bitmask to the GPU workers.
    """

    grammar_bitmask: torch.Tensor


class BitmaskStructuredOutputBackend(StructuredOutputBackend):

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.vllm_config = vllm_config
        # Reuse our bitmask - this will also be assigned to batches
        self._grammar_bitmask: Optional[torch.Tensor] = None
        self.tpu_vocab_size: Optional[int] = None
        self.max_num_reqs: Optional[int] = None
        if current_platform.is_tpu():
            self.init_tpu()

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
    ) -> None:
        if not current_platform.is_tpu():
            BitmaskStructuredOutputBackend.apply_grammar_bitmask(
                input_batch,
                device,
                scheduler_output,
                logits,
            )
        else:
            require_struct_decoding, grammar_bitmask_padded, arange = \
            self.prepare_structured_decoding_input_tpu(logits,
                                                scheduler_output, input_batch)
            self.structured_decode_tpu(require_struct_decoding,
                                       grammar_bitmask_padded, logits, arange)

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        pass

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> Optional[npt.NDArray[np.int32]]:
        """
        Method used by XGrammar and Guidance to process and filter all logits
        """

        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        if self._grammar_bitmask is None:
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
            if self.vllm_config.speculative_config is not None:
                max_num_spec_tokens = self.vllm_config.\
                    speculative_config.num_speculative_tokens
            else:
                max_num_spec_tokens = 0

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = \
                self.allocate_token_bitmask(
                    max_batch_size * (1 + max_num_spec_tokens))
        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0
        ordered_seq = sorted(structured_output_request_ids.items(),
                             key=lambda x: x[1])
        # NOTE: This outer loop can likely be parallelized to improve
        # performance of bitmask generation for large batches.
        for req_id, _ in ordered_seq:
            request = requests[req_id].structured_output_request
            assert request is not None and isinstance(request.grammar,
                                                      BitmaskGrammar)
            state_advancements = 0
            req_tokens = scheduled_spec_decode_tokens.get(req_id, []) + [None]
            for i, token in enumerate(req_tokens):
                if not request.grammar.is_terminated():
                    request.grammar.fill_bitmask(self._grammar_bitmask,
                                                 cumulative_index)
                    if token is not None:
                        # In order to generate the correct bitmask for each
                        # position in the speculative sequence, we advance
                        # the FSM state for each speculative token and rollback
                        # to restore the previous state when we are finished.
                        assert request.grammar.accept_tokens(req_id, [token])
                        state_advancements += 1
                cumulative_index += 1
            if state_advancements > 0:
                request.grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < self._grammar_bitmask.shape[0]:
            bitmask_tensor = self._grammar_bitmask[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def init_batch(
        self, requests: dict[str, "Request"],
        structured_output_request_ids: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]]
    ) -> StructuredOutputBatchMetaData:
        bitmask = self.grammar_bitmask(requests, structured_output_request_ids,
                                       scheduled_spec_decode_tokens)
        return BitmaskSOBatchMetaData(structured_output_request_ids, bitmask)

    def init_tpu(self):
        # TPU specific tensors
        if self.max_num_reqs is None or \
                self.tpu_vocab_size is None:
            self.tpu_vocab_size = self.vllm_config.model_config.get_vocab_size(
            )
            MIN_NUM_SEQS = 8
            self.max_num_reqs = max(
                self.vllm_config.scheduler_config.max_num_seqs, MIN_NUM_SEQS)
            pin_memory = is_pin_memory_available()
            self.require_structured_out_cpu = torch.zeros(
                (self.max_num_reqs),
                dtype=torch.bool,
                device="cpu",
                pin_memory=pin_memory)
            self.structured_decode_arange = torch.arange(0,
                                                         32,
                                                         device="cpu",
                                                         pin_memory=pin_memory)
            self.grammar_bitmask_cpu = torch.zeros(
                (self.max_num_reqs, cdiv(self.tpu_vocab_size, 32)),
                dtype=torch.int32,
                device="cpu",
                pin_memory=pin_memory)

    def prepare_structured_decoding_input_tpu(
        self, logits: torch.Tensor, scheduler_output: "SchedulerOutput",
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
    def structured_decode_tpu(self, require_struct_decoding: torch.Tensor,
                              grammar_bitmask: torch.Tensor,
                              logits: torch.Tensor, arange: torch.Tensor):
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

    def precompile(self, num_reqs_paddings: list[int], device: torch.device,
                   hidden_states_dtype: torch.dtype):
        if current_platform.is_tpu():
            for num_reqs in num_reqs_paddings:
                dummy_logits = torch.zeros((num_reqs, self.tpu_vocab_size),
                                           device=device,
                                           dtype=hidden_states_dtype)
                dummy_require_struct_decoding = \
                    self.require_structured_out_cpu[:num_reqs].to(device)
                dummy_grammar_bitmask = \
                    self.grammar_bitmask_cpu[:num_reqs].to(device)
                # The first dimension of the above 3 dummy tensors cannot be
                # mark_dynamic because some operations in structured_decode
                # require them to be static.
                arange = self.structured_decode_arange.to(device)
                self.structured_decode_tpu(dummy_require_struct_decoding,
                                           dummy_grammar_bitmask, dummy_logits,
                                           arange)
                logger.info("  -- num_seqs: %d", num_reqs)


class BitmaskGrammar(StructuredOutputGrammar):

    @abstractmethod
    def is_terminated(self) -> bool:
        """
        Checks whether the structured output process has terminated.

        Returns:
            bool: True if the process is terminated, False otherwise.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the state of the structured output grammar.
        """

    @abstractmethod
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        pass
