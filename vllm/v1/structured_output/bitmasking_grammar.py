# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import numpy.typing as npt
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import LazyLoader
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
        BitmaskStructuredOutputBackend.apply_grammar_bitmask(
            input_batch,
            device,
            scheduler_output,
            logits,
        )

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
