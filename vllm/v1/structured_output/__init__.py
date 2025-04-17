# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request

logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: Optional[StructuredOutputBackend] = None
        self.reasoner: Optional[ReasoningParser] = None
        self.vllm_config = vllm_config
        self._grammar_bitmask: Optional[torch.Tensor] = None

        # The default max_workers if not specified is the number of CPUs * 5,
        # which is way too high since these tasks are CPU-bound, not I/O bound.
        # We also know we would never dominate CPU usage with just grammar
        # compilation, so we set it to half the number of CPUs.
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def grammar_init(self, request: Request) -> None:
        if request.structured_output_request is None:
            return

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        if self.backend is None:
            backend_name = request.sampling_params.guided_decoding.backend_name
            tokenizer_group = init_tokenizer_from_configs(
                model_config=self.vllm_config.model_config,
                scheduler_config=self.vllm_config.scheduler_config,
                parallel_config=self.vllm_config.parallel_config,
                lora_config=self.vllm_config.lora_config)
            tokenizer_group.ping()
            tokenizer = tokenizer_group.get_lora_tokenizer(None)
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend_name == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend_name == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(
                    f"Unsupported structured output backend: {backend_name}")

            if (reasoning_backend :=
                    self.vllm_config.decoding_config.reasoning_backend
                ) is not None and self.reasoner is None:
                self.reasoner = ReasoningParserManager.get_reasoning_parser(
                    reasoning_backend)(tokenizer=tokenizer)

        grammar = self.executor.submit(self._async_create_grammar, request)
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _async_create_grammar(
        self,
        request: Request,
    ) -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def grammar_bitmask(
        self,
        requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
        batch_len: int,
    ) -> Optional[npt.NDArray[np.int32]]:
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        if self._grammar_bitmask is None:
            assert self.backend is not None
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                self.vllm_config.scheduler_config.max_num_seqs)

        # Fill the bitmask using the index of each request equal to its
        # position in the batch. Resize the bitmask down to the size of
        # the batch.
        bitmask_tensor = self._grammar_bitmask
        # Reset the relevant part of the bitmask before filling
        if batch_len > 0:
            bitmask_tensor[:batch_len].fill_(-1)

        for req_id, batch_index in structured_output_request_ids.items():
            full_request = requests[req_id]
            so_request = full_request.structured_output_request
            assert so_request is not None and so_request.grammar is not None

            apply_bitmask = (self.reasoner is None
                             or so_request.reasoning_ended
                             or self.reasoner.is_reasoning_end(
                                 full_request.all_token_ids))

            if apply_bitmask and not so_request.grammar.is_terminated():
                so_request.grammar.fill_bitmask(bitmask_tensor, batch_index)

        if batch_len < bitmask_tensor.shape[0]:
            final_bitmask_tensor = bitmask_tensor[:batch_len]
        else:
            final_bitmask_tensor = bitmask_tensor

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return final_bitmask_tensor.numpy()

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
