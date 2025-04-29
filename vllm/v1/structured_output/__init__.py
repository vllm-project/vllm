# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.v1.request import Request

logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: Optional[StructuredOutputBackend] = None
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
            backend = request.sampling_params.guided_decoding.backend
            if backend == "xgrammar":
                from vllm.v1.structured_output.backend_xgrammar import (
                    XgrammarBackend)

                self.backend = XgrammarBackend(self.vllm_config)
            elif backend == "guidance":
                self.backend = GuidanceBackend(self.vllm_config)
            else:
                raise ValueError(
                    f"Unsupported structured output backend: {backend}")

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
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> Optional[npt.NDArray[np.int32]]:
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        if self._grammar_bitmask is None:
            assert self.backend is not None
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
                self.backend.allocate_token_bitmask(
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
            assert request is not None and request.grammar is not None
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

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
