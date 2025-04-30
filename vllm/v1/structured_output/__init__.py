# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend, StructuredOutputBatchMetaData,
    StructuredOutputGrammar)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.worker.gpu_input_batch import InputBatch

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.v1.request import Request

logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests.
    This manager holds a backend property used to initialise and compile grammars
    Each v1 request will then have the compiled grammar assigned to request.structured_output_request.grammar
    """

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

    def accept_tokens(self, request: Request, req_id: str,
                      tokens: list[int]) -> bool:
        """
        Called in v1.core.sched.Scheduler.update_from_output after tokens have been accepted
        Accepts a list of tokens and advances the FSM.
        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        assert request.structured_output_request is not None and request.structured_output_request.grammar is not None
        return request.structured_output_request.grammar.accept_tokens(
            req_id, tokens)

    def filter_logits(
        self,
        input_batch: InputBatch,
        device: torch.device,
        scheduler_output: SchedulerOutput,
        logits: torch.Tensor,
        sample_hidden_states: torch.Tensor,
    ) -> None:
        """
        Called in v1.worker.GPUModelRunner.execute_model immediately after the model forward pass"""
        assert self.backend is not None
        self.backend.filter_logits(input_batch, device, scheduler_output,
                                   logits, sample_hidden_states)

    def init_batch(
        self, requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]]
    ) -> StructuredOutputBatchMetaData:
        """
        Called in the v1/core/sched/Scheduler.schedule to initialize the batch of requests.
        At this point, we have completed scheduling for the current step.
        The `structured_output_request_ids` dictionary maps request IDs
        that use structured output to their corresponding indices in the
        running queue.
        """
        assert self.backend is not None
        return self.backend.init_batch(
            requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens,
        )

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
