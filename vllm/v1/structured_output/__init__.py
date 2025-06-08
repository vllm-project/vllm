# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import (
    GuidanceBackend, validate_guidance_grammar)
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend, StructuredOutputBatchMetaData,
    StructuredOutputGrammar)
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend, validate_xgrammar_grammar)
from vllm.v1.structured_output.worker_backend import (
    StructuredOutputWorkerBackend)
from vllm.v1.structured_output.worker_backend_bitmasking_gpu import (
    BitmaskGPUBackend)
from vllm.v1.structured_output.worker_backend_bitmasking_tpu import (
    BitmaskTPUBackend)

if TYPE_CHECKING:
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")

logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests.
    This manager holds a backend property used to initialise and 
     compile grammars
    Each v1 request will then have the compiled grammar assigned to 
     request.structured_output_request.grammar
    """

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
        self.tokenizer = init_tokenizer_from_configs(
            model_config=self.vllm_config.model_config,
            scheduler_config=self.vllm_config.scheduler_config,
            lora_config=self.vllm_config.lora_config,
        ).get_lora_tokenizer(None)
        reasoning_backend = vllm_config.decoding_config.reasoning_backend
        if reasoning_backend:
            reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                reasoning_backend)
            self.reasoner = reasoner_cls(tokenizer=self.tokenizer)

    @staticmethod
    def get_worker_backend(
            vllm_config: VllmConfig) -> StructuredOutputWorkerBackend:
        if current_platform.is_tpu():
            return BitmaskTPUBackend(vllm_config)
        else:
            return BitmaskGPUBackend(vllm_config)

    def init_backend(self, backend: str) -> None:
        """
        Initialize the backend for structured output processing.
        This method is called when the engine starts up and is responsible
        for setting up the backend for structured output requests.
        """
        if self.backend is not None:
            return
        if backend == "auto":
            if self.vllm_config.decoding_config.backend != "auto":
                backend = self.vllm_config.decoding_config.backend
            else:
                backend = "xgrammar"  # default to xgrammar

        vocab_size = self.vllm_config.model_config.get_vocab_size()

        if backend in ["xgrammar", "guidance"]:  # Bitmasking Backends
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                    reasoner=self.reasoner,
                )
            else:  # Guidance
                self.backend = GuidanceBackend(  # type: ignore[assignment]
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                    reasoner=self.reasoner,
                )
        else:
            raise ValueError(
                f"Unsupported structured output backend: {backend}")

    def grammar_init(self, request: Request) -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert request.sampling_params.guided_decoding is not None

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        if self.backend is None:
            self.init_backend(request.sampling_params.guided_decoding.backend
                              )  # type: ignore[union-attr]

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
        Validates whether the provided tokens are acceptable based on 
        the grammar defined in the structured output request.
        
        Called in v1.core.sched.Scheduler.update_from_output after 
        tokens have been accepted
        Args:
            request (Request): The request object containing the 
             structured output request and its associated grammar.
            req_id (str): The unique identifier for the request.
            tokens (list[int]): A list of integer tokens to be validated.
        Returns:
            bool: True if the FSM was advanced successfully. 
            False if the FSM failed to advance.
        """
        assert request.structured_output_request is not None and \
            request.structured_output_request.grammar is not None
        return request.structured_output_request.grammar.accept_tokens(
            req_id, tokens)

    def init_batch(
        self, requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
        scheduled_spec_decode_tokens: dict[str, list[int]]
    ) -> StructuredOutputBatchMetaData | None:
        """
        Called in the v1/core/sched/Scheduler.schedule to initialize
        the batch of requests.
        At this point, we have completed scheduling for the current step.
        The `structured_output_request_ids` dictionary maps request IDs
        that use structured output to their corresponding indices in the
        running queue.
        Args:
            requests (dict[str, Request]): A dictionary mapping request IDs
                to their corresponding `Request` objects.
            structured_output_request_ids (dict[str, int]): A dictionary mapping
                request IDs that use structured output to their respective 
                indices in the running queue.
            scheduled_spec_decode_tokens (dict[str, list[int]]): A dictionary
                mapping request IDs to lists of token IDs that are scheduled
                for decoding.
        Returns:
            StructuredOutputBatchMetaData: Metadata for the initialized batch
            of structured output requests.
        """

        assert self.backend is not None
        if not structured_output_request_ids:
            return None
        else:
            return self.backend.init_batch(
                requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )

    def should_advance(self, request: Request) -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that doesn't uses thinking mode.
        if self.reasoner is not None:
            structured_req = request.structured_output_request

            if structured_req.reasoning_ended:
                return True

            # Check if reasoning ends in *this* step
            if self.reasoner.is_reasoning_end(request.all_token_ids):
                # Reasoning just ended, so we shouldn't advanced til
                # next pass
                structured_req.reasoning_ended = True

            return False
        else:
            return True

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()

    def precompile(self, dummy_logits: torch.Tensor, **kwargs):
        """
        Allow backend precompilation for the device
            - Currently only used in the TPU model runner

        Args:
            num_reqs_paddings (List[int]): A list of padding sizes for the 
                number of requests.
            vocab_size (int): The size of the vocabulary.
            device (torch.device): The device on which the model is running.
                hidden_states_dtype (torch.dtype): The data type of the 
                hidden states.
        """
        assert self.backend is not None
        self.backend.precompile(dummy_logits, **kwargs)

    @staticmethod
    def validate_request(params: SamplingParams,
                         vllm_config: VllmConfig) -> None:
        """
        Validate the request for structured output.
        This method checks the request for any errors or inconsistencies
        
        If one backend fails validation, we try the next one.

        The SamplingParams object is modified to set the backend and
        backend_was_auto attributes based on the validation results.
        
        This needs to be a static method as it is called from the request 
        Processor which runs in a different process

        Args:
            params (SamplingParams): The sampling parameters for the request.

        Raises:
            ValueError: If the request contains an invalid backend or if the
                request-level backend selection is not supported.
        """
        if not params.guided_decoding or not vllm_config.decoding_config:
            return

        engine_level_backend = vllm_config.decoding_config.backend
        if params.guided_decoding.backend:
            # Request-level backend selection is not supported in V1.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_auto` option set on the backend in the params.
            if (params.guided_decoding.backend != engine_level_backend
                    and not (engine_level_backend == "auto"
                             and params.guided_decoding.backend_was_auto)):
                raise ValueError(
                    "Request-level structured output backend selection is no "
                    "longer supported. The request specified "
                    f"'{params.guided_decoding.backend}', but vLLM was "
                    f"initialised with '{engine_level_backend}'. This error "
                    "can be resolved by removing backend selection from the "
                    "request.")
        else:
            params.guided_decoding.backend = engine_level_backend

        # Request content validation
        if engine_level_backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(params)
        elif engine_level_backend.startswith("guidance"):
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            validate_guidance_grammar(params, tokenizer=None)
        else:
            # NOTE: engine_level_backend must be "auto" here, because we have
            # checked supported_backends above.
            # "auto" is an opt-in to opinionated behavior where we try to
            # choose a backend based on request contents. This is not the
            # default as it is less predictable and subject to change
            # between releases as feature support changes.
            try:
                validate_xgrammar_grammar(params)
                params.guided_decoding.backend = "xgrammar"
            except ValueError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar. Fall back to guidance.
                validate_guidance_grammar(params, tokenizer=None)
                params.guided_decoding.backend = "guidance"
            # Remember that this backend was set automatically
            params.guided_decoding.backend_was_auto = True
