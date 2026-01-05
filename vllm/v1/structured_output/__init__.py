# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        self.reasoner: ReasoningParser | None = None
        self.vllm_config = vllm_config

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the WAITING_FOR_FSM → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        self._grammar_bitmask: torch.Tensor | None = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            reasoning_parser_plugin = (
                self.vllm_config.structured_outputs_config.reasoning_parser_plugin
            )
            if reasoning_parser_plugin and len(reasoning_parser_plugin) > 3:
                ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

            reasoning_parser = (
                self.vllm_config.structured_outputs_config.reasoning_parser
            )
            if reasoning_parser:
                reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )
                self.reasoner = reasoner_cls(tokenizer=self.tokenizer)

        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def grammar_init(self, request: "Request") -> None:
        if request.structured_output_request is None:
            return

        if TYPE_CHECKING:
            assert (
                request.sampling_params is not None
                and request.sampling_params.structured_outputs is not None
            )

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            assert request.sampling_params is not None
            backend = request.sampling_params.structured_outputs._backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "outlines":
                from vllm.v1.structured_output.backend_outlines import OutlinesBackend

                self.backend = OutlinesBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "lm-format-enforcer":
                from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E501
                    LMFormatEnforcerBackend,
                )

                self.backend = LMFormatEnforcerBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            else:
                raise ValueError(f"Unsupported structured output backend: {backend}")

        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            grammar = self._create_grammar(request)  # type: ignore[assignment]
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        #
        # TODO: we still need to handle xgrammar compilation failures,
        # though it should be unlikely as we test that up front as well.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)

                # Check if reasoning ends within the scheduled spec tokens.
                # This handles MTP/spec decode where </think> might be a draft
                # token, and we need to constrain the bonus token that follows.
                reasoning_ends_at_idx = -1
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                if (
                    not apply_bitmask
                    and self.reasoner is not None
                    and req_tokens
                    and not structured_output_request.reasoning_ended
                ):
                    # Check each spec token position to find where reasoning ends
                    for idx in range(len(req_tokens)):
                        check_seq = list(request.all_token_ids) + list(
                            req_tokens[: idx + 1]
                        )
                        if self.reasoner.is_reasoning_end(check_seq):
                            reasoning_ends_at_idx = idx
                            break

                state_advancements = 0
                allow_advance = True
                apply_bitmask_from_reasoning = (
                    not apply_bitmask and reasoning_ends_at_idx >= 0
                )
                if apply_bitmask:
                    advance_from_idx = 0
                elif reasoning_ends_at_idx >= 0:
                    # Start advancing after reasoning ends within spec tokens.
                    advance_from_idx = reasoning_ends_at_idx + 1
                else:
                    advance_from_idx = None
                for token_idx, token in enumerate(itertools.chain(req_tokens, (-1,))):
                    # Enable bitmask for positions after reasoning ends in spec
                    # tokens. token_idx corresponds to spec token positions,
                    # with -1 being the bonus token position.
                    if (
                        not apply_bitmask
                        and reasoning_ends_at_idx >= 0
                        and token_idx > reasoning_ends_at_idx
                    ):
                        apply_bitmask = True

                    if apply_bitmask_from_reasoning and token == -1:
                        # Don't constrain the bonus token based on draft-only
                        # reasoning end predictions.
                        apply_bitmask = False

                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    if token == -1:
                        # Stop advancing the grammar once we hit a padding token.
                        apply_bitmask = False
                    if (
                        advance_from_idx is not None
                        and not grammar.is_terminated()
                        and allow_advance
                        and token_idx >= advance_from_idx
                        and token != -1
                    ):
                        if grammar.validate_tokens([token]):
                            accepted = grammar.accept_tokens(req_id, [token])
                            assert accepted, (
                                token,
                                req_id,
                                scheduled_spec_decode_tokens,
                            )
                            state_advancements += 1
                        else:
                            # Draft token is invalid; stop advancing for
                            # subsequent speculative positions.
                            allow_advance = False
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        if self.reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                request.structured_output_request.reasoning_ended = (
                    self.reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    def should_advance(self, request: "Request") -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        if self.reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            # Guard against speculative draft tokens that suggested reasoning
            # ended but were not actually accepted in the output.
            if (
                not structured_req.reasoning_ended_by_fallback
                and not self.reasoner.is_reasoning_end(list(request.all_token_ids))
            ):
                structured_req.reasoning_ended = False
                return False
            return True

        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        delta_ids = list(request.all_token_ids[delta_from:])
        if self.reasoner.is_reasoning_end_streaming(request.all_token_ids, delta_ids):
            structured_req.reasoning_ended = True
            # Try to sync grammar with tokens after </think> that were
            # generated in this step. Use validate_tokens to only accept
            # tokens that are actually valid according to the grammar.
            # This handles MTP/spec decode where tokens after </think>
            # may have been generated without constraints.
            content_ids = self.reasoner.extract_content_ids(delta_ids)
            if content_ids:
                grammar = structured_req.grammar
                assert grammar is not None
                # Only accept the valid prefix of content tokens
                valid_tokens = grammar.validate_tokens(content_ids)
                if valid_tokens:
                    grammar.accept_tokens(request.request_id, valid_tokens)
            # Return False so caller doesn't try to accept all delta tokens
            # (which would include </think> and reasoning tokens).
            return False

        return False

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
