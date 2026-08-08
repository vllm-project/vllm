# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import multiprocessing
from collections.abc import Iterable, Sequence
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
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        self.reasoner_cls: type[ReasoningParser] | None = None
        self.vllm_config = vllm_config

        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
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
                self.reasoner_cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )

        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        structured_req = request.structured_output_request
        if structured_req is None or self.reasoner_cls is None:
            return None

        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(
                tokenizer=self.tokenizer,
                **parser_kwargs,
            )
        if not structured_req.reasoning_prompt_state_initialized:
            prompt_state = structured_req.reasoner.is_reasoning_end_from_prompt(
                request.prompt_token_ids or []
            )
            if structured_req.reasoning_ended is None:
                structured_req.reasoning_ended = prompt_state
            structured_req.reasoning_prompt_state_initialized = True
        return structured_req.reasoner

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

        grammar: Future[StructuredOutputGrammar] | StructuredOutputGrammar
        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            try:
                grammar = self._create_grammar(request)
            except Exception as e:
                grammar = Future()
                grammar.set_exception(e)
        request.structured_output_request.grammar = grammar

    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        struct_request = request.structured_output_request
        assert struct_request is not None
        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request. Grammar
        # compilation may still fail; the Future carries that error to the
        # scheduler so it can fail only this request.
        try:
            request_type, grammar_spec = struct_request.structured_output_key
            assert self.backend is not None
            return self.backend.compile_grammar(request_type, grammar_spec)
        except Exception:
            logger.exception(
                "Failed to compile grammar for request %s", request.request_id
            )
            raise

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

    def _fill_undetermined_reasoning_bitmask(
        self,
        grammar: StructuredOutputGrammar,
        index: int,
        request_id: str,
        reasoner: "ReasoningParser",
        output_token_ids: Sequence[int],
    ) -> None:
        """Allow grammar tokens plus tokens that can continue a marker."""
        assert self._grammar_bitmask is not None
        marker_sequences = getattr(reasoner, "reasoning_marker_token_ids", ())
        if not marker_sequences:
            self._grammar_bitmask[index].fill_(self._full_mask)
            return

        output_prefix = tuple(output_token_ids)
        marker_token_state = getattr(reasoner, "reasoning_marker_token_state", None)
        if marker_token_state is not None:
            marker_complete, next_marker_token_ids = marker_token_state(output_prefix)
            next_marker_token_ids = set(next_marker_token_ids)
        else:
            marker_complete = any(
                len(output_prefix) >= len(marker)
                and output_prefix[: len(marker)] == marker
                for marker in marker_sequences
            )
            next_marker_token_ids = {
                marker[len(output_prefix)]
                for marker in marker_sequences
                if len(output_prefix) < len(marker)
                and output_prefix == marker[: len(output_prefix)]
            }

        if marker_complete:
            # A complete marker means generated reasoning has started (or a
            # leading closer is being discarded). Leave reasoning unconstrained.
            self._grammar_bitmask[index].fill_(self._full_mask)
            return

        row = self._grammar_bitmask[index]
        grammar_advanced = 0
        grammar_prefix_is_valid = not output_prefix
        if output_prefix:
            prefix = list(output_prefix)
            grammar_prefix_is_valid = grammar.validate_tokens(prefix) == prefix
            if grammar_prefix_is_valid and grammar.accept_tokens(request_id, prefix):
                grammar_advanced = len(prefix)
            else:
                grammar_prefix_is_valid = False

        try:
            if grammar_prefix_is_valid:
                if grammar.is_terminated():
                    row.fill_(self._full_mask)
                else:
                    grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                row.zero_()

            for token_id in next_marker_token_ids:
                if token_id < 0 or token_id >= row.numel() * 32:
                    continue
                word_index, bit_index = divmod(token_id, 32)
                bit = 1 << bit_index
                if bit_index == 31:
                    bit = -(1 << 31)
                row[word_index] |= bit
        finally:
            if grammar_advanced:
                grammar.rollback(grammar_advanced)

    @staticmethod
    def _output_start_index(request: "Request") -> int:
        return min(request.num_prompt_tokens, len(request.all_token_ids))

    @staticmethod
    def _accept_validated_tokens(
        grammar: StructuredOutputGrammar,
        request_id: str,
        token_ids: list[int],
    ) -> int:
        if grammar.validate_tokens(token_ids) != token_ids:
            return 0
        return len(token_ids) if grammar.accept_tokens(request_id, token_ids) else 0

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

        # Covers both speculative decoding and diffusion LLMs (canvas_length).
        max_num_spec_tokens = self.vllm_config.num_speculative_tokens

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
                grammar = structured_output_request.grammar
                if TYPE_CHECKING:
                    assert isinstance(grammar, StructuredOutputGrammar)

                apply_bitmask = self.should_fill_bitmask(request)
                structured_req = request.structured_output_request
                reasoner = self._get_reasoner(request)
                if (
                    not apply_bitmask
                    and reasoner is not None
                    and structured_req is not None
                    and structured_req.reasoning_ended is None
                    and not self.enable_in_reasoning
                ):
                    output_start = self._output_start_index(request)
                    self._fill_undetermined_reasoning_bitmask(
                        grammar,
                        cumulative_index,
                        req_id,
                        reasoner,
                        request.all_token_ids[output_start:],
                    )
                else:
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
                grammar = structured_output_request.grammar
                if TYPE_CHECKING:
                    assert isinstance(grammar, StructuredOutputGrammar)
                apply_bitmask = self.should_fill_bitmask(request)

                reasoner = self._get_reasoner(request)
                detect_reasoning_end = (
                    not apply_bitmask
                    and reasoner is not None
                    and not self.enable_in_reasoning
                )
                reasoning_was_undetermined = (
                    structured_output_request.reasoning_ended is None
                )
                simulated_buf: list[int] | None = None
                history_len = 0

                state_advancements = 0
                post_reasoning_end_in_window = False
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for i, token in enumerate(req_tokens):
                    if (
                        not apply_bitmask
                        and detect_reasoning_end
                        and reasoning_was_undetermined
                        and reasoner is not None
                    ):
                        if simulated_buf is None:
                            output_start = self._output_start_index(request)
                            history = list(request.all_token_ids[output_start:])
                            history_len = len(history)
                            simulated_buf = history + list(req_tokens)
                        simulated_prefix = history + [
                            draft for draft in req_tokens[:i] if draft != -1
                        ]
                        self._fill_undetermined_reasoning_bitmask(
                            grammar,
                            cumulative_index,
                            req_id,
                            reasoner,
                            simulated_prefix,
                        )
                    else:
                        self._fill_bitmasks(
                            ((grammar, cumulative_index, apply_bitmask),)
                        )
                    advance_grammar = apply_bitmask
                    if token == -1:
                        apply_bitmask = False
                        advance_grammar = False
                    elif (
                        detect_reasoning_end
                        and reasoner is not None
                        and not apply_bitmask
                    ):
                        if simulated_buf is None:
                            output_start = self._output_start_index(request)
                            history = list(request.all_token_ids[output_start:])
                            history_len = len(history)
                            simulated_buf = history + list(req_tokens)
                        simulated = simulated_buf[: history_len + i + 1]
                        if reasoner.is_reasoning_end_streaming(simulated, [token]):
                            # Reasoning ended mid-window. Constrain the rest
                            # of the window via bitmask. Skip grammar advance
                            # through the marker (it is reasoning content);
                            # try to advance through subsequent drafts so the
                            # next bitmask row reflects the post-advance state,
                            # but tolerate rejection since those drafts predate
                            # the bitmask and are not guaranteed valid.
                            apply_bitmask = True
                            advance_grammar = False
                            post_reasoning_end_in_window = True
                            if reasoning_was_undetermined:
                                content_ids = reasoner.extract_content_ids(simulated)
                                if (
                                    content_ids == simulated
                                    and not grammar.is_terminated()
                                ):
                                    state_advancements += self._accept_validated_tokens(
                                        grammar, req_id, content_ids
                                    )
                                reasoning_was_undetermined = False
                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )
                    cumulative_index += 1
                # Diffusion LLMs don't sample a bonus token after the
                # scheduled positions, so skip its bitmask in that case.
                if not (self.vllm_config.model_config.is_diffusion and req_tokens):
                    # bonus_apply must be True when the bonus-row position
                    # should be grammar-constrained. Two triggers:
                    # - should_fill_bitmask(request): reasoning was already
                    #   over at step start (or no reasoner /
                    #   enable_in_reasoning).
                    # - apply_bitmask: reasoning ended mid-window in this
                    #   call and was flipped True after the marker;
                    #   should_fill_bitmask still returns False here because
                    #   reasoning_ended is only persisted later by
                    #   should_advance.
                    bonus_apply = self.should_fill_bitmask(request) or apply_bitmask
                    if (
                        not bonus_apply
                        and detect_reasoning_end
                        and reasoning_was_undetermined
                        and reasoner is not None
                    ):
                        if simulated_buf is None:
                            output_start = self._output_start_index(request)
                            history = list(request.all_token_ids[output_start:])
                        bonus_prefix = history + [
                            draft for draft in req_tokens if draft != -1
                        ]
                        self._fill_undetermined_reasoning_bitmask(
                            grammar,
                            cumulative_index,
                            req_id,
                            reasoner,
                            bonus_prefix,
                        )
                    else:
                        self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
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
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            return request.structured_output_request.reasoning_ended is True
        return True

    def should_advance(
        self,
        request: "Request",
        new_token_ids: list[int] | None = None,
    ) -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        reasoning_was_undetermined = structured_req.reasoning_ended is None

        # Check if reasoning ends in *this* step.
        # When the caller passes new_token_ids (the tokens that were just
        # appended this step), use it directly as the delta window. The
        # placeholder-derived fallback assumes num_output_placeholders ==
        # len(new_token_ids), which breaks under async scheduling + spec
        # decode when some drafts are rejected (#43388): the placeholder
        # count remains > 0 after the step and the computed delta window
        # starts past the reasoning-end marker.
        all_token_ids = request.all_token_ids
        if new_token_ids:
            # The tokens were already appended this step, so the step window
            # starts exactly len(new_token_ids) from the end.
            start = len(all_token_ids) - len(new_token_ids)
        else:
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            start = (
                delta_from
                if delta_from >= 0
                else max(len(all_token_ids) + delta_from, 0)
            )
        output_start = self._output_start_index(request)
        output_token_ids = all_token_ids[output_start:]
        step_start = max(start, output_start)
        output_delta_start = step_start - output_start
        if reasoner.is_reasoning_end_streaming(
            output_token_ids,
            itertools.islice(output_token_ids, output_delta_start, None),
        ):
            structured_req.reasoning_ended = True

            if reasoning_was_undetermined:
                content_ids = reasoner.extract_content_ids(list(output_token_ids))
                if content_ids == output_token_ids:
                    structured_req.reasoning_end_token_index = None
                    structured_req.deferred_grammar_start_index = output_start
                    return True
                structured_req.deferred_grammar_start_index = None

            # Record the boundary so the scheduler can exclude reasoning tokens.
            end_index = self._find_reasoning_end_index(
                reasoner, all_token_ids, output_start, step_start
            )

            structured_req.reasoning_end_token_index = end_index
            return True

        return False

    @staticmethod
    def _find_reasoning_end_index(
        reasoner: "ReasoningParser",
        all_token_ids: Sequence[int],
        output_start: int,
        start: int,
    ) -> int:
        """Locates the last reasoning token within ``all_token_ids[start:]``.

        Returns:
            The absolute index of the token at which
            ``is_reasoning_end_streaming`` first fires. Falls back to the
            final index when no single token triggers the detection (e.g.
            a multi-token marker only recognized on the full delta), which
            conservatively treats the whole step as reasoning content.
        """
        prefix = list(itertools.islice(all_token_ids, output_start, start))
        for idx in range(start, len(all_token_ids)):
            token = all_token_ids[idx]
            prefix.append(token)
            if reasoner.is_reasoning_end_streaming(prefix, [token]):
                return idx
        return len(all_token_ids) - 1

    def trim_reasoning_for_advance(
        self, request: "Request", new_token_ids: list[int]
    ) -> list[int]:
        """Drops reasoning content from tokens about to advance the grammar.

        When reasoning ends mid-step (see should_advance), the step's output
        still contains reasoning tokens up to and including the end marker.
        Those are not grammar content: feeding them to accept_tokens makes
        the grammar reject the marker and kills the request (#44006).

        Returns:
            The suffix of ``new_token_ids`` that follows the reasoning-end
            marker. Steps fully after the boundary are returned unchanged.
        """
        structured_req = request.structured_output_request
        if structured_req is None:
            return new_token_ids
        deferred_start = structured_req.deferred_grammar_start_index
        if deferred_start is not None:
            structured_req.deferred_grammar_start_index = None
            return list(request.all_token_ids[deferred_start:])
        end_idx = structured_req.reasoning_end_token_index
        if end_idx is None:
            return new_token_ids
        first_idx = len(request.all_token_ids) - len(new_token_ids)
        num_reasoning = end_idx + 1 - first_idx
        if num_reasoning <= 0:
            return new_token_ids
        return new_token_ids[num_reasoning:]

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
