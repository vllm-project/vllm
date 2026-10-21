# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import chain
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
from vllm.v1.structured_output.utils import strip_speculative_padding

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request
else:
    torch = LazyLoader("torch", globals(), "torch")


logger = init_logger(__name__)


class ConstraintBounds(NamedTuple):
    """Where tokens start being fed to the grammar and being masked."""

    grammar_start: int
    constraint_start: int


_CONSTRAINED_FROM_START = ConstraintBounds(0, 0)
_NO_SPEC_TOKENS = ConstraintBounds(1, 1)


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
            stop_token_ids = (
                request.sampling_params.all_stop_token_ids
                if request.sampling_params is not None
                else None
            )
            return self.backend.compile_grammar(
                request_type, grammar_spec, stop_token_ids=stop_token_ids
            )
        except Exception:
            logger.exception(
                "Failed to compile grammar for request %s", request.request_id
            )
            raise

    def _fill_bitmasks(
        self,
        batch: Iterable[tuple[StructuredOutputGrammar, int, bool, tuple[int, ...]]],
    ) -> None:
        """Fill one bitmask row per entry.

        A row the grammar does not constrain allows every token except
        `masked_ids`, the implicit reasoning ends masked while reasoning.
        """
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask, masked_ids in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)
                if masked_ids:
                    row = self._grammar_bitmask[index].numpy().view(np.uint32)
                    for token_id in masked_ids:
                        row[token_id >> 5] &= ~np.uint32(1 << (token_id & 31))

    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool, tuple[int, ...]]]
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def _masked_reasoning_ends(self, request: "Request") -> tuple[int, ...]:
        """Implicit reasoning-end tokens the grammar rejects as first content.

        Such a token (e.g. a tool-call start) ends reasoning and is also the
        first content token, so sampling one where the grammar cannot start
        with it would make the output invalid. These stay masked while the
        request reasons. The grammar does not advance before reasoning ends,
        so the set is fixed per request.
        """
        structured_req = request.structured_output_request
        assert structured_req is not None
        if structured_req.masked_reasoning_end_ids is None:
            masked: tuple[int, ...] = ()
            reasoner = self._get_reasoner(request)
            if isinstance(reasoner, ParserEngineReasoningAdapter):
                grammar = structured_req.grammar
                assert isinstance(grammar, StructuredOutputGrammar)
                masked = tuple(
                    token_id
                    for token_id in sorted(reasoner.implicit_reasoning_end_token_ids)
                    if not grammar.validate_tokens([token_id])
                )
            structured_req.masked_reasoning_end_ids = masked
        return structured_req.masked_reasoning_end_ids

    def _get_constraint_bounds(
        self,
        request: "Request",
        spec_tokens: Sequence[int],
        spec_tokens_committed: bool = False,
    ) -> ConstraintBounds:
        """Return the index into `spec_tokens` where tokens should start being
        constrained by the grammar.
        Assumes `spec_tokens` are stripped of -1 padding tokens.
        Returns `len(spec_tokens) + 1` if no token should be constrained after
        accepting all `spec_tokens`.
        This is `constraint_start`; `grammar_start` is one less if reasoning ends
        implicitly on a content token (e.g. a tool-call start).

        `spec_tokens_committed` is True if spec_tokens is already in
        `request.all_token_ids`.
        """
        if self.enable_in_reasoning:
            return _CONSTRAINED_FROM_START

        structured_req = request.structured_output_request
        assert structured_req is not None
        if structured_req.reasoning_ended:
            return _CONSTRAINED_FROM_START

        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return _CONSTRAINED_FROM_START

        if structured_req.reasoning_ended is None:
            # This should be removed here, but since `openai_gptoss`
            # is an independent code path, it is kept for now.
            # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
            # it can be removed.
            if reasoner.is_reasoning_end(request.prompt_token_ids or []):
                return _CONSTRAINED_FROM_START
            structured_req.reasoning_ended = False

        num_spec_tokens = len(spec_tokens)
        if num_spec_tokens <= 0:
            return _NO_SPEC_TOKENS

        # Engine-based parsers also report whether the end token is content.
        if (
            isinstance(reasoner, ParserEngineReasoningAdapter)
            and reasoner.reasoning_end_token_ids
        ):
            end = reasoner.find_reasoning_end(spec_tokens)
            if end is not None:
                constraint_start = end.offset + 1
                if end.implicit:
                    return ConstraintBounds(end.offset, constraint_start)
                return ConstraintBounds(constraint_start, constraint_start)

        # Fallback: the grammar starts after the end token.
        # TODO: Build a read-only Sequence view over all_token_ids
        # instead of copying the entire all_token_ids into input_ids.
        if spec_tokens_committed:
            if not spec_tokens or not reasoner.is_reasoning_end_streaming(
                request.all_token_ids, spec_tokens
            ):
                return ConstraintBounds(num_spec_tokens + 1, num_spec_tokens + 1)

            input_ids = request.all_token_ids.copy()
            delta_ids = list(spec_tokens)
        else:
            input_ids = request.all_token_ids.copy()
            input_ids.extend(spec_tokens)
            if not reasoner.is_reasoning_end_streaming(input_ids, spec_tokens):
                return ConstraintBounds(num_spec_tokens + 1, num_spec_tokens + 1)
            delta_ids = list(spec_tokens)

        for i in range(num_spec_tokens - 1, 0, -1):
            input_ids.pop()
            delta_ids.pop()
            if not reasoner.is_reasoning_end_streaming(input_ids, delta_ids):
                return ConstraintBounds(i + 1, i + 1)
        return ConstraintBounds(1, 1)

    def trim_spec_tokens(self, request: "Request", spec_tokens: list[int]) -> list[int]:
        """Trim structured-output padding and grammar-invalid suffixes."""
        if not request.use_structured_output:
            # Synchronous unstructured decoding uses -1 values as scheduler-side
            # markers for speculative slots, so preserve their length.
            return spec_tokens

        spec_tokens = strip_speculative_padding(spec_tokens)

        grammar_start, _ = self._get_constraint_bounds(request, spec_tokens)
        if grammar_start >= len(spec_tokens):
            return spec_tokens

        structured_req = request.structured_output_request
        if TYPE_CHECKING:
            assert structured_req is not None
        grammar = structured_req.grammar
        if TYPE_CHECKING:
            assert isinstance(grammar, StructuredOutputGrammar)
        if grammar.is_terminated():
            return spec_tokens[:grammar_start]
        prefix = spec_tokens[:grammar_start]
        len_validated = len(grammar.validate_tokens(spec_tokens[grammar_start:]))
        return prefix + spec_tokens[grammar_start : grammar_start + len_validated + 1]

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
        is_diffusion = self.vllm_config.model_config.is_diffusion
        mask_stride = max_num_spec_tokens + (0 if is_diffusion else 1)

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
        # Each request owns a fixed-width block of `mask_stride` rows.
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

                apply_bitmask = (
                    self._get_constraint_bounds(request, ()).constraint_start == 0
                )
                masked_ids = (
                    () if apply_bitmask else self._masked_reasoning_ends(request)
                )
                batch.append((grammar, cumulative_index, apply_bitmask, masked_ids))
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
            for grammar_idx, req_id in enumerate(structured_output_request_ids):
                base_index = grammar_idx * mask_stride
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                grammar = structured_output_request.grammar
                if TYPE_CHECKING:
                    assert isinstance(grammar, StructuredOutputGrammar)

                req_tokens = scheduled_spec_decode_tokens.get(req_id, list())
                unpadded_tokens = strip_speculative_padding(req_tokens)
                grammar_start, constraint_start = self._get_constraint_bounds(
                    request, unpadded_tokens
                )
                # Rows before constraint_start are still reasoning.
                masked_ids = (
                    self._masked_reasoning_ends(request) if constraint_start else ()
                )
                # Diffusion LLMs don't sample a bonus token after the
                # scheduled positions, so skip its bitmask in that case.
                tokens_to_mask = (
                    chain(req_tokens, (-1,))
                    if not (is_diffusion and req_tokens)
                    else req_tokens
                )

                state_advancements = 0
                padding_or_grammar_invalid = False
                for i, token in enumerate(tokens_to_mask):
                    if padding_or_grammar_invalid:
                        self._fill_bitmasks(((grammar, base_index + i, False, ()),))
                        continue

                    apply_bitmask = i >= constraint_start
                    row_masked = masked_ids if i < constraint_start else ()
                    self._fill_bitmasks(
                        ((grammar, base_index + i, apply_bitmask, row_masked),)
                    )
                    if token == -1:
                        padding_or_grammar_invalid = True
                    elif i >= grammar_start:
                        if not grammar.is_terminated() and grammar.accept_tokens(
                            req_id, [token]
                        ):
                            state_advancements += 1
                        else:
                            padding_or_grammar_invalid = True

                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        num_masks = len(structured_output_request_ids) * mask_stride
        if num_masks < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:num_masks]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def accept_tokens(self, request: "Request", new_token_ids: list[int]) -> bool:
        """Advance grammar with accepted tokens. Returns False on rejection."""
        if not request.use_structured_output or not new_token_ids:
            return True

        structured_req = request.structured_output_request
        if TYPE_CHECKING:
            assert structured_req is not None
        grammar = structured_req.grammar
        if TYPE_CHECKING:
            assert isinstance(grammar, StructuredOutputGrammar)

        grammar_start, _ = self._get_constraint_bounds(
            request, new_token_ids, spec_tokens_committed=True
        )
        # Early return only when no reasoning-end token is present.
        # Otherwise, latch structured_req.reasoning_ended.
        if grammar_start > len(new_token_ids):
            return True
        structured_req.reasoning_ended = True
        return grammar.accept_tokens(request.request_id, new_token_ids[grammar_start:])

    def clear_backend(self) -> None:
        # Finish tasks that can still use the backend before releasing it.
        if not self.vllm_config.model_config.skip_tokenizer_init:
            self.executor.shutdown(wait=True, cancel_futures=True)
        if (
            self.fill_bitmask_parallel_threshold
            < self.vllm_config.scheduler_config.max_num_seqs
        ):
            self.executor_for_fillmask.shutdown(wait=True, cancel_futures=True)
        if self.backend is not None:
            self.backend.destroy()
            self.backend = None
