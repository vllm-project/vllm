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

                # Determine where the reasoning_end token falls within
                # the speculative tokens (if applicable). This handles
                # two scenarios:
                # 1. Reasoning not yet ended (apply_bitmask=False): the
                #    reasoning_end token may appear mid-speculation;
                #    positions after it need grammar constraints.
                # 2. Reasoning already ended (apply_bitmask=True): the
                #    spec tokens from the previous step may contain the
                #    reasoning_end token; the grammar must NOT accept
                #    reasoning tokens (up to and including the end
                #    marker).
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                reasoning_end_idx: int | None = None
                if (
                    self.reasoner is not None
                    and not self.enable_in_reasoning
                    and req_tokens
                ):
                    reasoning_end_idx = self.find_reasoning_end_in_tokens(
                        list(req_tokens)
                    )

                state_advancements = 0
                for tok_idx, token in enumerate(itertools.chain(req_tokens, (-1,))):
                    # Determine whether grammar applies at this position.
                    # Tokens up to and including reasoning_end are
                    # unconstrained; tokens after are grammar-constrained.
                    if reasoning_end_idx is not None:
                        is_post_reasoning = tok_idx > reasoning_end_idx
                        pos_apply_bitmask = is_post_reasoning
                        # Once we pass the reasoning_end position, mark
                        # reasoning as ended so subsequent steps know.
                        if (
                            is_post_reasoning
                            and not structured_output_request.reasoning_ended
                        ):
                            structured_output_request.reasoning_ended = True
                    else:
                        pos_apply_bitmask = apply_bitmask

                    self._fill_bitmasks(
                        ((grammar, cumulative_index, pos_apply_bitmask),)
                    )
                    if token == -1:
                        # Stop advancing the grammar once we hit a
                        # padding token.
                        pos_apply_bitmask = False
                    if pos_apply_bitmask and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        assert accepted, (
                            token,
                            req_id,
                            scheduled_spec_decode_tokens,
                        )
                        state_advancements += 1
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
        if self.reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step.
        # When new_token_ids is provided (e.g. from speculative decoding
        # verification), check those tokens for reasoning end. This handles
        # the case where the reasoning_end token appears within a batch of
        # accepted tokens — we need to return True so the caller can feed
        # the post-reasoning tokens to the grammar.
        if new_token_ids is not None:
            all_token_ids = request.all_token_ids
            if self.reasoner.is_reasoning_end_streaming(all_token_ids, new_token_ids):
                structured_req.reasoning_ended = True
                return True
        else:
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            all_token_ids = request.all_token_ids
            if self.reasoner.is_reasoning_end_streaming(
                all_token_ids, all_token_ids[delta_from:]
            ):
                # Reasoning just ended, so we shouldn't advance til
                # next pass
                structured_req.reasoning_ended = True

        return False

    def find_reasoning_end_in_tokens(self, token_ids: list[int]) -> int | None:
        """Find the index of the reasoning-end token within a token list.

        Uses is_reasoning_end_streaming to check progressively longer
        prefixes, supporting multi-token end markers.

        Returns:
            The index of the last token of the reasoning-end marker,
            or None if not found.
        """
        if self.reasoner is None or self.enable_in_reasoning:
            return None

        for i, token in enumerate(token_ids):
            # Check if reasoning ends at position i by testing the
            # prefix up to and including this token.
            prefix = token_ids[: i + 1]
            if self.reasoner.is_reasoning_end_streaming(prefix, [token]):
                return i
        return None

    def get_tokens_after_reasoning(
        self,
        request: "Request",
        new_token_ids: list[int],
    ) -> list[int]:
        """Return the suffix of new_token_ids that comes after reasoning end.

        If reasoning was already ended before these tokens, returns all of
        them. If reasoning ends within these tokens, returns only the tokens
        after the end marker. If reasoning doesn't end, returns empty list.
        """
        if self.reasoner is None or self.enable_in_reasoning:
            return new_token_ids

        assert request.structured_output_request is not None
        structured_req = request.structured_output_request

        if structured_req.reasoning_ended:
            # Check whether reasoning_ended was set *before* this call
            # (i.e. from a previous step) vs just now by should_advance.
            # If it was set before, all tokens are post-reasoning.
            # We need to check if the end marker is actually in new_token_ids.
            idx = self.find_reasoning_end_in_tokens(new_token_ids)
            if idx is not None:
                # The end marker is in new_token_ids — reasoning ended
                # within this batch. Return tokens after the end marker.
                return new_token_ids[idx + 1 :]
            else:
                # End marker not in these tokens — reasoning ended before
                # this batch. All tokens are post-reasoning.
                return new_token_ids

        # Reasoning hasn't ended — no tokens to process
        return []

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
