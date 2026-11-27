# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from dataclasses import dataclass

import numpy as np

from vllm.logger import init_logger
from vllm.logprobs import (
    FlatLogprobs,
    PromptLogprobs,
    SampleLogprobs,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)
from vllm.tokenizers.detokenizer_utils import (
    TokenizerLike,
    convert_ids_list_to_tokens,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

NONES = itertools.repeat(None)
_MAX_UTF8_CONTEXT = 4


@dataclass(slots=True)
class LogprobsProcessor:
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: TokenizerLike | None

    # Logprobs for this request
    logprobs: SampleLogprobs | None
    prompt_logprobs: PromptLogprobs | None
    cumulative_logprob: float | None
    num_logprobs: int | None
    num_prompt_logprobs: int | None
    # [num_scored_rows, num_token_ids], set once on the final prefill chunk.
    prompt_token_id_logprobs: np.ndarray | None = None
    # False keeps sample logprobs without decoded token strings.
    detokenize_sample_logprobs: bool = True
    # The unscored prompt prefix may contain the first bytes of a scored token.
    prompt_logprob_seed: tuple[int, ...] = ()

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.num_logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        prompt_logprobs = None
        prompt_logprob_seed: tuple[int, ...] = ()
        if num_prompt_logprobs is not None:
            prompt_logprobs = create_prompt_logprobs(sampling_params.flat_logprobs)
            # Positions up to prompt_logprob_start are not scored.
            for _ in range(sampling_params.prompt_logprob_start or 0):
                prompt_logprobs.append(None)
            if tokenizer is not None and request.prompt_token_ids is not None:
                end = len(prompt_logprobs)
                start = max(0, end - _MAX_UTF8_CONTEXT)
                prompt_logprob_seed = tuple(
                    map(int, request.prompt_token_ids[start:end])
                )
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(
                None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
            ),
            prompt_logprobs=prompt_logprobs,
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
            detokenize_sample_logprobs=sampling_params._detokenize_logprobs,
            prompt_logprob_seed=prompt_logprob_seed,
        )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """
        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids, logprobs, ranks, _ = logprobs_lists
        # Sampler puts the sampled logprob in first.
        for sampled_token_logprob in logprobs[:, 0].tolist():
            self.cumulative_logprob += sampled_token_logprob
        self._append_rows(
            self.logprobs,
            self.num_logprobs,
            token_ids,
            logprobs,
            ranks,
            self.detokenize_sample_logprobs,
        )

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """
        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks, *_ = prompt_logprobs_tensors
        self._append_rows(
            self.prompt_logprobs,
            self.num_prompt_logprobs,
            token_ids.numpy(),
            logprobs.numpy(),
            ranks.numpy(),
            detokenize=True,
        )

    def _append_rows(
        self,
        container: SampleLogprobs | PromptLogprobs,
        num_logprobs: int,
        token_ids: np.ndarray,
        logprobs: np.ndarray,
        ranks: np.ndarray,
        detokenize: bool,
    ) -> None:
        """Appends `[n, S]` engine rows, cut to the requested width, with the
        `[n]` ranks of their sampled tokens."""
        width = token_ids.shape[1] if num_logprobs == -1 else num_logprobs + 1
        token_ids, logprobs = token_ids[:, :width], logprobs[:, :width]
        decoded = (
            self._decode_rows(token_ids, container)
            if detokenize and self.tokenizer is not None
            else None
        )
        if isinstance(container, FlatLogprobs):
            container.append_rows(token_ids, logprobs, ranks, decoded)
            return
        for i, (row_ids, row_logprobs, rank) in enumerate(
            zip(token_ids.tolist(), logprobs.tolist(), ranks.tolist())
        ):
            append_logprobs_for_next_position(
                container,
                row_ids,
                row_logprobs,
                NONES if decoded is None else decoded[i * width : (i + 1) * width],
                rank,
                num_logprobs,
            )

    def _decode_rows(
        self,
        token_ids: np.ndarray,
        logprobs_source: SampleLogprobs | PromptLogprobs,
    ) -> list[str]:
        """The decoded tokens of `[n, S]` rows about to be appended to
        `logprobs_source`, flattened. Rows with a byte-fallback piece are
        corrected using the tokens of the positions before them: the
        source's last positions and this batch's earlier rows."""
        assert self.tokenizer is not None
        width = token_ids.shape[1]
        decoded = convert_ids_list_to_tokens(self.tokenizer, token_ids.ravel().tolist())
        for row in sorted(
            {i // width for i, t in enumerate(decoded) if t.endswith("�")}
        ):
            in_batch = token_ids[max(0, row - _MAX_UTF8_CONTEXT) : row, 0].tolist()
            needed = _MAX_UTF8_CONTEXT - len(in_batch)
            previous = self._get_sampled_context_ids(logprobs_source, needed)
            missing = needed - len(previous)
            if missing and logprobs_source is self.prompt_logprobs:
                previous = [*self.prompt_logprob_seed[-missing:], *previous]
            context = previous + in_batch
            begin, end = row * width, (row + 1) * width
            decoded[begin:end] = self._verify_tokens(
                decoded_tokens_list=decoded[begin:end],
                tokens=token_ids[row].tolist(),
                context_token_ids=context,
            )
        return decoded

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """Pop and return all request prompt logprobs.

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.

        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
            self.prompt_logprob_seed = ()
        return plp

    def pop_prompt_token_id_logprobs(self) -> np.ndarray | None:
        """Pop and return the fixed-ID prompt scores."""
        scores = self.prompt_token_id_logprobs
        self.prompt_token_id_logprobs = None
        return scores

    @staticmethod
    def _get_sampled_context_ids(
        logprobs_source: SampleLogprobs | PromptLogprobs | None,
        max_context: int = _MAX_UTF8_CONTEXT,
    ) -> list[int]:
        """Extract recent sampled token IDs from a logprobs source.

        The sampled (or prompt) token at each position is the first
        entry, since it is always inserted first by
        append_logprobs_for_next_position.

        Args:
            logprobs_source: The logprobs container to extract from.
            max_context: Maximum number of preceding tokens to return.
                4 is sufficient for any UTF-8 multi-byte sequence.

        Returns:
            List of sampled token IDs, oldest first, most recent last.

        """
        if not logprobs_source:
            return []

        n = len(logprobs_source)
        start = max(0, n - max_context)

        # Efficient path for FlatLogprobs: access token_ids directly.
        if isinstance(logprobs_source, FlatLogprobs):
            return logprobs_source.first_token_ids(start, n)

        # list[dict] path
        result: list[int] = []
        for i in range(start, n):
            entry = logprobs_source[i]
            if entry is not None:
                result.append(next(iter(entry)))
        return result

    def _correct_decoded_token(
        self, token_id: int, context_token_ids: list[int]
    ) -> str:
        """Correct a decoded token that contains the replacement character.

        When byte-fallback tokenization splits multi-byte UTF-8
        characters across tokens, individual token decoding produces
        the replacement character U+FFFD. This method uses preceding
        sampled tokens as context to reconstruct the correct text.

        Args:
            token_id: The single token ID to correct.
            context_token_ids: Preceding sampled token IDs in sequential
                order (oldest first). These are the actual tokens in
                the generated sequence, NOT top-k alternatives.

        Returns:
            The corrected decoded string, or empty string if the byte
            sequence is genuinely incomplete at this point.

        """
        assert self.tokenizer is not None

        max_ctx = min(len(context_token_ids), _MAX_UTF8_CONTEXT)

        for num_ctx in range(1, max_ctx + 1):
            context = context_token_ids[-num_ctx:]
            full_decoded = self.tokenizer.decode(context + [token_id])

            if full_decoded.endswith("�"):
                continue

            # Find the boundary between "clean" context tokens and
            # byte-fallback tokens that are part of the same incomplete
            # sequence. Byte-fallback context tokens returned "" when
            # they were processed, so their text must be attributed to
            # this completing token.
            clean_end = len(context)
            for j in range(len(context) - 1, -1, -1):
                if self.tokenizer.decode([context[j]]).endswith("�"):
                    clean_end = j
                else:
                    break

            # Decode only the clean (non-byte-fallback) prefix.
            if clean_end > 0:
                clean_prefix = self.tokenizer.decode(context[:clean_end])
            else:
                clean_prefix = ""

            if full_decoded.startswith(clean_prefix):
                return full_decoded[len(clean_prefix) :]

            # Tokenizer normalization may cause prefix mismatch.
            # Find the longest common prefix between them.
            common_len = 0
            for a, b in zip(clean_prefix, full_decoded):
                if a != b:
                    break
                common_len += 1
            return full_decoded[common_len:]

        return ""

    def _verify_tokens(
        self,
        decoded_tokens_list: list[str],
        tokens: list[int],
        context_token_ids: list[int] | None = None,
    ) -> list[str]:
        """Verify and correct decoded tokens with replacement characters.

        Args:
            decoded_tokens_list: Decoded token strings to verify.
            tokens: Token IDs corresponding to decoded_tokens_list.
                These are alternatives at the SAME position (e.g.
                [sampled, top1, top2]), NOT sequential tokens.
            context_token_ids: Preceding sampled token IDs providing
                sequential context. If None, extracted from
                self.logprobs.

        """
        if context_token_ids is None:
            context_token_ids = self._get_sampled_context_ids(self.logprobs)

        corrected_decoded_token_map = dict()
        for idx, text in enumerate(decoded_tokens_list):
            if text.endswith("�"):
                # Replacement char at the end means a potential
                # unfinished byte sequence from byte-fallback
                # tokenization. Correct each token independently
                # using only the sequential context.
                corrected_decoded_token_map[idx] = self._correct_decoded_token(
                    tokens[idx], context_token_ids
                )

        for idx, text in corrected_decoded_token_map.items():
            decoded_tokens_list[idx] = text

        return decoded_tokens_list

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
        if output.prompt_token_id_logprobs is not None:
            self.prompt_token_id_logprobs = output.prompt_token_id_logprobs.numpy()
