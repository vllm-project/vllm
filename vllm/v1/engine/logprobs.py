# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

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


@dataclass
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

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(
                None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
            ),
            prompt_logprobs=(
                None
                if num_prompt_logprobs is None
                else create_prompt_logprobs(sampling_params.flat_logprobs)
            ),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
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

        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists

        for rank_np, logprobs_np, token_ids_np in zip(
            ranks_lst, logprobs_lst, token_ids_lst
        ):
            rank = rank_np.tolist()
            logprobs = logprobs_np.tolist()
            token_ids = token_ids_np.tolist()
            # Detokenize (non-incrementally).
            decoded_tokens: list[str] | Iterable[None]
            if self.tokenizer is None:
                decoded_tokens = NONES
            else:
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
                context_token_ids = self._get_sampled_context_ids(self.logprobs)
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_list,
                    tokens=token_ids,
                    context_token_ids=context_token_ids,
                )

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.logprobs,
                token_ids,
                logprobs,
                decoded_tokens,
                rank,
                self.num_logprobs,
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

        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        all_decoded_tokens: list[str] | None = (
            None
            if self.tokenizer is None
            else convert_ids_list_to_tokens(
                self.tokenizer, token_ids.flatten().tolist()
            )
        )

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids_list = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening and UTF-8 correction per position
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs

            decoded_tokens_for_pos: list[str] | Iterable[None]
            if all_decoded_tokens is None:
                decoded_tokens_for_pos = NONES
            else:
                # Extract decoded tokens for this position
                decoded_tokens_slice = all_decoded_tokens[offset:offset_end]
                # Context: preceding prompt tokens accumulated in
                # self.prompt_logprobs from previous loop iterations.
                context_token_ids = self._get_sampled_context_ids(self.prompt_logprobs)
                # Apply UTF-8 correction within this position's token boundaries
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice,
                    tokens=token_ids_list[pos],
                    context_token_ids=context_token_ids,
                )

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                token_ids_list[pos],
                prompt_logprobs[pos],
                decoded_tokens_for_pos,
                prompt_token_ranks[pos],
                self.num_prompt_logprobs,
            )

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """Pop and return all request prompt logprobs

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
        return plp

    @staticmethod
    def _get_sampled_context_ids(
        logprobs_source: SampleLogprobs | PromptLogprobs | None,
        max_context: int = 4,
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
            return [
                logprobs_source.token_ids[logprobs_source.start_indices[i]]
                for i in range(start, n)
                if logprobs_source.start_indices[i] < logprobs_source.end_indices[i]
            ]

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

        max_ctx = min(len(context_token_ids), 4)

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
