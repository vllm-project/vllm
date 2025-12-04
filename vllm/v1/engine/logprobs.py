# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.logprobs import (
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
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, TrackedLogprobsLists

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

    # Tracked logprobs for specific tokens
    # Format: {token_id: [logprob_step0, logprob_step1, ...]}
    tracked_logprobs: dict[int, list[float]] | None
    track_token_ids: list[int] | None

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
        track_token_ids = sampling_params.track_token_ids

        # Initialize tracked_logprobs dict if track_token_ids is specified
        # Use `is not None` to correctly handle empty list (which should return {})
        tracked_logprobs: dict[int, list[float]] | None = None
        if track_token_ids is not None:
            tracked_logprobs = {token_id: [] for token_id in track_token_ids}

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
            tracked_logprobs=tracked_logprobs,
            track_token_ids=track_token_ids,
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
            decoded_tokens = (
                NONES
                if self.tokenizer is None
                else (convert_ids_list_to_tokens(self.tokenizer, token_ids))
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

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = (
            None
            if self.tokenizer is None
            else (
                convert_ids_list_to_tokens(self.tokenizer, token_ids.flatten().tolist())
            )
        )

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = (
                NONES if decoded_tokens is None else decoded_tokens[offset:offset_end]
            )

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                token_ids[pos],
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

    def _update_tracked_logprobs(
        self,
        tracked_logprobs_lists: TrackedLogprobsLists,
    ) -> None:
        """Update with tracked logprobs from EngineCore.

        Args:
            tracked_logprobs_lists: The tracked logprobs for this request.
                Shape: [num_generated_tokens, num_tracked_tokens]
                In speculative decoding, num_generated_tokens can be > 1.
        """
        if self.tracked_logprobs is None or self.track_token_ids is None:
            return

        logprobs_array = tracked_logprobs_lists.logprobs
        token_ids = tracked_logprobs_lists.token_ids

        # logprobs_array shape: [num_tokens, num_tracked_tokens]
        # We need to match the tracked token IDs from the merged set
        # to our specific track_token_ids
        token_id_to_idx = {tid: idx for idx, tid in enumerate(token_ids)}

        # Process each generated token (may be > 1 in speculative decoding)
        num_tokens = logprobs_array.shape[0]
        for token_id in self.track_token_ids:
            if token_id in token_id_to_idx:
                idx = token_id_to_idx[token_id]
                # Append logprobs for all generated tokens
                for row_idx in range(num_tokens):
                    logprob_value = float(logprobs_array[row_idx, idx])
                    self.tracked_logprobs[token_id].append(logprob_value)

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
        if output.new_tracked_logprobs is not None:
            self._update_tracked_logprobs(output.new_tracked_logprobs)
