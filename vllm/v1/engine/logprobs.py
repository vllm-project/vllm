# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import torch

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
    target_token_ids: list[int] | None = None
    """Target token IDs for score mode perplexity calculation.
    When provided, enables fast path processing of [N, 1] logprobs tensors."""

    prompt_logits: torch.Tensor | None = None
    """Raw logits for prompt positions when return_prompt_logits."""

    kld_result: tuple[float, int] | None = None
    """(kld_sum, kld_count) when kld_mode."""

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
            target_token_ids=request.target_token_ids,
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
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_list, tokens=token_ids
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
                # Apply UTF-8 correction within this position's token boundaries
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice, tokens=token_ids_list[pos]
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

    def _update_prompt_logprobs_fast_path(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
        target_token_ids: list[int],
    ) -> None:
        """Fast path for updating prompt logprobs in score mode.

        This method processes [N, 1] tensors directly without creating
        Python objects for all vocabulary tokens, significantly improving
        performance for perplexity calculation.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors with shape [N, 1]
          target_token_ids: list of target token IDs corresponding to
                           each position in the prompt
        """
        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors

        # Recover shapes - should be [N, 1] for score mode
        num_positions, num_logprobs_cols = logprobs.shape
        assert num_logprobs_cols == 1, (
            f"Fast path expects [N, 1] tensors, got [N, {num_logprobs_cols}]"
        )
        assert len(target_token_ids) == num_positions, (
            f"target_token_ids length ({len(target_token_ids)}) "
            f"must match num_positions ({num_positions})"
        )

        # Extract logprobs directly - they're already filtered to target tokens
        prompt_logprobs_list = logprobs.squeeze(-1).tolist()
        prompt_token_ranks = ranks.tolist()
        token_ids_list = token_ids.squeeze(-1).tolist()

        # Verify that token_ids match target_token_ids
        for pos, (token_id, target_id) in enumerate(
            zip(token_ids_list, target_token_ids)
        ):
            if token_id != target_id:
                raise ValueError(
                    f"Token ID mismatch at position {pos}: "
                    f"expected {target_id}, got {token_id}"
                )

        # Detokenize only the target tokens (non-incrementally)
        all_decoded_tokens: list[str] | None = (
            None
            if self.tokenizer is None
            else convert_ids_list_to_tokens(self.tokenizer, target_token_ids)
        )

        # Update with the Logprob container for each position
        for pos in range(num_positions):
            decoded_tokens_for_pos: list[str] | Iterable[None]
            if all_decoded_tokens is None:
                decoded_tokens_for_pos = NONES
            else:
                # For fast path, we only have one token per position
                decoded_tokens_slice = [all_decoded_tokens[pos]]
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice,
                    tokens=[target_token_ids[pos]],
                )

            # Update with the Logprob container for this pos
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                [target_token_ids[pos]],
                [prompt_logprobs_list[pos]],
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

    def _correct_decoded_token(self, idx: int, tokens: list[int]) -> str:
        assert self.tokenizer is not None, "self.tokenizer should not be None"

        # try with prev token id in same list
        if idx > 0:
            possible_decoded_token = self.tokenizer.decode(tokens[idx - 1 : idx + 1])
            if not possible_decoded_token.endswith("�"):
                return possible_decoded_token
        # try with previous logprob token id
        if self.logprobs:
            latest_token_id = next(iter(self.logprobs[-1]))

            decode_ids = [latest_token_id]
            if idx > 0:
                decode_ids.extend(tokens[idx - 1 : idx + 1])
            else:
                decode_ids.extend(tokens[idx : idx + 1])

            possible_decoded_token = self.tokenizer.decode(decode_ids)
            if not possible_decoded_token.endswith("�"):
                return possible_decoded_token

        # by default return empty string
        return ""

    def _verify_tokens(
        self, decoded_tokens_list: list[str], tokens: list[int]
    ) -> list[str]:
        corrected_decoded_token_map = dict()
        for idx, text in enumerate(decoded_tokens_list):
            if text.endswith("�"):
                # utf-8 char at the end means it's a potential unfinished byte sequence
                # from byte fallback tokenization.
                corrected_decoded_token_map[idx] = self._correct_decoded_token(
                    idx, tokens
                )

        for idx, text in corrected_decoded_token_map.items():
            decoded_tokens_list[idx] = text

        return decoded_tokens_list

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            # Use fast path if target_token_ids are available
            if self.target_token_ids is not None:
                self._update_prompt_logprobs_fast_path(
                    output.new_prompt_logprobs_tensors, self.target_token_ids
                )
            else:
                self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
        if output.new_prompt_logits is not None:
            self.prompt_logits = output.new_prompt_logits
        if output.kld_result is not None:
            self.kld_result = output.kld_result
