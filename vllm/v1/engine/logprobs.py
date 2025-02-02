# SPDX-License-Identifier: Apache-2.0

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from vllm.logger import init_logger
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_ids_list_to_tokens)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest

logger = init_logger(__name__)


@dataclass
class LogprobsProcessor:

    # Tokenizer for this request
    tokenizer: AnyTokenizer

    # Logprobs for this request
    logprobs: Optional[SampleLogprobs]
    prompt_logprobs: Optional[PromptLogprobs]
    cumulative_logprob: Optional[float]
    num_logprobs: Optional[int]
    num_prompt_logprobs: Optional[int]

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        num_logprobs = request.sampling_params.logprobs
        num_prompt_logprobs = request.sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.),
            logprobs=(None if num_logprobs is None else []),
            # NOTE: logprob of first prompt token is None.
            prompt_logprobs=(None if num_prompt_logprobs is None else [None]),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    def _update_sample_logprobs(
        self,
        token_ids_lst: List[List[int]],
        logprobs_lst: List[List[float]],
        ranks_lst: List[int],
    ) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          token_ids_lst: list of (topk + 1) token ids tensors at each pos
          logprobs_lst: list of (topk + 1) logprobs tensors at each pos
          ranks_lst: list of rank of each samples token

        Lists are empty if logprobs are not enabled for this req.
        """
        if self.num_logprobs is None:
            # Sample logprobs disabled for this request.
            return
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst,
                                             token_ids_lst):

            # Detokenize (non-incrementally).
            decoded_tokens = convert_ids_list_to_tokens(
                self.tokenizer, token_ids)

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob dictionary for this pos.
            self.logprobs.append(
                self._make_logprob_dict(
                    logprobs,
                    token_ids,
                    decoded_tokens,
                    rank,
                    self.num_logprobs,
                ))

    def _update_prompt_logprobs(
        self,
        token_ids: Optional[torch.Tensor],
        logprobs: Optional[torch.Tensor],
        ranks: Optional[torch.Tensor],
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        If prompt logprobs are enabled but prefill is completed, both
        arguments should be empty tensors.

        If prompt logprobs are disabled, both arguments should be `None`.

        Token rank = (index in logprob-sorted vocab vector) + 1

        Args:
          token_ids: (num prompt tokens-1) x (topk + 1) token ids tensor
                     `None` if prompt logprobs are disabled in this req
          logprobs: (num prompt tokens-1) x (topk + 1) logprobs tensor
          ranks: (num prompt_tokens-1) prompt token rank tensor

        Return:
          Prompt logprobs, if required for this request
        """
        num_prompt_logprobs = self.num_prompt_logprobs
        if num_prompt_logprobs is None:
            # Prompt logprobs disabled for this request
            return

        # Prompt logprobs are enabled.
        assert logprobs is not None
        assert token_ids is not None
        assert ranks is not None
        assert self.prompt_logprobs is not None

        # TODO(rob): can we avoid this case with a better
        # invariant from EngineCore?
        # Prompt logprobs are enabled but prefill is finished
        # so no more logprobs are streamed from EngineCore.
        if logprobs.numel() == 0:
            return

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = convert_ids_list_to_tokens(
            self.tokenizer,
            token_ids.flatten().tolist())

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the torch tensors.
        # TODO(rob): experiment with doing this in EngineCore?
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = decoded_tokens[offset:offset_end]

            # Update with the Logprob dictionary for this pos.
            self.prompt_logprobs.append(
                self._make_logprob_dict(prompt_logprobs[pos], token_ids[pos],
                                        decoded_tokens_for_pos,
                                        prompt_token_ranks[pos],
                                        num_prompt_logprobs))

    def pop_prompt_logprobs(self) -> Optional[PromptLogprobs]:
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
    def _make_logprob_dict(
        logprobs: List[float],
        logprob_token_ids: List[int],
        decoded_tokens: List[str],
        rank: int,
        num_logprobs: int,
    ) -> Dict[int, Logprob]:
        """Make a Logprob dictionary for a position.

        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          rank: rank of the sampled token
          num_logprobs: number of logprobs requested
            by the user (in addition to sampled logprob)

        Returns:
          Dict[token id, Logprob]
        """

        # We do not need a special case for the sampled token
        # being in the topk, since inserting duplicated data
        # into a dictionary twice is the same as doing it once.
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank, ), topk_ranks)

        return {
            token_id: Logprob(
                logprob=logprob,
                rank=rank,
                decoded_token=token,
            )
            for token_id, logprob, rank, token in zip(
                logprob_token_ids, logprobs, ranks, decoded_tokens)
        }

    def update_from_output(self, output: EngineCoreOutput) -> None:
        self._update_sample_logprobs(output.new_logprobs_token_ids,
                                     output.new_logprobs,
                                     output.new_sampled_token_ranks)

        self._update_prompt_logprobs(output.new_prompt_logprobs_token_ids,
                                     output.new_prompt_logprobs,
                                     output.new_prompt_token_ranks)
