import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

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

    # Prompt tokens
    prompt_token_ids: List[int]

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
            prompt_token_ids=request.prompt_token_ids,
            cumulative_logprob=(None if num_logprobs is None else 0.),
            logprobs=(None if num_logprobs is None else []),
            prompt_logprobs=(None if num_prompt_logprobs is None else []),
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

        for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst,
                                             token_ids_lst):

            assert len(logprobs) > 0, (
                "Always generate at least 1 logprobs if enabled.")

            # Detokenize (non-incrementally).
            decoded_tokens = convert_ids_list_to_tokens(
                self.tokenizer, token_ids)

            # NOTE(rob): this relies on the invariant that the
            # sampler concatenates the sampled token logprob
            # in front of the topk logprobs.
            sampled_token_logprob = logprobs[0]
            assert self.cumulative_logprob is not None
            self.cumulative_logprob += sampled_token_logprob

            # Ranks of the tokens.
            topk_ranks = range(1, self.num_logprobs + 1)
            ranks = itertools.chain((rank, ), topk_ranks)

            # NOTE(rob): we do not need a special case for the
            # sampled token being in the topk, since we insert
            # the same key into the dictionary twice.
            logprobs_dict = self._make_logprob_dict(logprobs, token_ids,
                                                    decoded_tokens, ranks)
            self.logprobs.append(logprobs_dict)

    def _update_prompt_logprobs(
        self,
        token_ids: Optional[torch.Tensor],
        prompt_logprobs: Optional[torch.Tensor],
        prompt_token_ranks: Optional[torch.Tensor],
        prompt_token_ids_lst: List[int],
    ) -> None:
        """Incorporate prompt logprobs from this step, if they exist.

        If prompt logprobs are enabled for this request and EngineCore
        prefilled the prompt or a chunk of the prompt in this step,
        both arguments should be non-empty lists.

        If prompt logprobs are enabled but prefill is completed, both
        arguments should be empty lists.

        If prompt logprobs are disabled, both arguments should be `None`.

        Token rank = (index in logprob-sorted vocab vector) + 1

        Args:
          token_ids: (num prompt tokens-1) x (topk + 1) token ids tensor
                     `None` if prompt logprobs are disabled in this req
          prompt_logprobs: (num prompt tokens-1) x (topk + 1) logprobs tensor
          prompt_token_ranks: (num prompt_tokens-1) prompt token rank tensor
          prompt_token_ids_lst: (num prompt tokens)-length list of prompt
                                token ids

        Return:
          Prompt logprobs, if required for this request
        """
        num_prompt_logprobs = self.num_prompt_logprobs
        if num_prompt_logprobs is None:
            # Prompt logprobs disabled for this request
            return

        assert prompt_logprobs is not None
        assert token_ids is not None
        assert prompt_token_ranks is not None

        if prompt_logprobs.numel() == 0:
            # Prompt logprobs are enabled for this request but prefill
            # is finished and no more logprobs are being streamed from
            # engine core.
            return

        # Prompt logprobs are enabled & engine core is streaming prompt
        # logprobs, in one or more chunks.
        assert self.prompt_logprobs is not None

        if not self.prompt_logprobs:
            self.prompt_logprobs = [None]

        if num_prompt_logprobs:
            # We need to also include topk logprob tokens.
            prompt_token_ids_lst = token_ids.flatten().tolist()

        # Detokenize non-incrementally.
        # NOTE(rob): the output is flattened:
        # [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = convert_ids_list_to_tokens(self.tokenizer,
                                                    prompt_token_ids_lst)

        # Make Logprob for each token.
        num_chunk_tokens, decoded_tokens_stride = prompt_logprobs.shape
        for tok_idx in range(num_chunk_tokens):
            # Iterate over prefill chunk

            decoded_tokens_offset = tok_idx * decoded_tokens_stride
            prompt_token_id = token_ids[tok_idx, 0].item()
            prompt_token_rank = prompt_token_ranks[tok_idx].item()
            ranks: Iterable[int]

            if num_prompt_logprobs:
                ranks = range(1, num_prompt_logprobs + 1)
                topk_token_ids_tensor = token_ids[tok_idx, 1:]
                if prompt_token_id in topk_token_ids_tensor:
                    # Slice off the prompt token first element since
                    # it's already in the subsequent top-k tokens.
                    token_ids_list = topk_token_ids_tensor.tolist()
                    logprobs = prompt_logprobs[tok_idx, 1:].tolist()
                    # Shift decoded token starting offset by one.
                    decoded_tokens_offset += 1
                else:
                    token_ids_list = token_ids[tok_idx, :].tolist()
                    logprobs = prompt_logprobs[tok_idx, :].tolist()
                    # First token in token_ids/logprobs is the sampled token.
                    ranks = itertools.chain((prompt_token_rank, ), ranks)
            else:
                token_ids_list = (prompt_token_id, )
                logprobs = (prompt_logprobs[tok_idx, 0].item(), )
                ranks = (prompt_token_rank, )

            prompt_logprobs_dict = self._make_logprob_dict(
                logprobs,
                token_ids_list,
                decoded_tokens[decoded_tokens_offset:],
                ranks,
            )

            self.prompt_logprobs.append(prompt_logprobs_dict)

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
        ranks: Iterable[int],
    ) -> Dict[int, Logprob]:
        """Make a Logprob dictionary for a position.

        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          ranks: ranks for each of the tokens

        Returns:
          Dict[top token id, Logprob]
        """

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
        # Sample Logprobs.
        self._update_sample_logprobs(output.new_logprobs_token_ids,
                                     output.new_logprobs,
                                     output.new_sampled_token_ranks)

        # Prompt Logprobs.
        self._update_prompt_logprobs(output.new_prompt_logprobs_token_ids,
                                     output.new_prompt_logprobs,
                                     output.new_prompt_token_ranks,
                                     self.prompt_token_ids)
