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
        sample_logprobs_lst: List[List[float]],
        sampled_token_ranks_lst: List[int],
    ) -> None:
        """Incorporate sample logprobs from this step, if they exist.

        Lists are only of length >1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Token rank = (index in logprob-sorted vocab vector) + 1

        Args:
          token_ids_lst: list of (topk + 1) token ids tensors at each pos;
                          `None` if sample logprobs are disabled in this req
          sample_logprobs_lst: list of (topk + 1) logprobs tensors at
                          each pos; `None` if sample logprobs are
                          disabled in this req
          sampled_token_ranks_lst: list of individual sampled token ranks
                                   for each sampled token
        Return:
          Sample logprobs, if required for this request
        """
        num_logprobs = self.num_logprobs
        if num_logprobs is None:
            # Sample logprobs disabled for this request
            return
        assert self.logprobs is not None

        # If False, only sampled token logprobs are
        # needed for this request
        need_non_sampled_logprobs = num_logprobs > 0

        for sampled_token_rank, logprobs, token_ids in zip(
                sampled_token_ranks_lst, sample_logprobs_lst, token_ids_lst):

            # First token in `token_ids` is sampled token
            sampled_token_id = token_ids[0]

            # Split into sampled vs top_k.
            sampled_token_logprob = logprobs[0]
            topk_token_ids = token_ids[1:] if need_non_sampled_logprobs else []
            topk_logprobs = logprobs[1:] if need_non_sampled_logprobs else []

            # Detokenize non-incrementally.
            decoded_tokens = convert_ids_list_to_tokens(
                self.tokenizer,
                topk_token_ids) if need_non_sampled_logprobs else []

            # Make the dict of top-token Logprob objects associated with the
            # current sequence offset
            if need_non_sampled_logprobs and sampled_token_id in topk_token_ids:
                pos_logprobs_dict = self._make_pos_logprob_dict(
                    topk_logprobs, topk_token_ids, decoded_tokens,
                    num_logprobs)
            else:
                # If the sampled token is not one of the top tokens
                # at this sequence offset, inject the sampled token
                # & its Logprob instance into the dict
                sample_logprob_obj = Logprob(
                    logprob=sampled_token_logprob,
                    decoded_token=convert_ids_list_to_tokens(
                        self.tokenizer, [sampled_token_id])[0],
                    rank=sampled_token_rank)
                pos_logprobs_dict = self._make_pos_logprob_dict(
                    topk_logprobs, topk_token_ids, decoded_tokens,
                    num_logprobs, (sampled_token_id, sample_logprob_obj))

            self.logprobs.append(pos_logprobs_dict)
            assert self.cumulative_logprob is not None
            self.cumulative_logprob += sampled_token_logprob

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

        # If False, only prompt token logprobs are
        # needed for this request
        need_non_prompt_logprobs = num_prompt_logprobs > 0

        if prompt_logprobs.numel() == 0:
            # Prompt logprobs are enabled for this request but prefill
            # is finished and no more logprobs are being streamed from
            # engine core
            return
        # Prompt logprobs are enabled & engine core is streaming prompt
        # logprobs, in one or more chunks.
        assert self.prompt_logprobs is not None

        if len(self.prompt_logprobs) == 0:
            self.prompt_logprobs = [None]

        # Detokenize non-incrementally.
        # NOTE(rob): the output is flattened:
        # [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = convert_ids_list_to_tokens(
            self.tokenizer,
            token_ids.tolist()) if need_non_prompt_logprobs else []

        # Make Logprob for each token.
        num_chunk_tokens, decoded_tokens_stride = prompt_logprobs.shape
        prompt_idx = len(self.prompt_logprobs)
        for tok_idx, prompt_token_id in zip(range(num_chunk_tokens),
                                            prompt_token_ids_lst[prompt_idx:]):
            # Iterate over prefill chunk
            assert prompt_token_id
            assert prompt_token_id == token_ids[tok_idx, 0].item(), (
                "Sampler concats the prompt token logprob in front of "
                f"the topk logprobs, but got {prompt_token_id=} and "
                f"{token_ids[tok_idx, 0].item()=}")
            # Split into prompt token vs top_k.
            prompt_token_logprob = prompt_logprobs[tok_idx, 0].item()
            prompt_token_rank = prompt_token_ranks[tok_idx].item()
            topk_token_ids = token_ids[
                tok_idx, 1:].tolist() if need_non_prompt_logprobs else []
            topk_logprobs = prompt_logprobs[
                tok_idx, 1:].tolist() if need_non_prompt_logprobs else []
            decoded_tokens_offset = tok_idx * decoded_tokens_stride + 1

            # Make the dict of top-token Logprob objects associated with the
            # current prompt offset
            if need_non_prompt_logprobs and prompt_token_id in topk_token_ids:
                self.prompt_logprobs.append(
                    self._make_pos_logprob_dict(
                        topk_logprobs,
                        topk_token_ids,
                        # Deal with the flattening from above.
                        decoded_tokens[decoded_tokens_offset:],
                        num_prompt_logprobs,
                    ))
            else:
                # If the prompt token is not one of the top tokens
                # at this prompt offset, inject the prompt token
                # & its Logprob instance into the dict
                prompt_logprob_obj = Logprob(
                    logprob=prompt_token_logprob,
                    decoded_token=convert_ids_list_to_tokens(
                        self.tokenizer, [prompt_token_id])[0],
                    rank=prompt_token_rank)
                self.prompt_logprobs.append(
                    self._make_pos_logprob_dict(
                        topk_logprobs, topk_token_ids,
                        decoded_tokens[decoded_tokens_offset:],
                        num_prompt_logprobs,
                        (prompt_token_id, prompt_logprob_obj)))

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
        # Pop all prompt logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    @staticmethod
    def _make_pos_logprob_dict(
        logprobs: List[float],
        logprob_token_ids: List[int],
        decoded_tokens: List[str],
        ranks: Iterable[int],
    ) -> Dict[int, Logprob]:
        """Make a Logprob dictionary for a position in the sequence.
        
        Returns a dictionary mapping top token ids to Logprob data
        structures. Each Logprob data structure includes log probability,
        decoded token, and rank.

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
        """
        Update Logprobs from the engine output.
        """

        # 1) Make Sample Logprobs, if requested.
        self._update_sample_logprobs(output.new_logprobs_token_ids,
                                     output.new_logprobs,
                                     output.new_sampled_token_ranks)

        # 4) Make Prompt Logprobs.
        self._update_prompt_logprobs(output.new_prompt_logprobs_token_ids,
                                     output.new_prompt_logprobs,
                                     output.new_prompt_token_ranks,
                                     self.prompt_token_ids)
