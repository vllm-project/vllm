from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import AnyTokenizer, detokenize
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest

logger = init_logger(__name__)


@dataclass
class LogprobsOutput:
    logprobs: Optional[SampleLogprobs]
    prompt_logprobs: Optional[PromptLogprobs]
    cumulative_logprob: Optional[float]


@dataclass
class LogprobsProcessor:

    # Tokenizer for this request
    tokenizer: AnyTokenizer

    # Request output kind
    output_kind: RequestOutputKind

    # Prompt tokens
    prompt_token_ids: List[int]

    # Logprobs for this request
    logprobs: Optional[SampleLogprobs]
    prompt_logprobs: Optional[PromptLogprobs]
    cumulative_logprob: Optional[float]
    num_logprobs: int
    num_prompt_logprobs: int

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
            output_kind=request.sampling_params.output_kind,
            prompt_token_ids=request.prompt_token_ids,
            cumulative_logprob=(0. if num_logprobs else None),
            logprobs=([] if num_logprobs else None),
            prompt_logprobs=([] if num_prompt_logprobs else None),
            num_prompt_logprobs=(num_prompt_logprobs or 0),
            num_logprobs=(num_logprobs or 0),
        )

    def _update_sample_logprobs(
        self,
        sampled_token_ids: List[int],
        token_ids_lst: List[torch.Tensor],
        sample_logprobs_lst: List[torch.Tensor],
    ) -> Optional[SampleLogprobs]:
        """Incorporate sample logprobs from this step, if they exist.

        Lists are only of length >1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          sampled_token_ids: list of int token ids
          token_ids_list: list of (topk + 1) token ids tensors at each pos;
                          `None` if sample logprobs are disabled in this req
          sample_logprobs: list of (topk + 1) logprobs tensors at each pos;
                          `None` if sample logprobs are disabled in this req

        Return:
          Sample logprobs, if required for this request
        """
        if self.num_logprobs == 0:
            # Sample logprobs disabled for this request
            return None
        assert self.logprobs is not None

        for sampled_token_id, logprobs, token_ids in zip(
                sampled_token_ids, sample_logprobs_lst, token_ids_lst):

            # Split into sampled vs top_k.
            assert sampled_token_id == token_ids[0].item(), (
                "Sampler concats the sampled token logprob in front of "
                f"the topk logprobs, but got {sampled_token_id=} and "
                f"{token_ids[0].item()=}")
            sampled_token_logprob = logprobs[0].item()
            topk_token_ids = token_ids[1:]
            topk_logprobs = logprobs[1:]

            # Detokenize non-incrementally.
            decoded_tokens = detokenize(self.tokenizer, topk_token_ids)

            # Make the dict of top-token Logprob objects associated with the
            # current sequence offset
            if sampled_token_id in topk_token_ids:
                pos_logprobs_dict = self._make_pos_logprob_dict(
                    topk_logprobs.tolist(), topk_token_ids.tolist(),
                    decoded_tokens, self.num_logprobs)
            else:
                # If the sampled token is not one of the top tokens
                # at this sequence offset, inject the sampled token
                # & its Logprob instance into the dict
                sample_logprob_obj = Logprob(
                    logprob=sampled_token_logprob,
                    decoded_token=self.tokenizer.decode(sampled_token_id))
                pos_logprobs_dict = self._make_pos_logprob_dict(
                    topk_logprobs.tolist(), topk_token_ids.tolist(),
                    decoded_tokens, self.num_logprobs,
                    (sampled_token_id, sample_logprob_obj))

            self.logprobs.append(pos_logprobs_dict)
            self.cumulative_logprob += sampled_token_logprob

        # Return just the newly generated sample logprobs.
        num_new_tokens = len(sampled_token_ids)
        return self.logprobs[-num_new_tokens:]

    def _update_prompt_logprobs(
        self,
        token_ids: Optional[torch.Tensor],
        prompt_logprobs: Optional[torch.Tensor],
        prompt_token_ids_lst: List[int],
    ) -> Optional[PromptLogprobs]:
        """Incorporate prompt logprobs from this step, if they exist.

        If prompt logprobs are enabled for this request and EngineCore
        prefilled the prompt or a chunk of the prompt in this step,
        both arguments should be non-empty lists. 

        If prompt logprobs are enabled but prefill is completed, both
        arguments should be empty lists.

        If prompt logprobs are disabled, both arguments should be `None`.

        Args:
          token_ids: (num prompt tokens-1) x (topk + 1) token ids tensor
                     `None` if prompt logprobs are disabled in this req
          prompt_logprobs: (num prompt tokens-1) x (topk + 1) logprobs tensor
          prompt_token_ids_lst: (num prompt tokens)-length list of prompt
                                token ids

        Return:
          Prompt logprobs, if required for this request
        """
        if self.num_prompt_logprobs == 0:
            # Prompt logprobs disabled for this request
            return None
        assert prompt_logprobs is not None
        assert token_ids is not None
        if prompt_logprobs.numel() == 0:
            # Prompt logprobs are enabled for this request but prefill
            # is finished and no more logprobs are being streamed from
            # engine core
            return []
        # Prompt logprobs are enabled & engine core is streaming prompt
        # logprobs, in one or more chunks.
        assert self.prompt_logprobs is not None

        if len(self.prompt_logprobs) == 0:
            self.prompt_logprobs = [None]

        # Detokenize non-incrementally.
        # NOTE(rob): the output is flattened:
        # [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = detokenize(self.tokenizer, token_ids)

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
            topk_token_ids = token_ids[tok_idx, 1:]
            topk_logprobs = prompt_logprobs[tok_idx, 1:]
            decoded_tokens_offset = tok_idx * decoded_tokens_stride + 1

            # Make the dict of top-token Logprob objects associated with the
            # current prompt offset
            if prompt_token_id in topk_token_ids:
                self.prompt_logprobs.append(
                    self._make_pos_logprob_dict(
                        topk_logprobs.tolist(),
                        topk_token_ids.tolist(),
                        # Deal with the flattening from above.
                        decoded_tokens[decoded_tokens_offset:],
                        self.num_prompt_logprobs,
                    ))
            else:
                # If the prompt token is not one of the top tokens
                # at this prompt offset, inject the prompt token
                # & its Logprob instance into the dict
                prompt_logprob_obj = Logprob(
                    logprob=prompt_token_logprob,
                    decoded_token=self.tokenizer.decode(prompt_token_id))
                self.prompt_logprobs.append(
                    self._make_pos_logprob_dict(
                        topk_logprobs.tolist(), topk_token_ids.tolist(),
                        decoded_tokens[decoded_tokens_offset:],
                        self.num_prompt_logprobs,
                        (prompt_token_id, prompt_logprob_obj)))
        return self.prompt_logprobs

    @staticmethod
    def _make_pos_logprob_dict(
        logprobs: List[float],
        logprob_token_ids: List[int],
        decoded_tokens: List[str],
        num_logprobs: int,
        special_token_id_logprob: Optional[Tuple[int, Logprob]] = None,
    ) -> Dict[int, Logprob]:
        """Make a Logprob dictionary for a position in the sequence.
        
        Returns a dictionary mapping top token ids to Logprob data
        structures. Each Logprob data structure includes log probability,
        decoded token, and rank (index+1). The size of the dict returned
        will be be num_logprobs.

        If the special token (sampled token or prompt token associated
        with the current sequence position) is not among the top logprobs,
        then special_token_id_logprob = (special_token_id,logprob) must be
        provided; an additional dictionary entry mapping special_token_id -> 
        logprob will be injected with rank equal to num_logprobs + 1 
        (special_token_id must be lowest-rank if we are having to inject it.)
        Note that the size of the dict returned will then be num_logprobs + 1.

        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          num_logprobs: number of top tokens
          special_token_id_logprob: (optional) tuple of
                                    (special_token_id,logprob) associated with
                                    sampled token or prompt token

        Returns:
          Dict[top token id, Logprob]; num_logprobs or num_logprobs+1
          keys in total
        
        """
        # Sampler uses torch.topk() which sorts so the
        # index in lists is equivalent to rank-1.
        logprobs_dict = {
            logprob_token_ids[idx]: Logprob(
                logprob=logprobs[idx],
                rank=idx + 1,
                decoded_token=decoded_tokens[idx],
            )
            for idx in range(num_logprobs)
        }

        # Inject special token Logprob if necessary
        if special_token_id_logprob:
            special_token_id = special_token_id_logprob[0]
            special_logprob_obj = special_token_id_logprob[1]
            assert special_token_id is not None
            assert special_logprob_obj is not None
            special_logprob_obj.rank = num_logprobs + 1
            logprobs_dict[special_token_id] = special_logprob_obj

        return logprobs_dict

    def update_from_output(
        self,
        output: EngineCoreOutput,
    ) -> Optional[LogprobsOutput]:
        """
        Update RequestState for the request_id by:
        """
        new_token_ids = output.new_token_ids
        new_logprobs_token_ids = output.new_logprobs_token_ids
        new_logprobs = output.new_logprobs
        new_prompt_logprobs_token_ids = output.new_prompt_logprobs_token_ids
        new_prompt_logprobs = output.new_prompt_logprobs

        # 1) Make Sample Logprobs, if requested
        logprobs = self._update_sample_logprobs(
            new_token_ids,
            new_logprobs_token_ids,
            new_logprobs,
        )

        # 4) Make Prompt Logprobs.
        prompt_logprobs = self._update_prompt_logprobs(
            new_prompt_logprobs_token_ids, new_prompt_logprobs,
            self.prompt_token_ids)

        # 5) Makes the LogprobsOutput object with the new text.
        finished = bool(output.finish_reason)
        if self.output_kind == RequestOutputKind.FINAL_ONLY \
            and not finished:
            return None
        delta = self.output_kind == RequestOutputKind.DELTA
        logprobs = logprobs if delta else self.logprobs
        prompt_logprobs = prompt_logprobs if delta else self.prompt_logprobs

        return LogprobsOutput(
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            cumulative_logprob=self.cumulative_logprob,
        )
