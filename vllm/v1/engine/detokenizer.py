from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally,
    detokenize_non_incrementally)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor_utils import RequestState

logger = init_logger(__name__)


@dataclass
class DetokenizerOutput:
    output_text: str
    token_ids: List[int]
    logprobs: Optional[SampleLogprobs]
    prompt_logprobs: Optional[PromptLogprobs]
    cumulative_logprob: Optional[float]
    finished: bool
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


@dataclass
class IncrementalDetokenizer:

    # Generation data
    output_text: str
    tokens: List[str]
    token_ids: List[int]
    prompt_len: int

    # Stop strings
    stop: List[str]
    include_stop_str_in_output: bool

    # Metadata for incremental detokenization
    prefix_offset: int
    read_offset: int

    # Parameters for detokenization
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind

    # Tokenizer for this request
    tokenizer: AnyTokenizer

    # Logprobs for this request
    logprobs: Optional[SampleLogprobs]
    prompt_logprobs: Optional[PromptLogprobs]
    cumulative_logprob: Optional[float]
    num_logprobs: int
    num_prompt_logprobs: int

    # Accounting for stop string buffering
    stop_buffer_length: int
    _last_output_text_offset: int = 0

    @property
    def output_token_ids(self) -> List[int]:
        return self.token_ids[self.prompt_len:]

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":

        tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer=tokenizer,
            prompt_ids=request.prompt_token_ids,
            skip_special_tokens=request.sampling_params.skip_special_tokens,
        )

        stops = request.sampling_params.stop
        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if stops and not request.sampling_params.include_stop_str_in_output:
            stop_buffer_length = max(len(s) for s in stops) - 1
        else:
            stop_buffer_length = 0

        logprobs = request.sampling_params.logprobs
        prompt_logprobs = request.sampling_params.prompt_logprobs
        return cls(
            output_text="",
            tokens=tokens,
            # Detokenizer mutates this list, so need a unique copy.
            # NOTE(Nick): could we take ownership of it though?
            token_ids=request.prompt_token_ids.copy(),
            stop=stops,
            include_stop_str_in_output=request.sampling_params.
            include_stop_str_in_output,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=request.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=request.sampling_params.
            spaces_between_special_tokens,
            output_kind=request.sampling_params.output_kind,
            prompt_len=len(request.prompt_token_ids),
            tokenizer=tokenizer,
            stop_buffer_length=stop_buffer_length,
            cumulative_logprob=(0. if logprobs else None),
            logprobs=([] if logprobs else None),
            prompt_logprobs=([] if prompt_logprobs else None),
            num_prompt_logprobs=(prompt_logprobs or 0),
            num_logprobs=(logprobs or 0),
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
            decoded_tokens = detokenize_non_incrementally(
                self.tokenizer, topk_token_ids)

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
        decoded_tokens = detokenize_non_incrementally(self.tokenizer,
                                                      token_ids)

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
        request_state: RequestState,
    ) -> Optional[DetokenizerOutput]:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.
            3) Detokenize sample logprobs non-incrementally.
            4) Detokenize prompt logprobs non-incrementally.
            5) Make the `RequestOutput` object with new text.
        """

        new_token_ids = output.new_token_ids
        finish_reason = output.finish_reason
        stop_reason = output.stop_reason
        new_logprobs_token_ids = output.logprobs_token_ids
        new_logprobs = output.logprobs
        new_prompt_logprobs_token_ids = output.prompt_logprobs_token_ids
        new_prompt_logprobs = output.prompt_logprobs

        # 1) Detokenize the new token ids incrementally.
        # TODO(woosuk): This method becomes very inefficient when the number of
        # new_token_ids is more than 1. We need to optimize this.
        decoded_text = ""
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            (new_tokens, new_decoded_token_text, prefix_offset,
             read_offset) = detokenize_incrementally(
                 tokenizer=self.tokenizer,
                 all_input_ids=self.token_ids,
                 prev_tokens=self.tokens,
                 prefix_offset=self.prefix_offset,
                 read_offset=self.read_offset,
                 skip_special_tokens=self.skip_special_tokens,
                 spaces_between_special_tokens=self.
                 spaces_between_special_tokens,
             )

            self.tokens.extend(new_tokens)
            self.prefix_offset = prefix_offset
            self.read_offset = read_offset
            self.output_text += new_decoded_token_text

            decoded_text += new_decoded_token_text

        # 2) Evaluate stop criteria.
        if self.stop:
            stop = StopChecker.check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(decoded_text),
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_str, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]
                finish_reason = "stop"  # TODO: use constant
                stop_reason = stop_str

        # 3) Make Sample Logprobs.
        logprobs = self._update_sample_logprobs(
            new_token_ids,
            new_logprobs_token_ids,
            new_logprobs,
        )

        # 4) Make Prompt Logprobs.
        prompt_logprobs = self._update_prompt_logprobs(
            new_prompt_logprobs_token_ids, new_prompt_logprobs,
            request_state.prompt_token_ids)

        # 5) Makes the RequestOutput object with the new text.
        finished = bool(finish_reason)
        if self.output_kind == RequestOutputKind.FINAL_ONLY \
            and not finished:
            return None

        delta = self.output_kind == RequestOutputKind.DELTA
        output_text = self._get_next_output_text(finished, delta)
        token_ids = new_token_ids if delta else self.output_token_ids
        logprobs = logprobs if delta else self.logprobs
        prompt_logprobs = prompt_logprobs if delta else self.prompt_logprobs

        return DetokenizerOutput(
            output_text=output_text,
            token_ids=token_ids,
            finished=finished,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            cumulative_logprob=self.cumulative_logprob,
        )

    def _get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            return self.output_text[:-buffer_length] if buffer_length else (
                self.output_text)
        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""
