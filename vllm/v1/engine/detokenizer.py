from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast

import torch

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.engine import DetokenizerRequest, EngineCoreOutput

logger = init_logger(__name__)


@dataclass
class IncrementalDetokenizer:

    # Generation data
    output_text: str
    tokens: List[str]
    token_ids: List[int]

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

    # TODO: Probably decouple these
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]

    # Tokenizer for this request
    tokenizer: AnyTokenizer

    # Logprobs for this request
    logprobs: SampleLogprobs
    prompt_logprobs: PromptLogprobs
    cumulative_logprob: float
    num_logprobs: int
    num_prompt_logprobs: int

    # Accounting for stop string buffering
    stop_buffer_length: int
    _last_output_text_offset: int = 0

    @property
    def output_token_ids(self) -> List[int]:
        assert len(self.token_ids) >= len(self.prompt_token_ids)
        return self.token_ids[len(self.prompt_token_ids):]

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: DetokenizerRequest,
    ) -> "IncrementalDetokenizer":

        tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer=tokenizer,
            prompt_ids=request.prompt_token_ids,
            skip_special_tokens=request.skip_special_tokens,
        )

        stops = request.stop
        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if stops and not request.include_stop_str_in_output:
            stop_buffer_length = max(len(s) for s in stops) - 1
        else:
            stop_buffer_length = 0

        return cls(
            output_text="",
            tokens=tokens,
            # Detokenizer mutates this list, so need a unique copy.
            # NOTE(Nick): could we take ownership of it though?
            token_ids=request.prompt_token_ids.copy(),
            stop=stops,
            include_stop_str_in_output=request.include_stop_str_in_output,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=request.
            spaces_between_special_tokens,
            output_kind=request.output_kind,
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            tokenizer=tokenizer,
            stop_buffer_length=stop_buffer_length,
            logprobs=[],
            prompt_logprobs=[],
            cumulative_logprob=0.,
            num_logprobs=request.logprobs,
            num_prompt_logprobs=request.prompt_logprobs,
        )

    def _pythonize_sequence_position(
        self,
        logprob_values: npt.NDArray,
        logprob_token_ids: npt.NDArray,
        detokenize: bool,
    ) -> Dict[int, Logprob]:
        """Pythonize the numpy (np) logprobs & token ids for a sequence position
        
        Outputs the OpenAI-API-compatible representation of the top tokens and
        their logprobs at a single position in a sequence.

        Optionally detokenize (compute logprob `decoded_token`)

        Args:
          logprob_values: np logprob values
          logprob_token_ids: np logprob token ids
          detokenize: if True, detokenize logprob top token ids

        Return:
          mapping from top token id to Logprob data structure
        """
        logprob_values = logprob_values.tolist()
        logprob_token_ids = logprob_token_ids.tolist()
        logprob_token_strs = (cast(List[Optional[str]],
                                   self._detokenize_ids(logprob_token_ids)) if
                              detokenize else [None] * len(logprob_token_ids))

        return {
            lpt: Logprob(lpv, (idx + 1), lpstr)
            for idx, (lpv, lpt, lpstr) in enumerate(
                zip(logprob_values, logprob_token_ids, logprob_token_strs))
        }

    def _make_sample_logprobs(
        self,
        sampled_token_ids: List[int],
        logprobs_token_ids_lst: List[torch.Tensor],
        logprobs_lst: List[torch.Tensor],
    ) -> SampleLogprobs:
        """Pythonize sample logprobs, maybe detokenize.
        
        Only Pythonizes sample logprobs computed in the current
        step. Has the side effect of updating the incremental detokenizer
        state by (1) appending the new sample logprobs to the list of what
        was computed for previously-sampled tokens, and (2) accumulating
        into the request's cumulative logprob value.ß

        Pythonization entails the conversion from a numpy (np)
        values/token ids representation to the more idiomatically
        Pythonic representation required by the OpenAI API,
        List[Dict[int,Logprob]]

        The Logprob.decoded_token field is only computed (detokenized
        from the associated top token id) if detokenize=True

        Args:
          new_sample_logprobs: List of (logprobs,logprob token ids) numpy array
                               tuples
          new_sample_token_ids: List of sample token ids
          detokenize: Logprob.decoded_token is computed if True, otherwise None
        
        Returns:
          Sample logprobs compute in this step, Pythonized and possibly
          detokenized
        """

        # NOTE(rob): the lists are of length > 1 if a single step
        # of engine core generates > 1 token (e.g. spec decoding).
        assert len(sampled_token_ids) == len(logprobs_token_ids_lst)
        assert len(sampled_token_ids) == len(logprobs_lst)
        output_list: SampleLogprobs = []
        for sampled_token_id, logprobs, logprobs_token_ids in zip(
            sampled_token_ids, logprobs_lst, logprobs_token_ids_lst):

            # Sampler cats the lps of sampled tok before the topk lps.
            assert sampled_token_id == logprobs_token_ids[0].item(), (
                "Sampler cats the sampled tokens logprobs in front of "
                f"the topk logprobs, but got {sampled_token_id=} and "
                f"{logprobs_token_ids[0].item()=}")

            # Pythonize the torch tensors..
            sampled_token_logprob = logprobs[0].item()
            topk_token_ids = logprobs_token_ids[1:].tolist()
            topk_logprobs = logprobs[1:].tolist()

            # Make the Logprob objects.
            # Detokenize *non-incrementally* for simplicity.
            decoded_tokens = self.tokenizer.batch_decode(
                topk_token_ids.reshape(-1,1))
            # torch.topk used to select lps returns them
            # in sorted order, so we can use idx for rank.
            topk_logprobs_dict = {
                topk_token_ids[idx]: Logprob(
                    logprob=topk_logprobs[idx], rank=idx,
                    decoded_token=decoded_tokens[idx],
                ) for idx in range(self.num_logprobs)
            }

            # If the sampled token was not in the topk, add it.
            if sampled_token_id not in topk_logprobs_dict:
                # TODO(rob): is rank for sample Logprob needed? 
                # it is not used in Chat Completions.
                token = self.tokenizer.decode(sampled_token_id)
                topk_logprobs_dict[sampled_token_id] = Logprob(
                    logprob=sampled_token_logprob,
                    rank=None, decoded_token=token)

            output_list.append(topk_logprobs_dict)

        return output_list

    def _pythonize_maybe_detokenize_prompt_logprobs_for_request(
        self,
        prompt_logprob_values: npt.NDArray,
        prompt_logprob_token_ids: npt.NDArray,
        detokenize: bool,
    ) -> PromptLogprobs:
        """Pythonize prompt logprobs, maybe detokenize.
        
        Only Pythonizes prompt logprobs computed in the current
        step. Has the side effect of updating the incremental detokenizer
        state by appending the new prompt logprobs to the list of what
        was computed for previous prompt chunks. Forces the first prompt
        logprob associated with the request to be `None`.

        Pythonization entails the conversion from a numpy (np)
        values/token ids representation to the more idiomatically
        Pythonic representation required by the OpenAI API,
        List[Dict[int,Logprob]]

        The Logprob.decoded_token field is only computed (detokenized
        from the associated top token id) if detokenize=True

        Args:
          prompt_logprob_values: num_chunk_tokens x num_prompt_logprobs np array
                                 of top token log probabilities
          prompt_logprob_token_ids: num_chunk_tokens x num_prompt_logprobs np
                                    array of top token ids
          detokenize: Logprob.decoded_token is computed if True, otherwise None
        
        Returns:
          Prompt logprobs compute in this step, Pythonized and possibly
          detokenized
        """
        logprob_cnt = self.max_request_prompt_logprobs
        prompt_logprobs: List[Optional[Dict[int, Logprob]]] = [
            self._pythonize_sequence_position(plp_tok_values,
                                              plp_tok_token_ids, detokenize)
            for plp_tok_values, plp_tok_token_ids in zip(
                # Slice out top prompt logprobs
                prompt_logprob_values[:, 0:logprob_cnt],
                prompt_logprob_token_ids[:, 0:logprob_cnt])
        ]
        if not self.request_prompt_logprobs:
            # Ensure that None is the first prompt logprob
            prompt_logprobs = cast(List[Optional[Dict[int, Logprob]]],
                                   [None]) + prompt_logprobs
        assert self.request_prompt_logprobs is not None
        self.request_prompt_logprobs.extend(prompt_logprobs)
        return prompt_logprobs

    def add_tokens(
        self,
        new_token_ids: List[int],
        finish_reason: Optional[str],
        stop_reason: Optional[Union[int, str, None]],
        new_logprobs_token_ids: Optional[List[torch.Tensor]],
        new_logprobs: Optional[List[torch.Tensor]],
        prompt_logprobs_token_ids: Optional[torch.Tensor],
        prompt_logprobs: Optional[torch.Tensor],
    ) -> Optional[RequestOutput]:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally
            2) Evaluate stop criteria
            3) Detokenize sample logprobs non-incrementally
            4) Detokenize prompt logprobs non-incrementally
            5) Update the `RequestOutput` object with new text
        """

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
        if new_logprobs:
            sample_logprobs = self._make_sample_logprobs(
                sampled_token_ids=new_token_ids,
                logprobs_token_ids_lst=new_logprobs_token_ids,
                logprobs=new_logprobs)
            self.logprobs.append(sample_logprobs)
            # TODO: update cumulative logprob.
            # self.cumulative_logprob

        # 4) Pythonize & detokenizer prompt logprobs.
        if prompt_logprobs:
            # EngineCore does not stream out partial prefill,
            # so all prompt logprobs come in one step.
            assert len(self.prompt_logprobs) == 0
            assert prompt_logprobs_token_ids is not None
            self.prompt_logprobs = self._make_prompt_logprobs(
                prompt_logprobs_token_ids,
                prompt_logprobs)

        # 5) Update the RequestOutput object with the new text.
        finished = bool(finish_reason)
        if self.output_kind == RequestOutputKind.FINAL_ONLY \
            and not finished:
            return None

        # Return just newly created items if DELTA.
        delta = self.output_kind == RequestOutputKind.DELTA
        output_text = self._get_next_output_text(finished, delta)
        token_ids = new_token_ids if delta else self.output_token_ids
        logprobs = sample_logprobs if delta else self.logprobs
        prompt_logprobs = sample_logprobs if delta else self.prompt_logprobs
        cumulative_logprob = self.cumulative_logprob

        request_output = RequestOutput.new(
            self.request_id,
            self.prompt,
            self.prompt_token_ids,
            output_text,
            token_ids,
            logprobs,
            prompt_logprobs,
            cumulative_logprob,
            finished,
        )

        if finished:
            completion_output = request_output.outputs[0]
            completion_output.finish_reason = finish_reason
            completion_output.stop_reason = stop_reason

        return request_output

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


class Detokenizer:

    def __init__(self,
                 tokenizer_name: str,
                 tokenizer_mode: str = "auto",
                 trust_remote_code: bool = False,
                 revision: Optional[str] = None):
        # TODO: once we support LoRA, we should should pass the tokenizer
        # here. We currently have two copies (this + in the LLMEngine).
        self.tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                       tokenizer_mode=tokenizer_mode,
                                       trust_remote_code=trust_remote_code,
                                       revision=revision)

        # Request id -> IncrementalDetokenizer
        self.request_states: Dict[str, IncrementalDetokenizer] = {}

    def is_request_active(self, request_id: str):
        return request_id in self.request_states

    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    def abort_requests(
        self,
        request_ids: Iterable[str],
    ) -> None:
        """Remove the request_ids from the Detokenizer."""

        for request_id in request_ids:
            self.request_states.pop(request_id, None)

    def add_request(
        self,
        request: DetokenizerRequest,
    ):
        """Add new request to the Detokenizer."""

        assert (request.request_id not in self.request_states)

        request_state = IncrementalDetokenizer.from_new_request(
            self.tokenizer, request)
        self.request_states[request.request_id] = request_state

    def step(
        self, encore_core_outputs: List[EngineCoreOutput]
    ) -> Tuple[List[RequestOutput], List[str]]:
        """Update state and request the RequestOutputs to the LLMEngine."""

        request_outputs: List[RequestOutput] = []
        requests_to_abort: List[str] = []
        for engine_core_output in encore_core_outputs:
            request_id = engine_core_output.request_id
            detokenizer = self.request_states.get(request_id)
            if detokenizer is None:
                # Ignore output for already-aborted request.
                continue

            # Detokenize and update state.
            request_output = detokenizer.add_tokens(
                new_sampled_token_ids=engine_core_output.new_token_ids,
                finish_reason=engine_core_output.finish_reason,
                stop_reason=engine_core_output.stop_reason,
                new_sample_logprobs=engine_core_output.logprobs,
                new_prompt_logprobs=engine_core_output.prompt_logprobs,
                new_prompt_logprob_token_ids=engine_core_output.
                prompt_logprobs_token_ids,
            )

            if request_output is not None:
                # Add to RequestOutputs list.
                request_outputs.append(request_output)

                # Free completed requests.
                if request_output.finished:
                    self.request_states.pop(request_id)
                    if not engine_core_output.finished:
                        requests_to_abort.append(request_id)

        # Return to EngineClient.
        return request_outputs, requests_to_abort
