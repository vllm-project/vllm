from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

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

AnyLogprobs = Union[Optional[SampleLogprobs], Optional[PromptLogprobs]]


@dataclass
class IncrementalDetokenizer:

    # Generation data
    output_text: str
    tokens: List[str]
    token_ids: List[int]
    request_logprobs: Optional[SampleLogprobs]
    request_prompt_logprobs: Optional[PromptLogprobs]
    request_cumulative_logprob: Optional[float]

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

    # Maximum number of sample logprobs for this request
    max_request_sample_logprobs: Optional[int]

    # Maximum number of prompt logprobs for this request
    max_request_prompt_logprobs: Optional[int]

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

        # Logprobs & prompt logprobs settings
        do_request_logprobs = (request.logprobs is not None
                               and request.logprobs > 0)

        do_request_prompt_logprobs = (request.prompt_logprobs is not None
                                      and request.prompt_logprobs > 0)

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
            max_request_sample_logprobs=request.logprobs,
            max_request_prompt_logprobs=request.prompt_logprobs,
            request_logprobs=[] if do_request_logprobs else None,
            request_prompt_logprobs=[] if do_request_prompt_logprobs else None,
            request_cumulative_logprob=0 if do_request_logprobs else None)

    def _detokenize_ids(
        self,
        token_id_list: int,
        skip_special_tokens=False,
    ) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(
            token_id_list, skip_special_tokens=skip_special_tokens)

    def _pythonize_sequence_position(
        self,
        logprob_values: npt.NDArray,
        logprob_token_ids: npt.NDArray,
        detokenize: bool,
    ) -> Dict[int, Logprob]:
        """Pythonize the numpy (np) logprobs & token ids for a sequence position
        
        Optionally detokenize (compute logprob decoded token str)

        Args:
          logprob_values: np logprob values
          logprob_token_ids: np logprob token ids
          detokenize: if True, compute logprob decoded token str,
                      (o/w decoded_token=None)

        Return:
          mapping from top token id to Logprob data structure
        """
        logprob_values = logprob_values.tolist()
        logprob_token_ids = logprob_token_ids.tolist()
        logprob_token_strs = (self._detokenize_ids(logprob_token_ids) if
                              detokenize else [None] * len(logprob_token_ids))

        return {
            lpt: Logprob(lpv, (idx + 1), lpstr)
            for idx, (lpv, lpt, lpstr) in enumerate(
                zip(logprob_values, logprob_token_ids, logprob_token_strs))
        }

    def _pythonize_maybe_detokenize_sample_logprobs_for_request(
        self,
        new_logprobs: List[Tuple[npt.NDArray, npt.NDArray]],
        new_token_ids: List[int],
        detokenize: bool,
    ) -> Tuple[SampleLogprobs, float]:
        """Pythonize sample logprobs, maybe detokenize.
        
        Pythonization entails the conversion from a numpy (np)
        values/token ids representation to the more idiomatically
        Pythonic representation required by the OpenAI API,
        List[Dict[int,Logprob]]

        The Logprob.decoded_token field is only computed (detokenized
        from the associated top token id) if detokenize=True

        Also computes cumulative logprob.

        Args:
          new_logprobs: List of (logprobs,logprob token ids) numpy array tuples
          detokenize: Logprob.decoded_token is computed if True, otherwise None
        
        Returns:
          Sample logprobs, Pythonized and possibly detokenized
        """
        max_logprobs = self.max_request_sample_logprobs
        for (logprob_values,
             logprob_token_ids), token_id in zip(new_logprobs, new_token_ids):
            # Only keep the number of logprobs specified by the request
            # (plus possibly the sampled token id & its logprob)
            logprob_cnt = max_logprobs
            if token_id not in logprob_token_ids[0:logprob_cnt]:
                # Sampled token is not in the in the top logprobs;
                # inject it & resort, ensuring that excess logprobs
                # not requested by the user have -inf probability
                logprob_values[max_logprobs:-1] = float('-inf')
                # Get indices that would sort logprob_values in descending order
                indices = np.argsort(logprob_values)[::-1]
                # Use these indices to reorder logprob_values and
                # logprob_token_ids
                logprob_values = logprob_values[indices]
                logprob_token_ids = logprob_token_ids[indices]
                # There will be one more logprob than the user requested
                logprob_cnt = max_logprobs + 1

            new_pythonized_logprobs = self._pythonize_sequence_position(
                logprob_values[0:logprob_cnt],
                logprob_token_ids[0:logprob_cnt], detokenize)
            self.request_logprobs.append(new_pythonized_logprobs)
            self.request_cumulative_logprob += new_pythonized_logprobs[
                token_id].logprob

    def _pythonize_maybe_detokenize_prompt_logprobs_for_request(
        self,
        prompt_logprob_values: Optional[npt.NDArray],
        prompt_logprob_token_ids: Optional[npt.NDArray],
        detokenize: bool,
    ) -> PromptLogprobs:
        # Construct prompt logprobs, under the condition that
        # prompt logprobs were requested & a nonzero number of
        # prompt tokens were computed in this step for this request.
        #
        # Note that this scenario returns an EngineCoreOutput which
        # is empty except for the prompt logprobs which were
        # computed for these prompt tokens.
        logprob_cnt = self.max_request_prompt_logprobs
        prompt_logprobs = [
            self._pythonize_sequence_position(plp_tok_values,
                                              plp_tok_token_ids, detokenize)
            for plp_tok_values, plp_tok_token_ids in zip(
                # Slice out top prompt logprobs
                prompt_logprob_values[:, 0:logprob_cnt],
                prompt_logprob_token_ids[:, 0:logprob_cnt])
        ]

        if not self.request_prompt_logprobs:
            # Ensure that None is the first prompt logprob
            prompt_logprobs = [None] + prompt_logprobs

        self.request_prompt_logprobs.extend(prompt_logprobs)

    def add_tokens(
        self,
        new_token_ids: List[int],
        new_logprobs: Optional[List[Tuple[npt.NDArray, npt.NDArray]]],
        new_prompt_logprobs: Optional[npt.NDArray],
        new_prompt_logprob_token_ids: Optional[npt.NDArray],
        finish_reason: Optional[str],
        stop_reason: Optional[str],
    ) -> Optional[RequestOutput]:
        """
        Update RequestState for the request_id by:
            1) If necessary, detokenize logprobs *non*-incrementally
            2) If necessary, detokenize prompt logprobs *non*-incrementally
            3) Detokenize the new token ids incrementally.
            4) Update the RequestOutput with the new text.
        """

        do_request_sample_logprobs = new_logprobs is not None and len(
            new_logprobs) > 0
        assert not do_request_sample_logprobs or len(new_logprobs) == len(
            new_token_ids)
        do_request_prompt_logprobs = new_prompt_logprobs is not None and len(
            new_prompt_logprobs) > 0
        assert (not do_request_prompt_logprobs
                or new_prompt_logprob_token_ids is not None)

        # 1) If required, Pythonize & detokenize sample logprobs
        if do_request_sample_logprobs:
            self._pythonize_maybe_detokenize_sample_logprobs_for_request(
                new_logprobs, new_token_ids, detokenize=True)

        # 2) If necessary, detokenize prompt logprobs incrementally
        if do_request_prompt_logprobs:
            self._pythonize_maybe_detokenize_prompt_logprobs_for_request(
                new_prompt_logprobs,
                new_prompt_logprob_token_ids,
                detokenize=True)

        # 3) Detokenize the new token ids incrementally. If necessary,
        #    detokenize logprobs.
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
                _, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]
                finish_reason = "stop"  # TODO: use constant

        # TODO: handle stop_token_ids here too?

        # 3) Update the RequestOutput object with the new text.
        finished = bool(finish_reason)
        if self.output_kind == RequestOutputKind.FINAL_ONLY \
            and not finished:
            return None

        delta = self.output_kind == RequestOutputKind.DELTA
        output_text = self._get_next_output_text(finished, delta)
        token_ids = new_token_ids if delta else self.output_token_ids
        logprobs = new_logprobs if delta else self.request_logprobs
        prompt_logprobs = (new_prompt_logprobs
                           if delta else self.request_prompt_logprobs)
        cumulative_logprob = self.request_cumulative_logprob

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
                new_token_ids=engine_core_output.new_token_ids,
                new_logprobs=engine_core_output.logprobs,
                new_prompt_logprobs=engine_core_output.prompt_logprobs,
                new_prompt_logprob_token_ids=engine_core_output.
                prompt_logprobs_token_ids,
                finish_reason=engine_core_output.finish_reason,
                stop_reason=engine_core_output.stop_reason,
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
