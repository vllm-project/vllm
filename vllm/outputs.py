import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Union

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalPlaceholderDict
from vllm.sampling_params import RequestOutputKind
from vllm.sequence import (PromptLogprobs, RequestMetrics, SampleLogprobs,
                           SequenceGroup, SequenceGroupBase, SequenceStatus)


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
    """

    index: int
    text: str
    token_ids: GenericSequence[int]
    cumulative_logprob: Optional[float]
    logprobs: Optional[SampleLogprobs]
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None
    lora_request: Optional[LoRARequest] = None

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason}, "
                f"stop_reason={self.stop_reason})")


@dataclass
class EmbeddingOutput:
    """The output data of one completion output of a request.

    Args:
        embedding: The embedding vector, which is a list of floats. The
        length of vector depends on the model as listed in the embedding guide.
    """

    embedding: List[float]

    def __repr__(self) -> str:
        return (f"EmbeddingOutput("
                f"embedding={len(self.embedding)})")


class RequestOutput:
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]],
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: List[CompletionOutput],
        finished: bool,
        metrics: Optional[RequestMetrics] = None,
        lora_request: Optional[LoRARequest] = None,
        encoder_prompt: Optional[str] = None,
        encoder_prompt_token_ids: Optional[List[int]] = None,
        num_cached_tokens: Optional[int] = None,
        *,
        multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.multi_modal_placeholders = multi_modal_placeholders or {}
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request
        self.encoder_prompt = encoder_prompt
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.num_cached_tokens = num_cached_tokens

    @classmethod
    def new(
        cls,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]],
        text: str,
        token_ids: List[int],
        finished: bool = False,
    ) -> "RequestOutput":
        """Initialize a new RequestOutput object."""

        # TODO: Support `n` > 1.
        completion_output = CompletionOutput(
            index=0,
            text=text,
            token_ids=token_ids,
            cumulative_logprob=None,
            logprobs=None,  # TODO
        )

        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,  # TODO
            outputs=[completion_output],
            finished=finished,
        )

    @classmethod
    def from_seq_group(
        cls, seq_group: SequenceGroup, use_cache: bool,
        seq_id_to_seq_group: Dict[str, SequenceGroupBase]
    ) -> Optional["RequestOutput"]:
        finished = seq_group.is_finished()

        if seq_group.request_id in seq_id_to_seq_group:
            group: SequenceGroupBase = seq_id_to_seq_group[
                seq_group.request_id]
            if finished:
                group.finish_seq(seq_group)
            assembled_seq_group = group.maybe_assemble_group(seq_group)
            if assembled_seq_group is None:
                return None
            return cls.from_seq_group(assembled_seq_group, use_cache,
                                      seq_id_to_seq_group)

        sampling_params = seq_group.sampling_params
        if sampling_params is None:
            raise ValueError(
                "Sampling parameters are missing for a CompletionRequest.")

        if sampling_params.output_kind == RequestOutputKind.FINAL_ONLY and (
                not finished):
            return None

        # Init cache (if needed)
        if use_cache and seq_group.cached_request_output is None:
            seq_group.cached_request_output = RequestOutput(  # type: ignore
                request_id="",
                prompt=None,
                prompt_token_ids=[],
                prompt_logprobs=None,
                outputs=[],
                finished=False)

        top_n_seqs = seq_group.get_seqs()

        # Create the outputs.
        # NOTE: We need omit logprobs here explicitly because the sequence
        # always has the logprobs of the sampled tokens even if the
        # logprobs are not requested.
        include_logprobs = sampling_params.logprobs is not None
        text_buffer_length = sampling_params.output_text_buffer_length
        delta = sampling_params.output_kind == RequestOutputKind.DELTA

        outputs = []
        include_prompt = True
        # num_cached_tokens should be the same for all the sequences
        num_cached_tokens = None
        for i, seq in enumerate(top_n_seqs):
            output_text = seq.get_output_text_to_return(
                text_buffer_length, delta)

            output_token_ids = seq.get_output_token_ids_to_return(delta)
            num_output_tokens = 1 if isinstance(output_token_ids,
                                                int) else len(output_token_ids)
            num_cached_tokens = seq.data.get_num_cached_tokens()

            output_logprobs = seq.output_logprobs if include_logprobs else None

            if delta:
                # Slice logprobs delta if applicable
                if output_logprobs:
                    output_logprobs = output_logprobs[-num_output_tokens:]
                # Don't include prompt if this is after the first output
                # containing decode token ids
                if include_prompt and seq.get_output_len() > num_output_tokens:
                    include_prompt = False

            if use_cache:
                # Get cached output object
                cached_outputs = seq_group.cached_request_output.outputs  # type: ignore
                if i >= len(cached_outputs):
                    cached_outputs.append(
                        CompletionOutput(index=i,
                                         text="",
                                         token_ids=[],
                                         cumulative_logprob=None,
                                         logprobs=None,
                                         finish_reason=None,
                                         stop_reason=None))
                output = cached_outputs[i]

                # Init cached output object
                assert output.index == i
                output.text = output_text

                if isinstance(output_token_ids, int):
                    output.token_ids.clear()
                    output.token_ids.append(output_token_ids)
                else:
                    output.token_ids = output_token_ids

                output.cumulative_logprob = seq.get_cumulative_logprob() \
                    if include_logprobs else None
                output.logprobs = output_logprobs
                output.finish_reason = SequenceStatus.get_finished_reason(
                    seq.status)
                output.stop_reason = seq.stop_reason

            else:
                output = CompletionOutput(
                    top_n_seqs.index(seq), output_text, [output_token_ids]
                    if isinstance(output_token_ids, int) else output_token_ids,
                    seq.get_cumulative_logprob() if include_logprobs else None,
                    output_logprobs,
                    SequenceStatus.get_finished_reason(seq.status),
                    seq.stop_reason)

            outputs.append(output)

        # Every sequence in the sequence group should have the same prompt.
        if include_prompt:
            prompt = seq_group.prompt
            prompt_token_ids = seq_group.prompt_token_ids
            encoder_prompt = seq_group.encoder_prompt
            encoder_prompt_token_ids = seq_group.encoder_prompt_token_ids
            prompt_logprobs = seq_group.prompt_logprobs
        else:
            prompt = None
            prompt_token_ids = None
            encoder_prompt = None
            encoder_prompt_token_ids = None
            prompt_logprobs = None
        finished_time = time.time() if finished else None
        seq_group.set_finished_time(finished_time)

        init_kwargs = {
            "request_id": seq_group.request_id,
            "prompt": prompt,
            "prompt_token_ids": prompt_token_ids,
            "prompt_logprobs": prompt_logprobs,
            "outputs": outputs,
            "finished": finished,
            "metrics": seq_group.metrics,
            "lora_request": seq_group.lora_request,
            "encoder_prompt": encoder_prompt,
            "encoder_prompt_token_ids": encoder_prompt_token_ids,
            "num_cached_tokens": num_cached_tokens,
            "multi_modal_placeholders": seq_group.multi_modal_placeholders
        }

        if use_cache:
            request_output = seq_group.cached_request_output
            request_output.__init__(**init_kwargs)  # type: ignore
        else:
            request_output = cls(**init_kwargs)  # type: ignore

        return request_output

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"encoder_prompt={self.encoder_prompt!r}, "
                f"encoder_prompt_token_ids={self.encoder_prompt_token_ids}, "
                f"prompt_logprobs={self.prompt_logprobs}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished}, "
                f"metrics={self.metrics}, "
                f"lora_request={self.lora_request}, "
                f"num_cached_tokens={self.num_cached_tokens}, "
                f"multi_modal_placeholders={self.multi_modal_placeholders})")


class EmbeddingRequestOutput:
    """
    The output data of an embedding request to the LLM.

    Args:
        request_id (str): A unique identifier for the embedding request.
        outputs (EmbeddingOutput): The embedding results for the given input.
        prompt_token_ids (List[int]): A list of token IDs used in the prompt.
        finished (bool): A flag indicating whether the embedding is completed.
    """

    def __init__(self, request_id: str, outputs: "EmbeddingOutput",
                 prompt_token_ids: List[int], finished: bool):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.finished = finished
        self.outputs = outputs

    @classmethod
    def from_seq_group(cls,
                       seq_group: 'SequenceGroup') -> "EmbeddingRequestOutput":
        if seq_group.embeddings is None:
            raise ValueError(
                "Embeddings are missing in seq_group for EmbeddingRequest.")
        output = EmbeddingOutput(seq_group.embeddings)
        prompt_token_ids = seq_group.prompt_token_ids
        finished = seq_group.is_finished()

        return cls(seq_group.request_id, output, prompt_token_ids, finished)

    def __repr__(self):
        """
        Returns a string representation of an EmbeddingRequestOutput instance.

        The representation includes the request_id and the number of outputs,
        providing a quick overview of the embedding request's results.

        Returns:
            str: A string representation of the EmbeddingRequestOutput instance.
        """
        return (f"EmbeddingRequestOutput(request_id='{self.request_id}', "
                f"outputs={repr(self.outputs)}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"finished={self.finished})")


class RequestOutputFactory:

    @staticmethod
    def create(seq_group: SequenceGroup,
               seq_id_to_seq_group: Dict[str, SequenceGroupBase],
               use_cache: bool = False):
        # Determine the type based on a condition, for example:
        if hasattr(seq_group,
                   'embeddings') and seq_group.embeddings is not None:
            return EmbeddingRequestOutput.from_seq_group(seq_group)
        else:
            return RequestOutput.from_seq_group(seq_group, use_cache,
                                                seq_id_to_seq_group)
