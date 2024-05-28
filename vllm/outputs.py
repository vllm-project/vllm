import time
from dataclasses import dataclass
from typing import List, Optional, Union

from vllm.lora.request import LoRARequest
from vllm.sequence import (PromptLogprobs, RequestMetrics, SampleLogprobs,
                           SequenceGroup, SequenceStatus)


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
    token_ids: List[int]
    cumulative_logprob: float
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
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: List[CompletionOutput],
        finished: bool,
        metrics: Optional[RequestMetrics] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> "RequestOutput":
        if seq_group.sampling_params is None:
            raise ValueError(
                "Sampling parameters are missing for a CompletionRequest.")
        seqs = seq_group.get_seqs()
        if len(seqs) == 1:
            top_n_seqs = seqs
        else:
            # Get the top-n sequences.
            n = seq_group.sampling_params.n
            if seq_group.sampling_params.use_beam_search:
                sorting_key = lambda seq: seq.get_beam_search_score(
                    seq_group.sampling_params.length_penalty)
            else:
                sorting_key = lambda seq: seq.get_cumulative_logprob()
            sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
            top_n_seqs = sorted_seqs[:n]

        # Create the outputs.
        # NOTE: We need omit logprobs here explicitly because the sequence
        # always has the logprobs of the sampled tokens even if the
        # logprobs are not requested.
        include_logprobs = seq_group.sampling_params.logprobs is not None
        text_buffer_length = seq_group.sampling_params.output_text_buffer_length
        outputs = [
            CompletionOutput(seqs.index(seq),
                             seq.get_output_text_to_return(text_buffer_length),
                             seq.get_output_token_ids(),
                             seq.get_cumulative_logprob(),
                             seq.output_logprobs if include_logprobs else None,
                             SequenceStatus.get_finished_reason(seq.status),
                             seq.stop_reason) for seq in top_n_seqs
        ]

        # Every sequence in the sequence group should have the same prompt.
        prompt = seq_group.prompt
        prompt_token_ids = seq_group.prompt_token_ids
        prompt_logprobs = seq_group.prompt_logprobs
        finished = seq_group.is_finished()
        finished_time = time.time() if finished else None
        seq_group.set_finished_time(finished_time)
        return cls(seq_group.request_id,
                   prompt,
                   prompt_token_ids,
                   prompt_logprobs,
                   outputs,
                   finished,
                   seq_group.metrics,
                   lora_request=seq_group.lora_request)

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"prompt_logprobs={self.prompt_logprobs}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished}, "
                f"metrics={self.metrics}, "
                f"lora_request={self.lora_request})")


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
    def create(seq_group):
        # Determine the type based on a condition, for example:
        if hasattr(seq_group,
                   'embeddings') and seq_group.embeddings is not None:
            return EmbeddingRequestOutput.from_seq_group(seq_group)
        else:
            return RequestOutput.from_seq_group(seq_group)
