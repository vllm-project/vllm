from dataclasses import dataclass
from typing import List, Optional
import time

from vllm.sequence import (PromptLogprobs, SampleLogprobs, SequenceGroup,
                           SequenceStatus, RequestMetrics)
from vllm.lora.request import LoRARequest

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
        lora_request: The LoRA request that was used to generate the output.
    """
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: float
    logprobs: Optional[SampleLogprobs] = None
    probs: Optional[List[List[float]]] = None
    finish_reason: Optional[str] = None
    lora_request: Optional[LoRARequest] = None

    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    """The output data of a request to the LLM.

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
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    prompt_logprobs: Optional[PromptLogprobs]
    outputs: List[CompletionOutput]
    finished: bool
    metrics: Optional[RequestMetrics] = None,
    lora_request: Optional[LoRARequest] = None,

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> "RequestOutput":
        # Get the top-n sequences.
        n = seq_group.sampling_params.n
        seqs = seq_group.get_seqs()
        if seq_group.sampling_params.use_beam_search:
            sorting_key = lambda seq: seq.get_beam_search_score(
                seq_group.sampling_params.length_penalty)
        else:
            sorting_key = lambda seq: seq.get_cumulative_logprob()
        sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
        top_n_seqs = sorted_seqs[:n]

        # Create the outputs.
        outputs: List[CompletionOutput] = []
        for seq in top_n_seqs:
            logprobs = seq.output_logprobs
            if seq_group.sampling_params.logprobs is None:
                # NOTE: We need to take care of this case because the sequence
                # always has the logprobs of the sampled tokens even if the
                # logprobs are not requested.
                logprobs = None
            finshed_reason = SequenceStatus.get_finished_reason(seq.get_status())
            output = CompletionOutput(seqs.index(seq), seq.output_text,
                                      seq.get_output_token_ids(),
                                      seq.get_cumulative_logprob(), logprobs,
                                      finshed_reason)
            outputs.append(output)

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
