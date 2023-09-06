from typing import Dict, List, Optional

from vllm.sequence import SequenceGroup, SequenceStatus


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
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[List[float]] = None,
        top_logprobs: Optional[List[Dict[int, float]]] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.finish_reason = finish_reason

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason})")


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List[CompletionOutput],
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished

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
        echo = seq_group.sampling_params.echo
        outputs: List[CompletionOutput] = []
        for seq in top_n_seqs:
            logprobs = [
                seq.output_logprobs[i][x]
                for i, x in enumerate(seq.data.output_token_ids)
            ]
            top_logprobs = seq.output_logprobs
            output_text = seq.output_text
            output_token_ids = seq.get_output_token_ids()
            cumulative_logprob = seq.get_cumulative_logprob()
            if echo:
                if seq_group.sampling_params.logprobs is not None:
                    if seq_group.sampling_params.logprobs > 0:
                        top_logprobs = (seq_group.prompt_top_logprobs +
                                        top_logprobs)
                    else:
                        top_logprobs = None
                    logprobs = seq_group.prompt_logprobs + logprobs
                output_text = seq.prompt + output_text
                output_token_ids = seq.data.prompt_token_ids + output_token_ids
                if seq_group.sampling_params.logprobs is not None:
                    cumulative_logprob = (sum(seq.data.prompt_logprobs) +
                                          cumulative_logprob)
            if seq_group.sampling_params.logprobs is None:
                # NOTE: We need to take care of this case because the sequence
                # always has the logprobs of the sampled tokens even if the
                # logprobs are not requested.
                logprobs = None
                top_logprobs = None
            elif seq_group.sampling_params.logprobs == 0:
                top_logprobs = None
            finshed_reason = SequenceStatus.get_finished_reason(seq.status)
            output = CompletionOutput(
                index=seqs.index(seq),
                text=output_text,
                token_ids=output_token_ids,
                cumulative_logprob=cumulative_logprob,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                finish_reason=finshed_reason,
            )
            outputs.append(output)

        # Every sequence in the sequence group should have the same prompt.
        prompt = top_n_seqs[0].prompt
        prompt_token_ids = top_n_seqs[0].data.prompt_token_ids
        finished = seq_group.is_finished()
        return cls(
            seq_group.request_id,
            prompt,
            prompt_token_ids,
            outputs,
            finished,
        )

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished})")
