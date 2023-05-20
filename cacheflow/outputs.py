from typing import Dict, List, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from cacheflow.sequence import SequenceGroup


class CompletionOutput:

    def __init__(
        self,
        text: str,
        token_ids: List[int],
        cumulative_logprobs: float,
        logprobs: List[Dict[int, float]],
    ) -> None:
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprobs = cumulative_logprobs
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"CompletionOutput(output={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprobs={self.cumulative_logprobs}, "
                f"logprobs={self.logprobs})")


class RequestOutput:

    def __init__(
        self,
        request_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List[CompletionOutput],
        done: bool = False,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.done = done

    @staticmethod
    def from_seq_group(
        seq_group: SequenceGroup,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> "RequestOutput":
        outputs: List[CompletionOutput] = []
        seqs = seq_group.get_seqs()
        for seq in seqs:
            output_token_ids = seq.data.output_token_ids
            output_str = tokenizer.decode(output_token_ids,
                                          skip_special_tokens=True)
            seq_logprobs = seq.data.cumulative_logprobs

            logprobs = seq.output_logprobs
            if seq_group.sampling_params.logprobs == 0:
                # NOTE: We need to take care of this case because the sequence
                # always has the logprobs of the sampled tokens even if the
                # logprobs are not requested.
                logprobs = {}
            output = CompletionOutput(output_str, output_token_ids,
                                      seq_logprobs, logprobs)
            outputs.append(output)

        # Every sequence in the sequence group should have the same prompt.
        prompt = seqs[0].prompt
        prompt_token_ids = seqs[0].data.prompt_token_ids
        return RequestOutput(seq_group.request_id, prompt, prompt_token_ids,
                             outputs, seq_group.is_finished())

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"outputs={self.outputs}, "
                f"done={self.done})")
