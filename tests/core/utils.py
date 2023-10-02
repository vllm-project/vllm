import time

from vllm import SamplingParams
from vllm.sequence import Sequence, SequenceGroup


def create_dummy_prompt(request_id: str,
                        prompt_length: int,
                        block_size: int = None):
    if not block_size:
        block_size = prompt_length

    # Create dummy prompt sequence with tokens 0...block_size-1
    # and prompt "0 ... block_size".
    prompt_tokens = list(range(prompt_length))
    prompt_str = " ".join([str(t) for t in prompt_tokens])
    prompt = Sequence(int(request_id), prompt_str, prompt_tokens, block_size)
    seq_group = SequenceGroup(request_id, [prompt], SamplingParams(),
                              time.monotonic())

    return prompt, seq_group
