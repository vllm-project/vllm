import time
from typing import Tuple

from vllm import SamplingParams
from vllm.sequence import Logprob, Sequence, SequenceGroup


def create_dummy_prompt(
        request_id: str,
        prompt_length: int,
        block_size: int = None) -> Tuple[Sequence, SequenceGroup]:
    if not block_size:
        block_size = prompt_length

    # Create dummy prompt sequence with tokens 0...block_size-1
    # and prompt "0 ... block_size".
    prompt_tokens = list(range(prompt_length))
    prompt_str = " ".join([str(t) for t in prompt_tokens])
    prompt = Sequence(int(request_id), prompt_str, prompt_tokens, block_size)
    seq_group = SequenceGroup(request_id, [prompt], SamplingParams(),
                              time.time(), None)

    return prompt, seq_group


def create_seq_group(
    seq_prompt_lens=1024,
    seq_output_lens=(128, ),
    request_id='0',
    seq_id_start=0,
) -> SequenceGroup:

    assert len(seq_output_lens) > 0

    prompt_token_ids = [0] * seq_prompt_lens

    seqs = []
    for seq_id_offset, output_len in enumerate(seq_output_lens):
        seq = Sequence(
            seq_id=seq_id_start + seq_id_offset,
            prompt="",
            prompt_token_ids=prompt_token_ids,
            block_size=16,
        )

        for i in range(output_len):
            seq.append_token_id(
                token_id=i,
                logprobs={i: Logprob(0.0)},
            )
        seqs.append(seq)

    seq_group = SequenceGroup(
        request_id=request_id,
        seqs=seqs,
        sampling_params=SamplingParams(),
        arrival_time=time.time(),
    )

    return seq_group


def round_up_to_next_block(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size
