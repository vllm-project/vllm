import time
from typing import Iterable, Optional, Tuple

from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sequence import Logprob, Sequence, SequenceGroup


def create_dummy_prompt(
    request_id: str,
    prompt_length: int,
    block_size: Optional[int] = None,
    lora_request: Optional[LoRARequest] = None,
    use_beam_search: bool = False,
    best_of: int = 1,
) -> Tuple[Sequence, SequenceGroup]:
    if not block_size:
        block_size = prompt_length

    # Create dummy prompt sequence with tokens 0...block_size-1
    # and prompt "0 ... block_size".
    prompt_tokens = list(range(prompt_length))
    prompt_str = " ".join([str(t) for t in prompt_tokens])
    prompt = Sequence(int(request_id),
                      inputs={
                          "prompt": prompt_str,
                          "prompt_token_ids": prompt_tokens,
                      },
                      block_size=block_size)
    seq_group = SequenceGroup(request_id=request_id,
                              seqs=[prompt],
                              arrival_time=time.time(),
                              sampling_params=SamplingParams(
                                  use_beam_search=use_beam_search,
                                  best_of=best_of),
                              lora_request=lora_request)

    return prompt, seq_group


def create_dummy_prompt_encoder_decoder(
    request_id: str,
    decoder_prompt_length: int,
    encoder_prompt_length: int,
    block_size: Optional[int] = None,
    lora_request: Optional[LoRARequest] = None,
    use_beam_search: bool = False,
    best_of: int = 1,
) -> Tuple[Sequence, SequenceGroup]:
    if not block_size:
        block_size = decoder_prompt_length

    # Create dummy prompt sequence with tokens 0...block_size-1
    # and prompt "0 ... block_size".
    decoder_prompt_tokens = list(range(decoder_prompt_length))
    decoder_prompt_str = " ".join([str(t) for t in decoder_prompt_tokens])

    decoder_prompt = Sequence(int(request_id),
                              inputs={
                                  "prompt": decoder_prompt_str,
                                  "prompt_token_ids": decoder_prompt_tokens,
                                  "multi_modal_data": None,
                              },
                              block_size=block_size)

    encoder_prompt_tokens = list(reversed(list(range(encoder_prompt_length))))
    encoder_prompt_str = " ".join([str(t) for t in encoder_prompt_tokens])
    encoder_prompt = Sequence(int(request_id),
                              inputs={
                                  "prompt": encoder_prompt_str,
                                  "prompt_token_ids": encoder_prompt_tokens,
                                  "multi_modal_data": None,
                              },
                              block_size=block_size)
    seq_group = SequenceGroup(request_id=request_id,
                              seqs=[decoder_prompt],
                              sampling_params=SamplingParams(
                                  use_beam_search=use_beam_search,
                                  best_of=best_of),
                              arrival_time=time.time(),
                              lora_request=lora_request,
                              encoder_seq=encoder_prompt)

    return decoder_prompt, encoder_prompt, seq_group


def create_seq_group(
        seq_prompt_len: int = 1024,
        seq_output_lens: Iterable[int] = (128, ),
        request_id: str = '0',
        seq_id_start: int = 0,
        sampling_params: Optional[SamplingParams] = None) -> SequenceGroup:

    assert len(seq_output_lens) > 0

    if sampling_params is None:
        sampling_params = SamplingParams()

    prompt_token_ids = [0] * seq_prompt_len

    seqs = []
    for seq_id_offset, output_len in enumerate(seq_output_lens):
        seq = Sequence(
            seq_id=seq_id_start + seq_id_offset,
            inputs={"prompt_token_ids": prompt_token_ids},
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
        sampling_params=sampling_params,
        arrival_time=time.time(),
    )

    return seq_group


def create_seq_group_encoder_decoder(
        seq_prompt_len: int = 1024,
        seq_output_lens: Iterable[int] = (128, ),
        request_id: str = '0',
        seq_id_start: int = 0,
        sampling_params: Optional[SamplingParams] = None) -> SequenceGroup:

    assert len(seq_output_lens) > 0

    if sampling_params is None:
        sampling_params = SamplingParams()

    prompt_token_ids = [0] * seq_prompt_len

    seqs = []
    for seq_id_offset, output_len in enumerate(seq_output_lens):
        seq = Sequence(
            seq_id=seq_id_start + seq_id_offset,
            inputs={
                "prompt": "",
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": None,
            },
            block_size=16,
        )

        for i in range(output_len):
            seq.append_token_id(
                token_id=i,
                logprobs={i: Logprob(0.0)},
            )
        seqs.append(seq)

    # Encoder sequence
    encoder_seq = Sequence(
        seq_id=seq_id_start + len(seq_output_lens),
        inputs={
            "prompt": "",
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": None,
        },
        block_size=16,
    )

    return SequenceGroup(request_id=request_id,
                         seqs=seqs,
                         sampling_params=sampling_params,
                         arrival_time=time.time(),
                         encoder_seq=encoder_seq)


def round_up_to_next_block(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size