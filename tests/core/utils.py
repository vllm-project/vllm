# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from collections.abc import Sequence as GenericSequence
from itertools import count
from typing import Any, Optional, Union

import torch

from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.inputs import EncoderDecoderInputs, embeds_inputs, token_inputs
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Logprob, Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata)


def create_dummy_prompt(
    request_id: str,
    prompt_length: int = -1,
    block_size: Optional[int] = None,
    lora_request: Optional[LoRARequest] = None,
    prompt_tokens: Optional[list[int]] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    min_tokens: int = 0,
    max_tokens: int = 16,
) -> tuple[Sequence, SequenceGroup]:
    if not block_size:
        block_size = prompt_length

    if prompt_tokens is None:
        # Create dummy prompt sequence with tokens 0...block_size-1
        # and prompt "0 ... block_size".
        prompt_tokens = list(range(prompt_length))

    prompt_str = " ".join([str(t) for t in prompt_tokens])
    inputs = token_inputs(
        prompt_token_ids=prompt_tokens,
        prompt=prompt_str) if prompt_embeds is None else embeds_inputs(
            prompt_embeds=prompt_embeds)
    prompt = Sequence(
        int(request_id),
        inputs=inputs,
        block_size=block_size,
    )
    seq_group = SequenceGroup(
        request_id=request_id,
        seqs=[prompt],
        arrival_time=time.time(),
        sampling_params=SamplingParams(max_tokens=max_tokens,
                                       min_tokens=min_tokens),
        lora_request=lora_request,
    )

    return prompt, seq_group


def create_dummy_lora_sequence(request_id: int, token_ids: list[int],
                               block_size: int, lora_int_id: int) -> Sequence:
    return Sequence(seq_id=request_id,
                    inputs=token_inputs(token_ids),
                    block_size=block_size,
                    lora_request=LoRARequest(lora_name="dummy",
                                             lora_path="/dummy",
                                             lora_int_id=lora_int_id))


def create_dummy_sequence(request_id: int, token_ids: list[int],
                          block_size: int) -> Sequence:
    return Sequence(
        seq_id=request_id,
        inputs=token_inputs(token_ids),
        block_size=block_size,
    )


def create_dummy_prompt_encoder_decoder(
    request_id: str,
    decoder_prompt_length: int,
    encoder_prompt_length: int,
    block_size: Optional[int] = None,
    lora_request: Optional[LoRARequest] = None,
) -> tuple[Sequence, Sequence, SequenceGroup]:
    if not block_size:
        block_size = decoder_prompt_length

    # Create dummy prompt sequence with tokens 0...block_size-1
    # and prompt "0 ... block_size". Note that the prompt string
    # doesn't actually match the tokens
    decoder_prompt_tokens = list(range(decoder_prompt_length))
    decoder_prompt_str = " ".join([str(t) for t in decoder_prompt_tokens])
    encoder_prompt_tokens = list(reversed(list(range(encoder_prompt_length))))
    encoder_prompt_str = " ".join([str(t) for t in encoder_prompt_tokens])

    inputs: EncoderDecoderInputs = {
        "decoder": token_inputs(decoder_prompt_tokens,
                                prompt=decoder_prompt_str),
        "encoder": token_inputs(encoder_prompt_tokens,
                                prompt=encoder_prompt_str),
    }

    decoder_prompt = Sequence(int(request_id),
                              inputs=inputs["decoder"],
                              block_size=block_size)

    encoder_prompt = Sequence(int(request_id),
                              inputs=inputs["encoder"],
                              block_size=block_size)

    seq_group = SequenceGroup(request_id=request_id,
                              seqs=[decoder_prompt],
                              arrival_time=time.time(),
                              lora_request=lora_request,
                              encoder_seq=encoder_prompt)

    return decoder_prompt, encoder_prompt, seq_group


def create_seq_group(
        seq_prompt_len: int = 1024,
        seq_output_lens: GenericSequence[int] = (128, ),
        request_id: str = '0',
        seq_id_start: int = 0,
        sampling_params: Optional[SamplingParams] = None) -> SequenceGroup:

    assert len(seq_output_lens) > 0

    if sampling_params is None:
        sampling_params = SamplingParams()

    prompt_token_ids = [0] * seq_prompt_len

    seqs: list[Sequence] = []
    for seq_id_offset, output_len in enumerate(seq_output_lens):
        seq = Sequence(
            seq_id=seq_id_start + seq_id_offset,
            inputs=token_inputs(prompt_token_ids),
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
        seq_output_lens: GenericSequence[int] = (128, ),
        request_id: str = '0',
        seq_id_start: int = 0,
        sampling_params: Optional[SamplingParams] = None) -> SequenceGroup:

    assert len(seq_output_lens) > 0

    if sampling_params is None:
        sampling_params = SamplingParams()

    prompt_token_ids = [0] * seq_prompt_len

    inputs: EncoderDecoderInputs = {
        "decoder": token_inputs(prompt_token_ids),
        "encoder": token_inputs(prompt_token_ids),
    }

    seqs = []
    for seq_id_offset, output_len in enumerate(seq_output_lens):
        # Construct decoder input sequences
        seq = Sequence(
            seq_id=seq_id_start + seq_id_offset,
            inputs=inputs["decoder"],
            block_size=16,
        )

        for i in range(output_len):
            seq.append_token_id(
                token_id=i,
                logprobs={i: Logprob(0.0)},
            )
        seqs.append(seq)

    # Encoder input sequence
    encoder_seq = Sequence(
        seq_id=seq_id_start + len(seq_output_lens),
        inputs=inputs["encoder"],
        block_size=16,
    )

    return SequenceGroup(request_id=request_id,
                         seqs=seqs,
                         sampling_params=sampling_params,
                         arrival_time=time.time(),
                         encoder_seq=encoder_seq)


def round_up_to_next_block(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size


# Helper functions for scheduler tests


def get_sequence_groups(scheduler_output):
    return [s.seq_group for s in scheduler_output.scheduled_seq_groups]


def append_new_token(out, token_id: int):
    seq_groups = get_sequence_groups(out)
    for seq_group in seq_groups:
        for seq in seq_group.get_seqs():
            seq.append_token_id(token_id, {token_id: Logprob(token_id)})


def schedule_and_update_computed_tokens(scheduler):
    metas, out, _ = scheduler.schedule()
    for s in out.scheduled_seq_groups:
        s.seq_group.update_num_computed_tokens(s.token_chunk_size)
    return metas, out


def append_new_token_seq(seq: Sequence, token_id: int):
    seq.append_token_id(token_id, {token_id: Logprob(token_id)})


def append_new_token_seq_group(token_chunk_size, seq_group, token_id: int):
    seq_group.update_num_computed_tokens(token_chunk_size)
    for seq in seq_group.get_seqs():
        seq.append_token_id(token_id, {token_id: Logprob(token_id)})


class SchedulerProxy:
    """
    A proxy class to forward calls to the scheduler.
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler_ = scheduler
        self.call_history: dict[str, list[Any]] = defaultdict(list)

    def __getattr__(self, name: str) -> Any:

        def wrapper(*args, **kwargs):
            result = getattr(self.scheduler_, name)(*args, **kwargs)
            self.call_history[name].append((args, kwargs, result))
            return result

        return wrapper

    def last_schedule_ret(
        self, ) -> tuple[list[SequenceGroupMetadata], SchedulerOutputs, Any]:
        _, _, ret = self.call_history["schedule"][-1]
        return ret


def create_seq_group_metadata_from_prompts(
    prompts: list[list[int]],
    num_gpu_blocks: int,
    block_size: int,
    final_prompt_lens: list[int],
    continuations: Optional[list[list[int]]] = None,
    seq_ids: Optional[list[int]] = None,
) -> list[SequenceGroupMetadata]:

    if continuations is None:
        continuations = [[] for _ in prompts]

    if seq_ids is None:
        seq_ids = list(i for i, _ in enumerate(prompts))

    free_gpu_blocks = list(range(num_gpu_blocks))

    block_allocations = {
        i: [
            free_gpu_blocks.pop()
            for _ in range(round_up_to_next_block(final_len, block_size))
        ]
        for i, final_len in enumerate(final_prompt_lens)
    }

    seq_grou_metadata_list = []
    for i, (prompt_token_ids,
            cont_token_ids) in enumerate(zip(prompts, continuations)):
        data = SequenceData.from_seqs(prompt_token_ids, cont_token_ids)
        data.update_num_computed_tokens(
            len(prompt_token_ids) + len(cont_token_ids) - 1)
        seq_data = {i: data}
        seq_grou_metadata_list.append(
            SequenceGroupMetadata(
                request_id=str(i),
                is_prompt=len(cont_token_ids) == 0,
                seq_data=seq_data,
                sampling_params=SamplingParams(temperature=0.0),
                block_tables={i: block_allocations[i][:]},
            ))
    return seq_grou_metadata_list


def create_chunked_seq_group_metadata_from_prompt(
        prompt: list[int],
        num_gpu_blocks: int,
        chunk_size: int,
        block_size: int,
        seq_id: Optional[int] = None) -> list[SequenceGroupMetadata]:

    if seq_id is None:
        seq_id = 0

    free_gpu_blocks = list(range(num_gpu_blocks))

    block_allocations = [
        free_gpu_blocks.pop()
        for _ in range(round_up_to_next_block(len(prompt), block_size))
    ]

    seq_group_metadata_list = []
    for i, idx in enumerate(range(0, len(prompt), chunk_size)):
        chunk_ids = prompt[idx:idx + chunk_size]
        data = SequenceData.from_seqs(prompt)
        data.update_num_computed_tokens(idx)
        seq_data = {i: data}
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=str(seq_id),
                is_prompt=True,
                do_sample=idx + chunk_size >= len(prompt),  # terminal chunk
                seq_data=seq_data,
                sampling_params=SamplingParams(temperature=0.0),
                block_tables={i: block_allocations},
                token_chunk_size=len(chunk_ids)))
    return seq_group_metadata_list


def create_batch(batch_size,
                 k,
                 prompt_len: Union[int, list[int]] = 10,
                 prev_output_token_len: int = 10,
                 seq_ids: Optional[list[int]] = None,
                 num_gpu_blocks: Optional[int] = None,
                 block_size: Optional[int] = None,
                 prefill_chunk_size: Optional[int] = None):
    if block_size is None:
        block_size = 8

    if num_gpu_blocks is None:
        num_gpu_blocks = 2048 // block_size

    iterator = count()

    if isinstance(prompt_len, int):
        prompt_lens = [prompt_len for _ in range(batch_size)]
    else:
        prompt_lens = prompt_len

    prompts = [[next(iterator) for _ in range(p_len)] for p_len in prompt_lens]

    if prefill_chunk_size:
        # Create a batch of chunked prompts.
        if not seq_ids:
            seq_ids = list(range(len(prompts)))
        seq_group_metadata_list = []
        for p, sid in zip(prompts, seq_ids):
            seq_group_metadata_list += \
                create_chunked_seq_group_metadata_from_prompt(
                p, num_gpu_blocks, prefill_chunk_size, block_size, sid)
        seq_group_metadata_list = seq_group_metadata_list[:batch_size]
        prev_output_tokens = []
    else:
        prev_output_tokens = [[
            next(iterator) for _ in range(prev_output_token_len)
        ] for _ in range(batch_size)]
        final_prompt_lens = [
            len(prompt) + len(prev_output_token) + k + 1
            for prompt, prev_output_token in zip(prompts, prev_output_tokens)
        ]

        seq_group_metadata_list = create_seq_group_metadata_from_prompts(
            prompts, num_gpu_blocks, block_size, final_prompt_lens,
            prev_output_tokens, seq_ids)
    return seq_group_metadata_list, prompts, prev_output_tokens
