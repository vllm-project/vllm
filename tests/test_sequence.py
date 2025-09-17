# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Optional

import pytest
import torch

from vllm.inputs import embeds_inputs, token_inputs
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Sequence, SequenceData, SequenceGroup,
                           SequenceOutput)


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


@pytest.fixture
def sample_outputs():
    return [
        CompletionSequenceGroupOutput(samples=[
            SequenceOutput(parent_seq_id=0, output_token=i, logprobs={})
        ],
                                      prompt_logprobs=None) for i in range(5)
    ]


@pytest.fixture
def sampler_output(sample_outputs):
    return SamplerOutput(outputs=sample_outputs)


def test_sampler_output_initialization(sampler_output, sample_outputs):
    assert len(sampler_output) == len(sample_outputs)
    assert sampler_output.sampled_token_probs is None
    assert sampler_output.sampled_token_ids is None


def test_sampler_output_getitem(sampler_output, sample_outputs):
    assert sampler_output[2] == sample_outputs[2]


def test_sampler_output_setitem(sampler_output):
    new_output = CompletionSequenceGroupOutput(samples=[
        SequenceOutput(parent_seq_id=0, output_token=99, logprobs={})
    ],
                                               prompt_logprobs=None)
    sampler_output[2] = new_output
    assert sampler_output[2] == new_output


def test_sampler_output_len(sampler_output, sample_outputs):
    assert len(sampler_output) == len(sample_outputs)


def test_sampler_output_eq(sample_outputs):
    sampler_output1 = SamplerOutput(outputs=sample_outputs)
    sampler_output2 = SamplerOutput(outputs=sample_outputs.copy())
    sampler_output3 = SamplerOutput(outputs=sample_outputs[:-1])
    assert sampler_output1 == sampler_output2
    assert sampler_output1 != sampler_output3


def test_sequence_data_prefill():
    seq_data = SequenceData.from_seqs([1, 2, 3, 4])
    assert seq_data.get_num_uncomputed_tokens() == 4
    assert seq_data.get_num_computed_tokens() == 0
    # advance by 2
    seq_data.update_num_computed_tokens(2)
    assert seq_data.get_num_uncomputed_tokens() == 2
    assert seq_data.get_num_computed_tokens() == 2

    # advance by 1
    seq_data.update_num_computed_tokens(1)
    assert seq_data.get_num_uncomputed_tokens() == 1
    assert seq_data.get_num_computed_tokens() == 3

    # append tokens and reset, simulating recompute
    seq_data.append_token_id(1, logprob=0.0)
    seq_data.reset_state_for_recompute()
    assert seq_data.get_num_uncomputed_tokens() == 5
    assert seq_data.get_num_computed_tokens() == 0


def test_sequence_group_stage():
    _, seq_group = create_dummy_prompt("1", 12)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(6)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(5)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(1)
    assert seq_group.is_prefill() is False
    seqs = seq_group.get_seqs()
    assert len(seqs) == 1
    seqs[0].data.append_token_id(1, logprob=0.0)
    for seq in seq_group.get_seqs():
        seq.reset_state_for_recompute()
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(5)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(7)
    assert seq_group.is_prefill() is True
    seq_group.update_num_computed_tokens(1)
    assert seq_group.is_prefill() is False


def test_sequence_intermediate_tensors_equal():

    class AnotherIntermediateTensors(IntermediateTensors):
        pass

    intermediate_tensors = IntermediateTensors({})
    another_intermediate_tensors = AnotherIntermediateTensors({})
    assert intermediate_tensors != another_intermediate_tensors

    empty_intermediate_tensors_1 = IntermediateTensors({})
    empty_intermediate_tensors_2 = IntermediateTensors({})
    assert empty_intermediate_tensors_1 == empty_intermediate_tensors_2

    different_key_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)})
    difference_key_intermediate_tensors_2 = IntermediateTensors(
        {"2": torch.zeros([2, 4], dtype=torch.int32)})
    assert (different_key_intermediate_tensors_1
            != difference_key_intermediate_tensors_2)

    same_key_different_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)})
    same_key_different_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 5], dtype=torch.int32)})
    assert (same_key_different_value_intermediate_tensors_1
            != same_key_different_value_intermediate_tensors_2)

    same_key_same_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)})
    same_key_same_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)})
    assert (same_key_same_value_intermediate_tensors_1 ==
            same_key_same_value_intermediate_tensors_2)
