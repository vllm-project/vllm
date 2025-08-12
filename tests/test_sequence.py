import pytest

from vllm.sequence import (SamplerOutput, SequenceData, SequenceGroupOutput,
                           SequenceOutput)


@pytest.fixture
def sample_outputs():
    return [
        SequenceGroupOutput(samples=[
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
    assert sampler_output.spec_decode_worker_metrics is None


def test_sampler_output_getitem(sampler_output, sample_outputs):
    assert sampler_output[2] == sample_outputs[2]


def test_sampler_output_setitem(sampler_output):
    new_output = SequenceGroupOutput(samples=[
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
    seq_data = SequenceData(prompt_token_ids=[1, 2, 3, 4])
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
    seq_data.reset_num_computed_tokens()
    assert seq_data.get_num_uncomputed_tokens() == 5
    assert seq_data.get_num_computed_tokens() == 0
