
# pylint: disable=protected-access
import random
from unittest.mock import patch
from typing import Optional, Tuple

import pytest
import torch

from vllm.model_executor.layers.sampler import Sampler, pythonize_sampler_output
from vllm.config import ParallelConfig, SchedulerConfig
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.model_executor.input_metadata import InputMetadata
from vllm.sequence import SamplerOutput
from vllm.worker.worker import Worker


class MockLogitsSampler(Sampler):

    def __init__(self, vocab_size: int, fake_logits: torch.Tensor):
        super().__init__(vocab_size=vocab_size, org_vocab_size=vocab_size)
        self.fake_logits = fake_logits

    def _get_logits(self, *args, **kwargs) -> torch.Tensor:
        del args
        del kwargs
        return self.fake_logits

    def forward(
            self,
            embedding: torch.Tensor,
            hidden_states: torch.Tensor,
            input_metadata: InputMetadata,
            embedding_bias: Optional[torch.Tensor] = None) -> SamplerOutput:
        with patch("vllm.model_executor.layers.sampler._prune_hidden_states",
                   lambda x, y: x):
            return super().forward(embedding, hidden_states, input_metadata,
                                   embedding_bias)


def _prepare_test(
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, MockLogitsSampler, Worker]:
    vocab_size = 32000
    input_tensor = torch.rand((batch_size, 1024),
                              device="cuda",
                              dtype=torch.float16)
    fake_logits = torch.full((batch_size, vocab_size),
                             1e-2,
                             device=input_tensor.device,
                             dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(32000, fake_logits)
    scheduler_config = SchedulerConfig(2048, 2048, 2048)
    worker = Worker(None, ParallelConfig(1, 1, False), scheduler_config)
    worker.block_size = 16
    return input_tensor, fake_logits, sampler, worker


RANDOM_SEEDS = list(range(128))


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_all_greedy(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, worker = _prepare_test(batch_size)

    seq_group_metadata_list = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                is_chunked_prefill=False,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0, ),
                block_tables={0: [1]},
                lora_request=None,
            ))

    _, _, input_metadata, _, _ = worker._prepare_inputs(
        seq_group_metadata_list)
    sampler_output = pythonize_sampler_output(
        sampler(embedding=None,
                hidden_states=input_tensor,
                input_metadata=input_metadata), input_metadata)
    expected = torch.argmax(fake_logits, dim=-1)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == expected[i].item()


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_all_random(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, worker = _prepare_test(batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    seq_group_metadata_list = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                is_chunked_prefill=False,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(
                    temperature=1.0,
                    n=random.randint(1, 10),
                ),
                block_tables={0: [1]},
                lora_request=None,
            ))

    _, _, input_metadata, _, _ = worker._prepare_inputs(
        seq_group_metadata_list)
    sampler_output = pythonize_sampler_output(
        sampler(embedding=None,
                hidden_states=input_tensor,
                input_metadata=input_metadata), input_metadata)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == i


# @pytest.mark.parametrize("seed", RANDOM_SEEDS)
# def test_sampler_all_beam(seed: int):
#     set_random_seed(seed)
#     batch_size = random.randint(1, 256)
#     input_tensor, _, sampler, worker = _prepare_test(batch_size)

#     seq_group_metadata_list = []
#     for i in range(batch_size):
#         seq_group_metadata_list.append(
#             SequenceGroupMetadata(
#                 request_id=f"test_{i}",
#                 is_prompt=True,
#                 is_chunked_prefill=False,
#                 seq_data={0: SequenceData([1, 2, 3])},
#                 sampling_params=SamplingParams(
#                     temperature=0,
#                     best_of=2,
#                     use_beam_search=True,
#                 ),
#                 block_tables={0: [1]},
#                 lora_request=None,
#             ))

#     _, _, input_metadata, _, _ = worker._prepare_inputs(
#         seq_group_metadata_list)
#     pythonize_sampler_output(
#         sampler(embedding=None,
#                 hidden_states=input_tensor,
#                 input_metadata=input_metadata), input_metadata)
#     # no assertion here as I am not sure how to determine whether
#     # the outputs are expected - in other words, this just tests
#     # whether there are no exceptions in the sampler
#     # when handling an all-beam search case.


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_mixed(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, worker = _prepare_test(batch_size)

    seq_group_metadata_list = []
    expected_tokens = []
    for i in range(batch_size):
        n = 1
        sampling_type = random.randint(0, 1)
        if sampling_type == 0:
            sampling_params = SamplingParams(temperature=0)
        elif sampling_type == 1:
            n = random.randint(1, 10)
            sampling_params = SamplingParams(
                temperature=random.random() + 0.1,
                top_p=min(random.random() + 0.1, 1),
                top_k=random.randint(0, 10) or -1,
                n=n,
                presence_penalty=random.randint(0, 1),
            )
        # else:
        #     sampling_params = SamplingParams(temperature=0,
        #                                      use_beam_search=True,
        #                                      best_of=2)
        for idx in range(n):
            fake_logits[i, i + idx] = 1e2
            expected_tokens.append(i + idx)
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                is_chunked_prefill=False,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=sampling_params,
                block_tables={0: [1]},
                lora_request=None,
            ))

    _, _, input_metadata, _, _ = worker._prepare_inputs(
        seq_group_metadata_list)
    sampler_output = pythonize_sampler_output(
        sampler(embedding=None,
                hidden_states=input_tensor,
                input_metadata=input_metadata), input_metadata)
    for i, sequence_output in enumerate(sampler_output):
        # if seq_group_metadata_list[i].sampling_params.use_beam_search:
        #     continue
        for nth_output in sequence_output.samples:
            assert nth_output.output_token in expected_tokens


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_logits_processors(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, _, sampler, worker = _prepare_test(batch_size)

    # This sample logits processor gives infinite score to the i-th token,
    # where i is the length of the input sequence.
    # We therefore expect the output token sequence to be [0, 1, 2, ...]
    def pick_ith(token_ids, logits):
        logits[len(token_ids)] = float("inf")
        return logits

    seq_group_metadata_list = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                is_chunked_prefill=False,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0,
                                               logits_processors=[pick_ith]),
                block_tables={0: [1]},
                lora_request=None,
            ))

    _, _, input_metadata, _, _ = worker._prepare_inputs(
        seq_group_metadata_list)
    sampler_output = pythonize_sampler_output(
        sampler(embedding=None,
                hidden_states=input_tensor,
                input_metadata=input_metadata), input_metadata)
    for i, sequence_output in enumerate(sampler_output):
        for idx, nth_output in enumerate(sequence_output.samples):
            assert nth_output.output_token == idx
