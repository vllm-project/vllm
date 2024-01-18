import random
from typing import Tuple
from unittest.mock import patch

import pytest
import torch
from transformers import GenerationConfig, GenerationMixin

from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.worker.model_runner import ModelRunner


class MockLogitsSampler(Sampler):

    def __init__(self, vocab_size: int, fake_logits: torch.Tensor):
        super().__init__(vocab_size=vocab_size)
        self.fake_logits = fake_logits

    def forward(self, *args, **kwargs):
        with patch("vllm.model_executor.layers.sampler._prune_hidden_states",
                   lambda x, y: x), patch(
                       "vllm.model_executor.layers.sampler._get_logits",
                       lambda *args, **kwargs: self.fake_logits):
            return super().forward(*args, **kwargs)


def _prepare_test(
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, MockLogitsSampler, ModelRunner]:
    vocab_size = 32000
    input_tensor = torch.rand((batch_size, 1024),
                              device="cuda",
                              dtype=torch.float16)
    fake_logits = torch.full((batch_size, vocab_size),
                             1e-2,
                             device=input_tensor.device,
                             dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(32000, fake_logits)
    model_runner = ModelRunner(None, None, None)
    return input_tensor, fake_logits, sampler, model_runner


RANDOM_SEEDS = list(range(128))


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_all_greedy(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    seq_group_metadata_list = []
    prompt_lens = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0, ),
                block_tables={0: [1]},
            ))
        prompt_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = model_runner._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens=prompt_lens)
    sampler_output = sampler(embedding=None,
                             hidden_states=input_tensor,
                             sampling_metadata=sampling_metadata)
    expected = torch.argmax(fake_logits, dim=-1)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == expected[i].item()

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_all_random(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    seq_group_metadata_list = []
    prompt_lens = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(
                    temperature=1.0,
                    n=random.randint(1, 10),
                ),
                block_tables={0: [1]},
            ))
        prompt_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = model_runner._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens=prompt_lens)
    sampler_output = sampler(embedding=None,
                             hidden_states=input_tensor,
                             sampling_metadata=sampling_metadata)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == i

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_all_beam(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, _, sampler, model_runner = _prepare_test(batch_size)

    seq_group_metadata_list = []
    prompt_lens = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(
                    temperature=0,
                    best_of=2,
                    use_beam_search=True,
                ),
                block_tables={0: [1]},
            ))
        prompt_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = model_runner._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens=prompt_lens)
    sampler(embedding=None,
            hidden_states=input_tensor,
            sampling_metadata=sampling_metadata)
    # no assertion here as I am not sure how to determine whether
    # the outputs are expected - in other words, this just tests
    # whether there are no exceptions in the sampler
    # when handling an all-beam search case.
    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_mixed(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    seq_group_metadata_list = []
    expected_tokens = []
    prompt_lens = []
    for i in range(batch_size):
        n = 1
        sampling_type = random.randint(0, 2)
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
        else:
            sampling_params = SamplingParams(temperature=0,
                                             use_beam_search=True,
                                             best_of=2)
        for idx in range(n):
            fake_logits[i, i + idx] = 1e2
            expected_tokens.append(i + idx)
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=sampling_params,
                block_tables={0: [1]},
            ))
        prompt_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = model_runner._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens=prompt_lens)
    sampler_output = sampler(embedding=None,
                             hidden_states=input_tensor,
                             sampling_metadata=sampling_metadata)
    for i, sequence_output in enumerate(sampler_output):
        if seq_group_metadata_list[i].sampling_params.use_beam_search:
            continue
        for nth_output in sequence_output.samples:
            assert nth_output.output_token in expected_tokens

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_logits_processors(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    input_tensor, _, sampler, model_runner = _prepare_test(batch_size)

    # This sample logits processor gives infinite score to the i-th token,
    # where i is the length of the input sequence.
    # We therefore expect the output token sequence to be [0, 1, 2, ...]
    def pick_ith(token_ids, logits):
        logits[len(token_ids)] = float("inf")
        return logits

    seq_group_metadata_list = []
    prompt_lens = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(temperature=0,
                                               logits_processors=[pick_ith]),
                block_tables={0: [1]},
            ))
        prompt_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = model_runner._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens=prompt_lens)
    sampler_output = sampler(embedding=None,
                             hidden_states=input_tensor,
                             sampling_metadata=sampling_metadata)
    for _, sequence_output in enumerate(sampler_output):
        for idx, nth_output in enumerate(sequence_output.samples):
            assert nth_output.output_token == idx

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_sampler_top_k_top_p(seed: int):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    top_k = random.randint(100, 500)
    top_p = random.random() * 0.1
    vocab_size = 32000
    input_tensor = torch.rand((batch_size, 1024),
                              device="cuda",
                              dtype=torch.float16)
    fake_logits = torch.normal(0,
                               5,
                               size=(batch_size, vocab_size),
                               device=input_tensor.device,
                               dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(32000, fake_logits)
    model_runner = ModelRunner(None, None, None)

    generation_model = GenerationMixin()
    generation_config = GenerationConfig(top_k=top_k,
                                         top_p=top_p,
                                         do_sample=True)
    warpers = generation_model._get_logits_warper(generation_config)
    assert len(warpers) == 2  # top_p and top_k

    seq_group_metadata_list = []
    prompt_lens = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=SamplingParams(
                    temperature=1,
                    top_k=top_k,
                    top_p=top_p,
                ),
                block_tables={0: [1]},
            ))
        prompt_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = model_runner._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens,
                                                     subquery_lens=prompt_lens)

    sample_probs = None

    def mock_sample(probs, logprobs, sampling_metadata):
        nonlocal sample_probs
        sample_probs = probs
        return [[prob.topk(1, dim=-1).indices.tolist(), [0]] for prob in probs]

    with patch("vllm.model_executor.layers.sampler._sample", mock_sample):
        sampler(embedding=None,
                hidden_states=input_tensor,
                sampling_metadata=sampling_metadata)
    hf_probs = warpers(torch.zeros_like(fake_logits), fake_logits.clone())
    hf_probs = torch.softmax(hf_probs, dim=-1, dtype=torch.float)
    assert torch.allclose(hf_probs, sample_probs, atol=1e-5)
    assert torch.equal(hf_probs.eq(0), sample_probs.eq(0))

    del model_runner
