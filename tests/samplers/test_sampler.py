import random
from typing import Tuple, List
from unittest.mock import patch

import pytest
import torch
from transformers import GenerationConfig, GenerationMixin
from typing import Optional

from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.worker.model_runner import ModelRunner


class MockLogitsSampler(Sampler):

    def __init__(self, vocab_size: int, fake_logits: torch.Tensor):
        super().__init__(vocab_size=vocab_size)
        self.fake_logits = fake_logits

    def forward(self, *args, **kwargs):
        with patch(
                "vllm.model_executor.layers.sampler._prune_hidden_states",
                lambda x, y: x), patch(
                    "vllm.model_executor.layers.sampler.Sampler._get_logits",
                    lambda *args, **kwargs: self.fake_logits):
            return super().forward(*args, **kwargs)


def _prepare_test(
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, MockLogitsSampler, ModelRunner]:
    vocab_size = 32000
    input_tensor = torch.rand((batch_size, 1024), dtype=torch.float16)
    fake_logits = torch.full((batch_size, vocab_size),
                             1e-2,
                             dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(32000, fake_logits)
    model_runner = ModelRunner(None, None, None, None, None)
    return input_tensor, fake_logits, sampler, model_runner


RANDOM_SEEDS = list(range(128))
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


def _do_sample(
    batch_size: int,
    input_tensor: torch.Tensor,
    sampler: MockLogitsSampler,
    model_runner: ModelRunner,
    sampling_params: SamplingParams,
):
    seq_group_metadata_list = []
    prompt_lens = []
    for i in range(batch_size):
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
    return sampler(embedding=None,
                   hidden_states=input_tensor,
                   sampling_metadata=sampling_metadata)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_greedy(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    sampling_params = SamplingParams(temperature=0)
    sampler_output = _do_sample(batch_size, input_tensor, sampler,
                                model_runner, sampling_params)
    expected = torch.argmax(fake_logits, dim=-1)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == expected[i].item()

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_random(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
    )
    sampler_output = _do_sample(batch_size, input_tensor, sampler,
                                model_runner, sampling_params)

    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == i

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_random_seed(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
        seed=random.randint(0, 10000),
    )
    sampler_output = _do_sample(batch_size, input_tensor, sampler,
                                model_runner, sampling_params)

    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == i

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_random_seed_deterministic(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
        seed=random.randint(0, 10000),
    )
    first_sampler_output = _do_sample(batch_size, input_tensor, sampler,
                                      model_runner, sampling_params)

    second_sampler_output = _do_sample(batch_size, input_tensor, sampler,
                                       model_runner, sampling_params)

    assert first_sampler_output == second_sampler_output

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_beam(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, _, sampler, model_runner = _prepare_test(batch_size)

    sampling_params = SamplingParams(
        temperature=0,
        best_of=2,
        use_beam_search=True,
    )
    _do_sample(batch_size, input_tensor, sampler, model_runner,
               sampling_params)
    # no assertion here as I am not sure how to determine whether
    # the outputs are expected - in other words, this just tests
    # whether there are no exceptions in the sampler
    # when handling an all-beam search case.
    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_mixed(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    seq_group_metadata_list = []
    expected_tokens: List[Optional[List[int]]] = []
    prompt_lens = []
    for i in range(batch_size):
        expected: Optional[List[int]] = None
        sampling_type = random.randint(0, 3)
        if sampling_type == 0:
            sampling_params = SamplingParams(temperature=0)
            expected = [torch.argmax(fake_logits[i], dim=-1).item()]
        elif sampling_type in (1, 2):
            n = random.randint(1, 10)
            sampling_params = SamplingParams(
                temperature=random.random() + 0.1,
                top_p=min(random.random() + 0.1, 1),
                top_k=random.randint(0, 10) or -1,
                n=n,
                presence_penalty=random.randint(0, 1),
            )
            if sampling_type == 2:
                sampling_params.seed = random.randint(0, 10000)
            else:
                for idx in range(n):
                    fake_logits[i, i + idx] = 1e2
                expected = list(range(i, i + n))
        else:
            sampling_params = SamplingParams(temperature=0,
                                             use_beam_search=True,
                                             best_of=2)
        expected_tokens.append(expected)
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={0: SequenceData([1, 2, 3])},
                sampling_params=sampling_params,
                block_tables={0: [1]},
            ))
        prompt_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    def test_sampling(model_runner: ModelRunner):
        sampling_metadata = model_runner._prepare_sample(
            seq_group_metadata_list, prompt_lens, subquery_lens=prompt_lens)
        sampler_output = sampler(embedding=None,
                                 hidden_states=input_tensor,
                                 sampling_metadata=sampling_metadata)

        for i, (sequence_output, metadata) in enumerate(
                zip(sampler_output, seq_group_metadata_list)):
            if metadata.sampling_params.use_beam_search:
                continue

            if metadata.sampling_params.seed is not None \
                    and expected_tokens[i] is None:
                # Record seeded random result to compare with results of second invocation
                expected_tokens[i] = [
                    nth_output.output_token
                    for nth_output in sequence_output.samples
                ]
                continue

            for n, nth_output in enumerate(sequence_output.samples):
                if metadata.sampling_params.temperature == 0 or metadata.sampling_params.seed is not None:
                    # Ensure exact matches for greedy or random with seed
                    assert nth_output.output_token == expected_tokens[i][n]
                else:
                    # For non-seeded random check that one of the high-logit tokens were chosen
                    assert nth_output.output_token in expected_tokens[i]

    # Test batch
    test_sampling(model_runner)

    # Shuffle the batch and resample
    target_index = list(range(batch_size))
    for list_to_shuffle in (target_index, seq_group_metadata_list,
                            expected_tokens, prompt_lens):
        random.Random(seed).shuffle(list_to_shuffle)
    target_index = torch.tensor(target_index)
    input_tensor.data = input_tensor.index_select(0, target_index)
    fake_logits.data = fake_logits.index_select(0, target_index)

    # This time, results of seeded random samples will be compared with the corresponding
    # sample in the pre-shuffled batch
    test_sampling(model_runner)

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_logits_processors(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
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
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_top_k_top_p(seed: int, device: str):
    set_random_seed(seed)
    batch_size = random.randint(1, 256)
    top_k = random.randint(100, 500)
    top_p = random.random() * 0.1
    vocab_size = 32000
    input_tensor = torch.rand((batch_size, 1024),
                              device=device,
                              dtype=torch.float16)
    fake_logits = torch.normal(0,
                               5,
                               size=(batch_size, vocab_size),
                               device=input_tensor.device,
                               dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(32000, fake_logits)
    model_runner = ModelRunner(None, None, None, None, None)

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
