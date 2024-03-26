import random
from typing import List, Optional, Tuple
from unittest.mock import patch

import pytest
import torch
from transformers import GenerationConfig, GenerationMixin

from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import SamplingParams, SequenceData, SequenceGroupMetadata
from vllm.utils import Counter
from vllm.worker.model_runner import ModelRunner


class MockLogitsSampler(Sampler):

    def __init__(self, fake_logits: torch.Tensor):
        super().__init__()
        self.fake_logits = fake_logits

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


def _prepare_test(
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, MockLogitsSampler, ModelRunner]:
    input_tensor = torch.rand((batch_size, 1024), dtype=torch.float16)
    fake_logits = torch.full((batch_size, VOCAB_SIZE),
                             1e-2,
                             dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(fake_logits)
    model_runner = ModelRunner(None, None, None, None, None)
    return input_tensor, fake_logits, sampler, model_runner


VOCAB_SIZE = 32000
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
    return sampler(logits=input_tensor, sampling_metadata=sampling_metadata)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_greedy(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler, model_runner = _prepare_test(
        batch_size)

    sampling_params = SamplingParams(temperature=0)
    sampler_output = _do_sample(batch_size, fake_logits, sampler, model_runner,
                                sampling_params)
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
    sampler_output = _do_sample(batch_size, fake_logits, sampler, model_runner,
                                sampling_params)

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
    _, fake_logits, sampler, model_runner = _prepare_test(batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
        seed=random.randint(0, 10000),
    )
    sampler_output = _do_sample(batch_size, fake_logits, sampler, model_runner,
                                sampling_params)

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
    _, fake_logits, sampler, model_runner = _prepare_test(batch_size)

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
        seed=random.randint(0, 10000),
    )
    first_sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                      model_runner, sampling_params)

    second_sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                       model_runner, sampling_params)

    assert first_sampler_output == second_sampler_output

    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_beam(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    _, fake_logits, sampler, model_runner = _prepare_test(batch_size)

    sampling_params = SamplingParams(
        temperature=0,
        best_of=2,
        use_beam_search=True,
    )
    _do_sample(batch_size, fake_logits, sampler, model_runner, sampling_params)
    # no assertion here as I am not sure how to determine whether
    # the outputs are expected - in other words, this just tests
    # whether there are no exceptions in the sampler
    # when handling an all-beam search case.
    del model_runner


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_min_tokens_penalty(seed: int, device: str):
    seq_id_counter = Counter(start=random.randint(0, 100))
    set_random_seed(seed)
    torch.set_default_device(device)

    def create_sampling_params(min_tokens,
                               eos_token_id=0,
                               stop_token_ids=None):
        sampling_params = SamplingParams(
            min_tokens=min_tokens,
            max_tokens=9999,  # keep higher than max of min_tokens
            stop_token_ids=stop_token_ids,
        )
        sampling_params.eos_token_id = eos_token_id
        return sampling_params

    def create_sequence_data(num_input=3, num_generated=0):
        seq_data = SequenceData(
            random.choices(range(0, VOCAB_SIZE), k=num_input))
        if num_generated > 0:
            seq_data.output_token_ids = random.choices(range(0, VOCAB_SIZE),
                                                       k=num_generated)
        return seq_data

    def generate_test_case():
        # generate multiple seq groups but limit total batch size
        batch_size = random.randint(1, 128)

        expected_penalization = []
        sequence_metadata_list = []
        while batch_size > 0:
            # 20% chance to generate prompt seq group with single sequence
            is_prompt = random.random() < 0.2
            num_seqs = 1 if is_prompt else random.randint(1, batch_size)

            eos_token_id = random.randint(0, VOCAB_SIZE - 1)
            min_tokens = random.randint(0, 50)
            num_stop_tokens = random.randint(0, 8)
            if num_stop_tokens > 0:
                stop_token_ids = random.choices(range(0, VOCAB_SIZE - 1),
                                                k=num_stop_tokens)
            else:
                stop_token_ids = None

            sampling_params = create_sampling_params(
                min_tokens=min_tokens,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids)

            seq_data = {}
            seq_group_penalization = []
            for _ in range(num_seqs):
                num_input = random.randint(1, 100)
                num_generated = random.randint(1, 100) if not is_prompt else 0
                seq_data[next(seq_id_counter)] = create_sequence_data(
                    num_input=num_input, num_generated=num_generated)
                seq_group_penalization.append(num_generated < min_tokens)

            expected_penalization.extend(seq_group_penalization)
            sequence_metadata_list.append(
                SequenceGroupMetadata(
                    request_id=f"test_{batch_size}",
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=sampling_params,
                    block_tables={},
                ))
            batch_size -= num_seqs

        return {
            "expected_penalization": expected_penalization,
            "seq_group_metadata_list": sequence_metadata_list,
        }

    # define some explicit test cases for edge case behavior
    prompt_without_penalization = {
        "expected_penalization": [False],
        "seq_group_metadata_list": [
            SequenceGroupMetadata(
                request_id="test_1",
                is_prompt=True,
                seq_data={
                    next(seq_id_counter): create_sequence_data(),
                },
                sampling_params=create_sampling_params(0),
                block_tables={},
            ),
        ]
    }

    prompt_with_penalization = {
        "expected_penalization": [True],
        "seq_group_metadata_list": [
            SequenceGroupMetadata(
                request_id="test_1",
                is_prompt=True,
                seq_data={
                    next(seq_id_counter): create_sequence_data(),
                },
                sampling_params=create_sampling_params(1),
                block_tables={},
            ),
        ]
    }

    stop_penalizing_after_min_tokens = {
        "expected_penalization": [False],
        "seq_group_metadata_list": [
            SequenceGroupMetadata(
                request_id="test_1",
                is_prompt=False,
                seq_data={
                    next(seq_id_counter):
                    create_sequence_data(num_generated=1),
                },
                sampling_params=create_sampling_params(1),
                block_tables={},
            )
        ]
    }

    stop_token_ids = [42, 99, 42, 0]  # intentional duplication
    simple_combination = {
        "expected_penalization": [True, False, False],
        "seq_group_metadata_list": [
            SequenceGroupMetadata(
                request_id="test_1",
                is_prompt=False,
                seq_data={
                    next(seq_id_counter):
                    create_sequence_data(num_generated=1),
                    next(seq_id_counter):
                    create_sequence_data(num_generated=100),
                },
                sampling_params=create_sampling_params(
                    2, stop_token_ids=stop_token_ids),
                block_tables={},
            ),
            SequenceGroupMetadata(
                request_id="test_2",
                is_prompt=True,
                seq_data={
                    next(seq_id_counter): create_sequence_data(),
                },
                sampling_params=create_sampling_params(
                    0, stop_token_ids=stop_token_ids),
                block_tables={},
            )
        ]
    }

    if seed == 0:
        test_cases = [
            prompt_without_penalization,
            prompt_with_penalization,
            stop_penalizing_after_min_tokens,
            simple_combination,
        ]
    else:
        test_cases = [generate_test_case()]

    def run_test_case(*,
                      expected_penalization=None,
                      seq_group_metadata_list=None):
        assert expected_penalization, "Invalid test case"
        assert seq_group_metadata_list, "Invalid test case"

        batch_size = 0
        prompt_lens = []
        sampling_params_per_seq = []
        for sgm in seq_group_metadata_list:
            num_seqs = len(sgm.seq_data)
            batch_size += num_seqs
            sampling_params = sgm.sampling_params
            for seq_id in sgm.seq_data:
                prompt_lens.append(sgm.seq_data[seq_id].get_prompt_len())
                sampling_params_per_seq.append(sampling_params)

        _, fake_logits, sampler, model_runner = _prepare_test(batch_size)
        sampling_metadata = model_runner._prepare_sample(
            seq_group_metadata_list,
            prompt_lens=prompt_lens,
            subquery_lens=prompt_lens)
        # the logits tensor is modified in-place by the sampler
        _ = sampler(logits=fake_logits, sampling_metadata=sampling_metadata)

        for logits_idx, (should_penalize, sampling_params) in enumerate(
                zip(expected_penalization, sampling_params_per_seq)):

            tokens_to_check = [sampling_params.eos_token_id]
            if sampling_params.stop_token_ids:
                tokens_to_check.extend(sampling_params.stop_token_ids)
            tokens_to_check = set(tokens_to_check)

            if should_penalize:
                for token_id in tokens_to_check:
                    assert fake_logits[logits_idx, token_id] == -float(
                        'inf'
                    ), f"Expected token {token_id} for logits row {logits_idx}"
                    " to be penalized"
                # no other tokens should be set to -inf
                assert torch.count_nonzero(
                    fake_logits[logits_idx, :] == -float('inf')) == len(
                        tokens_to_check
                    ), f"Expected only {len(tokens_to_check)} to be penalized"
            else:
                # no tokens should be set to -inf
                assert torch.count_nonzero(
                    fake_logits[logits_idx, :] ==
                    -float('inf')) == 0, "No tokens should have been penalized"

        del model_runner

    for test_case in test_cases:
        run_test_case(**test_case)


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
        sampler_output = sampler(logits=fake_logits,
                                 sampling_metadata=sampling_metadata)

        for i, (sequence_output, metadata) in enumerate(
                zip(sampler_output, seq_group_metadata_list)):
            if metadata.sampling_params.use_beam_search:
                continue

            if (metadata.sampling_params.seed is not None
                    and expected_tokens[i] is None):
                # Record seeded random result to compare with results of
                # second invocation
                expected_tokens[i] = [
                    nth_output.output_token
                    for nth_output in sequence_output.samples
                ]
                continue

            for n, nth_output in enumerate(sequence_output.samples):
                if (metadata.sampling_params.temperature == 0
                        or metadata.sampling_params.seed is not None):
                    # Ensure exact matches for greedy or random with seed
                    assert nth_output.output_token == expected_tokens[i][n]
                else:
                    # For non-seeded random check that one of the high-logit
                    # tokens were chosen
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

    # This time, results of seeded random samples will be compared with
    # the corresponding sample in the pre-shuffled batch
    test_sampling(model_runner)

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
    sampler = MockLogitsSampler(fake_logits)
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

    def mock_sample(probs, *args, **kwargs):
        nonlocal sample_probs
        sample_probs = probs
        return [[prob.topk(1, dim=-1).indices.tolist(), [0]] for prob in probs]

    with patch("vllm.model_executor.layers.sampler._sample", mock_sample):
        sampler(logits=fake_logits, sampling_metadata=sampling_metadata)
    hf_probs = warpers(torch.zeros_like(fake_logits), fake_logits.clone())
    hf_probs = torch.softmax(hf_probs, dim=-1, dtype=torch.float)
    assert torch.allclose(hf_probs, sample_probs, atol=1e-5)
    assert torch.equal(hf_probs.eq(0), sample_probs.eq(0))

    del model_runner
