import itertools
import random
from array import array
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import GenerationConfig, GenerationMixin

import vllm.envs as envs
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import (VLLM_TOKEN_ID_ARRAY_TYPE, SamplingParams,
                           SequenceData, SequenceGroupMetadata)
from vllm.utils import Counter, is_pin_memory_available


class MockLogitsSampler(Sampler):

    def __init__(self, fake_logits: torch.Tensor):
        super().__init__()
        self.fake_logits = fake_logits

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


def _prepare_test(
        batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, MockLogitsSampler]:
    input_tensor = torch.rand((batch_size, 1024), dtype=torch.float16)
    fake_logits = torch.full((batch_size, VOCAB_SIZE),
                             1e-2,
                             dtype=input_tensor.dtype)
    sampler = MockLogitsSampler(fake_logits)
    return input_tensor, fake_logits, sampler


VOCAB_SIZE = 32000
RANDOM_SEEDS = list(range(128))
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


def _do_sample(
    batch_size: int,
    input_tensor: torch.Tensor,
    sampler: MockLogitsSampler,
    sampling_params: SamplingParams,
    device: str,
):
    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    seq_lens: List[int] = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={
                    0: SequenceData(array(VLLM_TOKEN_ID_ARRAY_TYPE, [1, 2, 3]))
                },
                sampling_params=sampling_params,
                block_tables={0: [1]},
            ))
        seq_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = SamplingMetadata.prepare(
        seq_group_metadata_list,
        seq_lens,
        query_lens=seq_lens,
        device=device,
        pin_memory=is_pin_memory_available())
    return sampler(logits=input_tensor, sampling_metadata=sampling_metadata)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_greedy(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler = _prepare_test(batch_size)

    sampling_params = SamplingParams(temperature=0)
    sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                sampling_params, device)
    expected = torch.argmax(fake_logits, dim=-1)
    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == expected[i].item()


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_random(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    _, fake_logits, sampler = _prepare_test(batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
    )
    sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                sampling_params, device)

    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == i


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_random_seed(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    _, fake_logits, sampler = _prepare_test(batch_size)

    for i in range(batch_size):
        fake_logits[i, i] = 1e2

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
        seed=random.randint(0, 10000),
    )
    sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                sampling_params, device)

    for i, sequence_output in enumerate(sampler_output):
        for nth_output in sequence_output.samples:
            assert nth_output.output_token == i


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_random_seed_deterministic(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    _, fake_logits, sampler = _prepare_test(batch_size)

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
        seed=random.randint(0, 10000),
    )
    first_sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                      sampling_params, device)

    second_sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                       sampling_params, device)

    assert first_sampler_output == second_sampler_output


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_all_beam(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    _, fake_logits, sampler = _prepare_test(batch_size)

    sampling_params = SamplingParams(
        temperature=0,
        best_of=2,
        use_beam_search=True,
    )
    _do_sample(batch_size, fake_logits, sampler, sampling_params, device)
    # no assertion here as I am not sure how to determine whether
    # the outputs are expected - in other words, this just tests
    # whether there are no exceptions in the sampler
    # when handling an all-beam search case.


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_min_tokens_penalty(seed: int, device: str):
    seq_id_counter = Counter(start=random.randint(0, 100))
    set_random_seed(seed)
    torch.set_default_device(device)

    def create_sampling_params(min_tokens,
                               eos_token_id=0,
                               *,
                               stop_token_ids: Optional[List[int]] = None,
                               prompt_logprobs: Optional[int] = None):
        sampling_params = SamplingParams(
            min_tokens=min_tokens,
            max_tokens=9999,  # keep higher than max of min_tokens
            stop_token_ids=stop_token_ids,
            # requesting prompt_logprobs changes the structure of `logits`
            prompt_logprobs=prompt_logprobs,
        )
        sampling_params.all_stop_token_ids.add(eos_token_id)
        return sampling_params

    def create_sequence_data(num_input=3, num_generated=0):
        seq_data = SequenceData(
            array(VLLM_TOKEN_ID_ARRAY_TYPE,
                  random.choices(range(0, VOCAB_SIZE), k=num_input)))
        if num_generated > 0:
            seq_data.output_token_ids = random.choices(range(0, VOCAB_SIZE),
                                                       k=num_generated)
        return seq_data

    def generate_test_case():
        # generate multiple seq groups but limit total batch size
        batch_size = random.randint(1, 128)

        expected_penalization = []
        sequence_metadata_list: List[SequenceGroupMetadata] = []
        # 20% chance to generate seq group metadata list with all prompts
        is_prompt = random.random() < 0.2
        while batch_size > 0:
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

            seq_data: Dict[int, SequenceData] = {}
            seq_group_penalization: List[bool] = []
            for _ in range(num_seqs):
                num_input = random.randint(1, 100)
                num_generated = 0 if is_prompt else random.randint(1, 100)
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

    prompt_with_penalization_and_prompt_logprobs = {
        "expected_penalization": [False, False, True],
        "seq_group_metadata_list": [
            SequenceGroupMetadata(
                request_id="test_1",
                is_prompt=True,
                seq_data={
                    next(seq_id_counter): create_sequence_data(num_input=3),
                },
                sampling_params=create_sampling_params(1, prompt_logprobs=3),
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
    prompt_combination = {
        "expected_penalization": [False, True, False],
        "seq_group_metadata_list": [
            SequenceGroupMetadata(
                request_id="test_2",
                is_prompt=True,
                seq_data={
                    next(seq_id_counter): create_sequence_data(num_input=2),
                },
                sampling_params=create_sampling_params(1, prompt_logprobs=3),
                block_tables={},
            ),
            SequenceGroupMetadata(
                request_id="test_3",
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

    stop_token_ids = [1, 999, 37, 37]  # intentional duplication
    decode_combination = {
        "expected_penalization": [True, False, False, True, False],
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
                is_prompt=False,
                seq_data={
                    next(seq_id_counter):
                    create_sequence_data(num_generated=20),
                    next(seq_id_counter):
                    create_sequence_data(num_generated=1),
                    next(seq_id_counter):
                    create_sequence_data(num_generated=10),
                },
                sampling_params=create_sampling_params(
                    10, prompt_logprobs=5, stop_token_ids=stop_token_ids),
                block_tables={},
            ),
        ]
    }

    if seed == 0:
        test_cases = [
            prompt_without_penalization,
            prompt_with_penalization,
            prompt_with_penalization_and_prompt_logprobs,
            stop_penalizing_after_min_tokens,
            prompt_combination,
            decode_combination,
        ]
    else:
        test_cases = [generate_test_case()]

    def run_test_case(*, expected_penalization: List[bool],
                      seq_group_metadata_list: List[SequenceGroupMetadata]):
        assert expected_penalization, \
            "Invalid test case, need expected_penalization"
        assert seq_group_metadata_list, \
            "Invalid test case, need seq_group_metadata_list"

        batch_size = 0
        seq_lens: List[int] = []
        sampling_params_per_row: List[SamplingParams] = []
        for sgm in seq_group_metadata_list:
            sampling_params = sgm.sampling_params

            num_rows = len(sgm.seq_data)
            if sgm.is_prompt:
                # a prompt seq_group has only one sequence
                seq_data = next(iter(sgm.seq_data.values()))
                prompt_len = seq_data.get_prompt_len()
                seq_lens.append(prompt_len)

                assert sgm.sampling_params is not None
                if sgm.sampling_params.prompt_logprobs:
                    # with prompt_logprobs each token in the prompt has a row in
                    # logits
                    num_rows = prompt_len

            batch_size += num_rows
            sampling_params_per_row.extend(
                itertools.repeat(sampling_params, num_rows))

        assert len(
            expected_penalization
        ) == batch_size, \
            ("Invalid test case, expected_penalization does not match computed"
             "batch size")

        _, fake_logits, sampler = _prepare_test(batch_size)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens=seq_lens if seq_lens else None,
            query_lens=seq_lens if seq_lens else None,
            device=device,
            pin_memory=is_pin_memory_available())
        # the logits tensor is modified in-place by the sampler
        _ = sampler(logits=fake_logits, sampling_metadata=sampling_metadata)

        for logits_idx, (should_penalize, sampling_params) in enumerate(
                zip(expected_penalization, sampling_params_per_row)):

            tokens_to_check = sampling_params.all_stop_token_ids

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

    for test_case in test_cases:
        run_test_case(**test_case)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_mixed(seed: int, device: str):
    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    input_tensor, fake_logits, sampler = _prepare_test(batch_size)

    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    expected_tokens: List[Optional[List[int]]] = []
    seq_lens: List[int] = []
    for i in range(batch_size):
        expected: Optional[List[int]] = None
        sampling_type = random.randint(0, 3)
        if sampling_type == 0:
            sampling_params = SamplingParams(temperature=0)
            expected = [int(torch.argmax(fake_logits[i], dim=-1).item())]
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
                seq_data={
                    0: SequenceData(array(VLLM_TOKEN_ID_ARRAY_TYPE, [1, 2, 3]))
                },
                sampling_params=sampling_params,
                block_tables={0: [1]},
            ))
        seq_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    generators: Dict[str, torch.Generator] = {}

    def test_sampling():
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            query_lens=seq_lens,
            device=device,
            pin_memory=is_pin_memory_available(),
            generators=generators)
        sampler_output = sampler(logits=fake_logits,
                                 sampling_metadata=sampling_metadata)

        for i, (sequence_output, metadata) in enumerate(
                zip(sampler_output, seq_group_metadata_list)):
            assert metadata.sampling_params is not None

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

            expected_tokens_item = expected_tokens[i]
            assert expected_tokens_item is not None

            for n, nth_output in enumerate(sequence_output.samples):
                assert metadata.sampling_params is not None

                if (metadata.sampling_params.temperature == 0
                        or metadata.sampling_params.seed is not None):
                    # Ensure exact matches for greedy or random with seed
                    assert nth_output.output_token == expected_tokens_item[n]
                else:
                    # For non-seeded random check that one of the high-logit
                    # tokens were chosen
                    assert nth_output.output_token in expected_tokens_item

    # Test batch
    test_sampling()

    # Shuffle the batch and resample
    target_index = list(range(batch_size))
    for list_to_shuffle in (target_index, seq_group_metadata_list,
                            expected_tokens, seq_lens):
        random.Random(seed).shuffle(list_to_shuffle)
    target_index = torch.tensor(target_index)
    input_tensor.data = input_tensor.index_select(0, target_index)
    fake_logits.data = fake_logits.index_select(0, target_index)

    # This time, results of seeded random samples will be compared with
    # the corresponding sample in the pre-shuffled batch
    test_sampling()


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

    generation_model = GenerationMixin()
    generation_config = GenerationConfig(top_k=top_k,
                                         top_p=top_p,
                                         do_sample=True)
    warpers = generation_model._get_logits_warper(generation_config, device)
    assert len(warpers) == 2  # top_p and top_k

    seq_group_metadata_list: List[SequenceGroupMetadata] = []
    seq_lens: List[int] = []
    for i in range(batch_size):
        seq_group_metadata_list.append(
            SequenceGroupMetadata(
                request_id=f"test_{i}",
                is_prompt=True,
                seq_data={
                    0: SequenceData(array(VLLM_TOKEN_ID_ARRAY_TYPE, [1, 2, 3]))
                },
                sampling_params=SamplingParams(
                    temperature=1,
                    top_k=top_k,
                    top_p=top_p,
                ),
                block_tables={0: [1]},
            ))
        seq_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

    sampling_metadata = SamplingMetadata.prepare(
        seq_group_metadata_list,
        seq_lens,
        query_lens=seq_lens,
        device=device,
        pin_memory=is_pin_memory_available())

    sample_probs = None

    def mock_sample(probs, *args, **kwargs):
        nonlocal sample_probs
        sample_probs = probs
        return ([[prob.topk(1, dim=-1).indices.tolist(), [0]]
                 for prob in probs], None)

    # top-k and top-p is only calculated when flashinfer kernel is not available
    with patch("vllm.model_executor.layers.sampler._sample", mock_sample), \
         patch("vllm.model_executor.layers.sampler."
               "flashinfer_top_k_top_p_sampling", None):
        sampler(logits=fake_logits, sampling_metadata=sampling_metadata)

    assert sample_probs is not None

    hf_probs = warpers(torch.zeros_like(fake_logits), fake_logits.clone())
    hf_probs = torch.softmax(hf_probs, dim=-1, dtype=torch.float)
    torch.testing.assert_close(hf_probs, sample_probs, rtol=0.0, atol=1e-5)
    assert torch.equal(hf_probs.eq(0), sample_probs.eq(0))


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_flashinfer_fallback(seed: int, device: str):
    if not envs.VLLM_USE_FLASHINFER_SAMPLER:
        pytest.skip("Flashinfer sampler is disabled")

    set_random_seed(seed)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    _, fake_logits, sampler = _prepare_test(batch_size)

    def failing_flashinfer_sampling(*_args, **_kwargs):
        return None, torch.zeros(batch_size, device=device, dtype=torch.int32)

    sampling_params = SamplingParams(
        temperature=1.0,
        n=random.randint(1, 10),
        seed=random.randint(0, 10000),
    )
    sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                sampling_params, device)

    with patch(
            "vllm.model_executor.layers.sampler."
            "flashinfer_top_k_top_p_sampling", failing_flashinfer_sampling):
        fallback_sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                             sampling_params, device)

    assert sampler_output == fallback_sampler_output


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_repetition_penalty_mixed(device: str):

    vocab_size = 8

    def test_sampling_params(sampling_params: List[SamplingParams]):

        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        seq_lens: List[int] = []
        for i in range(2):
            seq_group_metadata_list.append(
                SequenceGroupMetadata(
                    request_id=f"test_{i}",
                    is_prompt=True,
                    seq_data={
                        0:
                        SequenceData(array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                           [1, 2, 3]))
                    },
                    sampling_params=sampling_params[i],
                    block_tables={0: [1]},
                ))
            seq_lens.append(seq_group_metadata_list[-1].seq_data[0].get_len())

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            query_lens=seq_lens,
            device=device,
            pin_memory=is_pin_memory_available())

        fake_logits = torch.full((2, vocab_size),
                                 1e-2,
                                 device=device,
                                 dtype=torch.float16)

        fake_logits[:, 5] = 1.1e-2
        fake_logits[:, 1] = 1.2e-2

        sampler = MockLogitsSampler(fake_logits)

        sampler_output = sampler(logits=fake_logits,
                                 sampling_metadata=sampling_metadata)

        generated_tokens = []
        for output in sampler_output:
            generated_tokens.append(output.samples[0].output_token)

        return generated_tokens

    # one configuration is greedy with repetition_penalty
    sampling_params_rep = SamplingParams(
        temperature=0.0,
        repetition_penalty=2.0,
    )

    # other configuration is sampling w/o repetition_penalty
    sampling_params_sample = SamplingParams(
        temperature=1.0,
        top_k=1,
        seed=42,
    )

    tokens1 = test_sampling_params(
        [sampling_params_rep, sampling_params_sample])

    tokens2 = test_sampling_params(
        [sampling_params_sample, sampling_params_rep])

    assert tokens1[0] == tokens2[1]
    assert tokens1[1] == tokens2[0]


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_sampler_include_gpu_probs_tensor(device: str):
    set_random_seed(42)
    torch.set_default_device(device)
    batch_size = random.randint(1, 256)
    _, fake_logits, sampler = _prepare_test(batch_size)
    sampler.include_gpu_probs_tensor = True
    sampler.should_modify_greedy_probs_inplace = False

    sampling_params = SamplingParams(temperature=0)

    mock_inplace = Mock()
    with patch(
            "vllm.model_executor.layers.sampler._modify_greedy_probs_inplace",
            mock_inplace):

        sampler_output = _do_sample(batch_size, fake_logits, sampler,
                                    sampling_params, device)
        mock_inplace.assert_not_called()

    assert sampler_output.sampled_token_probs is not None
    assert sampler_output.logprobs is not None
    assert sampler_output.sampled_token_ids is not None
