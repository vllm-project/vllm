"""Verify that seeded random sampling is deterministic.

Run `pytest tests/samplers/test_seeded_generate.py --forked`.
"""
import copy
import random
from itertools import combinations

import pytest

from vllm.model_executor.utils import set_random_seed
from vllm import SamplingParams

MODEL = "facebook/opt-125m"
RANDOM_SEEDS = list(range(3))


@pytest.fixture
def vllm_model(vllm_runner):
    vllm_model = vllm_runner(MODEL, dtype="half")
    yield vllm_model
    del vllm_model


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_random_sample_with_seed(
    vllm_model,
    example_prompts,
    seed: int,
) -> None:
    set_random_seed(seed)

    sampling_params = SamplingParams(
        # Parameters to ensure sufficient randomness
        temperature=2.0,
        top_p=min(random.random() + 0.3, 1),
        top_k=random.randint(5, 20),
        n=random.randint(1, 10),
        presence_penalty=random.randint(0, 1),
        max_tokens=4,
        ignore_eos=True,
    )

    sampling_params_seed_1 = copy.deepcopy(sampling_params)
    sampling_params_seed_1.seed = 100
    sampling_params_seed_2 = copy.deepcopy(sampling_params)
    sampling_params_seed_2.seed = 200

    vllm_outputs_no_seed_1 = vllm_model.generate(example_prompts,
                                                 sampling_params)
    vllm_outputs_seed_1_1 = vllm_model.generate(example_prompts,
                                                sampling_params_seed_1)
    vllm_outputs_seed_2_1 = vllm_model.generate(example_prompts,
                                                sampling_params_seed_2)
    vllm_outputs_no_seed_2 = vllm_model.generate(example_prompts,
                                                 sampling_params)
    vllm_outputs_seed_1_2 = vllm_model.generate(example_prompts,
                                                sampling_params_seed_1)
    vllm_outputs_seed_2_2 = vllm_model.generate(example_prompts,
                                                sampling_params_seed_2)

    for output_a, output_b in combinations(
        (vllm_outputs_no_seed_1, vllm_outputs_no_seed_2, vllm_outputs_seed_1_1,
         vllm_outputs_seed_2_1), 2):
        assert output_a != output_b

    assert vllm_outputs_seed_1_1 == vllm_outputs_seed_1_2
    assert vllm_outputs_seed_2_1 == vllm_outputs_seed_2_2
