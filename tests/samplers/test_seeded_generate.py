"""Verify that seeded random sampling is deterministic.

Run `pytest tests/samplers/test_seeded_generate.py --forked`.
"""
import copy
import random
from itertools import combinations

import pytest

from vllm import SamplingParams
from vllm.model_executor.utils import set_random_seed

MODEL = "facebook/opt-125m"
RANDOM_SEEDS = list(range(5))


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
        max_tokens=8,
        ignore_eos=True,
    )

    sampling_params_seed_1 = copy.deepcopy(sampling_params)
    sampling_params_seed_1.seed = 100
    sampling_params_seed_2 = copy.deepcopy(sampling_params)
    sampling_params_seed_2.seed = 200

    llm = vllm_model.model

    for prompt in example_prompts:
        for params in (
                sampling_params,
                sampling_params_seed_1,
                sampling_params_seed_2,
                sampling_params,
                sampling_params_seed_1,
                sampling_params_seed_2,
        ):
            llm._add_request(
                prompt=prompt,
                prompt_token_ids=None,
                sampling_params=params,
            )

    results = llm._run_engine(use_tqdm=False)
    all_outputs = [[out.token_ids for out in output.outputs]
                   for output in results]

    for i in range(0, len(example_prompts), 6):
        outputs = all_outputs[i:i + 6]

        # verify all non-seeded requests differ
        for output_a, output_b in combinations(
            (outputs[0], outputs[1], outputs[2], outputs[3]),
                2,
        ):
            assert output_a != output_b

        # verify requests with the same seed match
        assert outputs[1] == outputs[4]
        assert outputs[2] == outputs[5]
