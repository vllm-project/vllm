from itertools import cycle
from typing import List, Tuple

import pytest

from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "speculative_model": "JackFram/llama-160m",
    "num_speculative_tokens": 3,
}])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("n_best_of", [(1, 2), (2, 2), (1, 3), (2, 3), (3, 3)])
@pytest.mark.parametrize("use_beam_search", [False, True])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        7,
    ])
@pytest.mark.parametrize("seed", [1])
def test_best_of_equality(baseline_llm_generator, test_llm_generator,
                          batch_size: int, output_len: int,
                          n_best_of: Tuple[int, int], use_beam_search: bool):
    """Validate that server with speculative model still produces correct output
    when n>1, best_of>1 or use_beam_search=True. The server will not speculate
    on these batches currently, but we should still get correct output.
    """
    run_best_of_correctness_test(baseline_llm_generator,
                                 test_llm_generator,
                                 batch_size,
                                 max_output_len=output_len,
                                 n=n_best_of[0],
                                 best_of=n_best_of[1],
                                 use_beam_search=use_beam_search)


def get_outputs_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> Tuple[List[List[str]], List[List[List[int]]]]:
    tokens: List[List[str]] = []
    token_ids: List[List[List[int]]] = []
    for llm in llm_generator():
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        token_ids = [[seq.token_ids for seq in output.outputs]
                     for output in outputs]
        tokens = [[seq.text for seq in output.outputs] for output in outputs]
        del llm

    return tokens, token_ids


def run_best_of_correctness_test(baseline_llm_generator,
                                 test_llm_generator,
                                 batch_size,
                                 max_output_len,
                                 n: int,
                                 best_of: int,
                                 use_beam_search: bool,
                                 print_tokens: bool = False):
    """Helper function of validating speculative decoding behavior when
    multiple sequences are involved.
    """

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    sampling_params = SamplingParams(
        max_tokens=max_output_len,
        ignore_eos=True,
        temperature=0.0 if use_beam_search else 1.0,
        n=n,
        best_of=best_of,
        use_beam_search=use_beam_search,
        seed=42,  # seed should be respected since spec will be disabled
    )

    spec_batch_tokens, spec_batch_token_ids = get_outputs_from_llm_generator(
        test_llm_generator, prompts, sampling_params)

    (baseline_batch_tokens,
     baseline_batch_token_ids) = get_outputs_from_llm_generator(
         baseline_llm_generator, prompts, sampling_params)

    assert len(baseline_batch_token_ids) == len(prompts)
    assert len(spec_batch_token_ids) == len(prompts)

    for baseline_group_token_ids in baseline_batch_token_ids:
        assert len(baseline_group_token_ids) == n

    for spec_group_token_ids in spec_batch_token_ids:
        assert len(spec_group_token_ids) == n

    for i, (baseline_group_token_ids, baseline_group_tokens,
            spec_group_token_ids, spec_group_tokens) in enumerate(
                zip(baseline_batch_token_ids, baseline_batch_tokens,
                    spec_batch_token_ids, spec_batch_tokens)):

        for j, (baseline_token_ids, baseline_tokens, spec_token_ids,
                spec_tokens) in enumerate(
                    zip(baseline_group_token_ids, baseline_group_tokens,
                        spec_group_token_ids, spec_group_tokens)):
            if print_tokens:
                print(f'{i=},{j=} {baseline_tokens=}')
                print(f'{i=},{j=}     {spec_tokens=}')
            print(f'{i=},{j=} {baseline_token_ids=}')
            print(f'{i=},{j=}     {spec_token_ids=}')
            assert baseline_token_ids == spec_token_ids
