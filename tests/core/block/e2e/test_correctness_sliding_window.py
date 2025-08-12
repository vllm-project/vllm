import random
from typing import List

import pytest

from vllm import LLM, SamplingParams

from .conftest import get_text_from_llm_generator

# relatively small model with 4k sliding window
MODEL = "bigcode/starcoder2-3b"
BLOCK_SIZE = 16


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": MODEL,

        # skip cuda graph creation for fast test.
        "enforce_eager": True,
        "block_size": BLOCK_SIZE,
        # needed due to https://github.com/vllm-project/vllm/issues/1908#issuecomment-2101122008
        "num_gpu_blocks_override": 100000 // BLOCK_SIZE,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{
    "use_v2_block_manager": False
}])
@pytest.mark.parametrize("test_llm_kwargs", [{"use_v2_block_manager": True}])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", [1])
def test_sliding_window_retrival(baseline_llm_generator, test_llm_generator,
                                 batch_size, seed):
    """
    The test does a bunch of assignments "x1 = 10\nx2 = 33\n..." and then
    asks for value of one of them (which is outside the sliding window).
    If we tell it upfront which we are going to be looking for, then
    it answers correctly (mostly).

    Additionally, we compare the results of the v1 and v2 managers.
    """
    sampling_params = SamplingParams(
        max_tokens=1024,
        ignore_eos=True,
        temperature=0.0,
    )

    prompts, answer, indices = prep_prompts(batch_size)

    print('Getting token ids from block manager v1')
    baseline_texts = get_text_from_llm_generator(baseline_llm_generator,
                                                 prompts,
                                                 sampling_params,
                                                 llm_cb=check_window(prompts))

    check_answers(indices, answer, baseline_texts)

    print('Getting token ids from block manager v2')
    test_texts = get_text_from_llm_generator(test_llm_generator, prompts,
                                             sampling_params)
    check_answers(indices, answer, test_texts)

    cmp = [
        expected_text == actual_text
        for expected_text, actual_text in zip(baseline_texts, test_texts)
    ]
    print(cmp)
    # make sure it's mostly OK; this is possibly because https://github.com/vllm-project/vllm/pull/4768
    # however, https://github.com/vllm-project/vllm/issues/3385#issuecomment-1995924290
    # states that xformers and flash_attn have different ideas about the window
    # size anyways
    assert sum(cmp) > 0.7 * len(cmp)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": MODEL,

        # skip cuda graph creation for fast test.
        "enforce_eager": True,
        "block_size": BLOCK_SIZE,
        "num_gpu_blocks_override": 100000 // BLOCK_SIZE,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "use_v2_block_manager": True,
    "enable_chunked_prefill": True
}])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", [1])
def test_sliding_window_chunked_prefill(test_llm_generator, batch_size, seed):
    """
    This is similar to test_sliding_window_retrival, however, it doesn't
    compare against the v1 block manager since v1 doesn't support
    chunked prefill with sliding window.

    The results with and without chunked prefill are not the same due to
    numerical instabilities.
    """
    sampling_params = SamplingParams(
        max_tokens=10,
        ignore_eos=True,
        temperature=0.0,
    )

    prompts, answer, indices = prep_prompts(batch_size)

    # We don't compare with the baseline model here, since the results
    # slightly different due to different tailing in attention.
    test_texts = get_text_from_llm_generator(test_llm_generator,
                                             prompts,
                                             sampling_params,
                                             llm_cb=check_window(prompts))
    check_answers(indices, answer, test_texts)


def prep_prompts(batch_size: int):
    """
    Generate prompts which a bunch of assignments,
    then asking for the value of one of them.
    The prompt is just under 10k tokens; sliding window is 4k
    so the answer is outside sliding window, but should still be correct.
    """
    prompts: List[str] = []
    answer: List[int] = []
    indices: List[int] = []
    random.seed(1)
    for _ in range(batch_size):
        idx = random.randint(30, 90)
        indices.append(idx)
        prompt = "```python\n# We set a number of variables, " + \
                 f"x{idx} will be important later\n"
        ln = random.randint(800, 1100)
        for k in range(30, ln):
            v = random.randint(10, 99)
            if k == idx:
                answer.append(v)
            prompt += f"x{k} = {v}\n"
        prompt += f"# Now, we check the value of x{idx}:\n"
        prompt += f"assert x{idx} == "
        prompts.append(prompt)
    return prompts, answer, indices


def check_answers(indices: List[int], answer: List[int], outputs: List[str]):
    answer2 = [int(text[0:2].strip()) for text in outputs]
    print(list(zip(indices, zip(answer, answer2))))
    numok = 0
    for a1, a2 in zip(answer, answer2):
        if a1 == a2:
            numok += 1
    frac_ok = numok / len(answer)
    print(f"Num OK: {numok}/{len(answer)} {frac_ok}")
    assert frac_ok > 0.7


def check_window(prompts: List[str]):

    def inner(llm: LLM):
        sliding_window = llm.llm_engine.model_config.get_sliding_window()
        assert sliding_window and sliding_window > 0
        assert any(
            len(llm.get_tokenizer().tokenize(prompt)) > sliding_window
            for prompt in prompts)

    return inner
