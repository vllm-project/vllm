import random
from typing import Iterable, List

import pytest

from vllm import LLM, SamplingParams

# relatively small model with 4k sliding window
MODEL = "bigcode/starcoder2-3b"


# the prompt is just under 10k tokens; sliding window is 4k
# so the answer is outside sliding window, but should still be correct
def prep_prompts(batch_size: int):
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
    print(f"Numok: {numok}/{len(answer)} {frac_ok}")
    assert frac_ok > 0.7


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": MODEL,

        # skip cuda graph creation for fast test.
        "enforce_eager": True,
        "block_size": 16,
        "num_gpu_blocks_override": 100000 // 16,
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
    sampling_params = SamplingParams(
        max_tokens=128,
        ignore_eos=True,
        temperature=0.0,
    )

    prompts, answer, indices = prep_prompts(batch_size)

    print('Getting token ids from block manager v1')
    baseline_texts = get_text_from_llm_generator(baseline_llm_generator,
                                                 prompts, sampling_params)

    check_answers(indices, answer, baseline_texts)

    print('Getting token ids from block manager v2')
    test_texts = get_text_from_llm_generator(test_llm_generator, prompts,
                                             sampling_params)
    check_answers(indices, answer, test_texts)

    for expected_text, actual_text in zip(baseline_texts, test_texts):
        assert expected_text == actual_text


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model": MODEL,

        # skip cuda graph creation for fast test.
        "enforce_eager": True,
        "block_size": 16,
        "num_gpu_blocks_override": 100000 // 16,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [{
    "use_v2_block_manager": True,
    "enable_chunked_prefill": True
}])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", [1])
def test_sliding_window_chunked_prefill(test_llm_generator, batch_size, seed):
    sampling_params = SamplingParams(
        max_tokens=10,
        ignore_eos=True,
        temperature=0.0,
    )

    prompts, answer, indices = prep_prompts(batch_size)

    # We don't compare with the baseline model here, since the results
    # slightly different due to different tailing in attention.
    test_texts = get_text_from_llm_generator(test_llm_generator, prompts,
                                             sampling_params)
    check_answers(indices, answer, test_texts)


def get_text_from_llm_generator(llm_generator: Iterable[LLM], prompts,
                                sampling_params):
    for llm in llm_generator:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        text = [output.outputs[0].text for output in outputs]
        del llm

    return text
