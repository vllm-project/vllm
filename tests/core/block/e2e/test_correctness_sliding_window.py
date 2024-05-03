from itertools import cycle
import random

import pytest

from vllm import SamplingParams, LLM
from typing import Iterable

# relatively small model with 4k sliding window
MODEL = "bigcode/starcoder2-3b"


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
    output_len = 128
    temperature = 0.0

    # the prompt is just under 10k tokens; sliding window is 4k
    # so the answer is outside sliding window, but should still be correct

    prompts = []
    ans = []
    indices = []
    random.seed(seed)
    for _ in range(batch_size):
        idx = random.randint(30, 90)
        indices.append(idx)
        prompt = f"```python\n# We set a number of variables, x{idx} will be important later\n"
        ln = random.randint(800, 1100)
        for k in range(30, ln):
            v = random.randint(10, 99)
            if k == idx:
                ans.append(v)
            prompt += f"x{k} = {v}\n"
        prompt += f"# Now, we check the value of x{idx}:\n"
        prompt += f"assert x{idx} =="
        prompts.append(prompt)

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    print('Getting token ids from block manager v1')
    baseline_texts = get_text_from_llm_generator(baseline_llm_generator,
                                                 prompts, sampling_params)

    ans2 = [int(text[0:4].strip()) for text in baseline_texts] 
    print(list(zip(ans, ans2)))
    print(indices)

    numok = 0
    for a1, a2 in zip(ans, ans2):
        if a1 == a2:
            numok += 1
    frac_ok = numok / len(ans)
    print(f"Numok: {numok}/{len(ans)} {frac_ok}")
    assert frac_ok > 0.7

    print('Getting token ids from block manager v2')
    test_texts = get_text_from_llm_generator(test_llm_generator, prompts,
                                             sampling_params)

    for expected_text, actual_text in zip(baseline_texts, test_texts):
        assert expected_text == actual_text


def get_text_from_llm_generator(llm_generator: Iterable[LLM], prompts,
                                sampling_params):
    for llm in llm_generator:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        text = [output.outputs[0].text for output in outputs]
        del llm

    return text
