import random
from typing import List
from vllm import LLM, SamplingParams
import pytest


@pytest.mark.parametrize("model", ["bigcode/starcoder2-3b"])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", [1])
def test_sliding_window_retrival(monkeypatch, model, batch_size, seed):
    """
    The test does a bunch of assignments "x1 = 10\nx2 = 33\n..." and then
    asks for value of one of them (which is outside the sliding window).
    If we tell it upfront which we are going to be looking for, then
    it answers correctly (mostly).
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        llm = LLM(model=model)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

        prompts, answer, indices = prep_prompts(batch_size)

        responses = llm.generate(prompts, sampling_params)
        check_answers(indices, answer,
                      [response.outputs[0].text for response in responses])


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
