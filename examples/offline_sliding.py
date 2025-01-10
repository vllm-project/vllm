import random
from typing import List
from vllm import LLM, SamplingParams


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
        ln = random.randint(600, 800)
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


# Sample prompts.
prompts, answer, indices = prep_prompts(1)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

# Create an LLM.
llm = LLM(model="google/gemma-2-9b-it", enforce_eager=True)
# llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
check_answers(indices, answer,
              [response.outputs[0].text for response in outputs])
