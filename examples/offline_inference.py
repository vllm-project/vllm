from vllm import LLM, SamplingParams
import math

llm = LLM(model="meta-llama/Meta-Llama-3-8B", enforce_eager=True)

for bs in [1, 2, 4, ] + list(range(8, 264, 8)):
    # in the beginning, we have bs number of sequences, in total about 512 tokens for the prefill
    prompt_token_ids = [[0] * math.floor(512 / bs)] * bs

    # all sequence generates 4 tokens
    sampling_params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)

    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
