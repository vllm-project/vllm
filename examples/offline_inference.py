from vllm import LLM, SamplingParams
import math

llm = LLM(model="meta-llama/Meta-Llama-3-8B", enforce_eager=True)

bs = 4
# in the beginning, we have bs number of sequences, in total about 512 tokens for the prefill
prompt_token_ids = [[0] * math.floor(512 / bs)] * bs

# all sequence generates 30 tokens
sampling_params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
