from vllm import LLM, SamplingParams

def test_prompt_logprobs():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B")

    # stress the system by asking for prompt logprobs with a long prompt
    sampling_params = SamplingParams(top_p=0.9, top_k=50, temperature=0.8, prompt_logprobs=10, max_tokens=1)
    token_ids = list(range(2048))
    # make sure sorting does not cause OOM
    outputs = llm.generate(prompt_token_ids=token_ids, sampling_params=sampling_params)
