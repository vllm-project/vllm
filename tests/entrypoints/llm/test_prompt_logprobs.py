from vllm import LLM, SamplingParams


def test_prompt_logprobs():
    llm = LLM(model="meta-llama/Meta-Llama-3-8B")

    # stress the system by asking for prompt logprobs with a long prompt
    sampling_params = SamplingParams(top_p=0.9,
                                     top_k=50,
                                     temperature=0.8,
                                     prompt_logprobs=10,
                                     max_tokens=1)
    # right now we use chunked sort and chunked logprobs to reduce
    # the peak memory, it reduces the peak memory, however, they cannot
    # make sure runtime peak memory <= profiling peak memory.
    # To fully solve this issue (i.e. we can use 8192 to test prompt logprobs),
    # we need to make sure the whole sampling process is chunked.
    token_ids = list(range(1024))
    # make sure sorting does not cause OOM
    llm.generate(prompt_token_ids=token_ids, sampling_params=sampling_params)
