from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        model="meta-llama/Llama-3.1-8B",
        tokenizer="meta-llama/Llama-3.1-8B",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    prompt = ["Explain why head sparsity speeds up transformer decoding."]
    out = llm.generate(prompt, SamplingParams(max_tokens=20, temperature=0.0))
    # print(out)
    print("Prompt: ", out[0].prompt)
    print(out[0].outputs[0].text)
