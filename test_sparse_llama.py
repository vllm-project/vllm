from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=2,
        enforce_eager=True,
    )

    prompt = ["Explain why head sparsity speeds up transformer decoding.", "How does head sparsity work?", 
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
            "What is the capital of the United States?"]
    out = llm.generate(prompt, SamplingParams(max_tokens=50, temperature=0.0))

    print("-" * 50)
    for output in out:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)