from vllm import LLM, SamplingParams


def test_gpu_memory_utilization():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # makes sure gpu_memory_utilization is per-instance limit,
    # not a global limit
    llms = [
        LLM(model="facebook/opt-125m",
            gpu_memory_utilization=0.3,
            enforce_eager=True) for i in range(3)
    ]
    for llm in llms:
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
