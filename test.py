from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
    "How are you",
    "Good morning",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m", enforce_eager=True, enable_prefix_caching=False)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
