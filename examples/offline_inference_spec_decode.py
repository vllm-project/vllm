from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Speculative decoding is a method",
]
sampling_params = SamplingParams(temperature=0.0)

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    # tensor_parallel_size=1,
    # speculative_model="/data/lily/eagle-8b-instruct-model",
    # speculative_draft_tensor_parallel_size=1,
    speculative_model='[ngram]',
    ngram_prompt_lookup_max=5,
    ngram_prompt_lookup_min=3,
    num_speculative_tokens=3
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")