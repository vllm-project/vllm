from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "hi" * 90000,
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)

# Create an LLM.
llm = LLM(model="mistralai/Mistral-7B-v0.1", max_model_len=160000)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    print(f"Prompt len: {len(output.prompt_token_ids)}, Generated text: {output.outputs[0].text!r}")
