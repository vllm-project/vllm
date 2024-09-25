from vllm import LLM, SamplingParams

# Sample prompts.
prompts = ["The president of the United States is", "How are you"]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m",
          speculative_model="facebook/opt-125m",
          num_speculative_tokens=3,
          enforce_eager=True,
          use_v2_block_manager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
