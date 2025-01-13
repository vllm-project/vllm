from vllm import LLM

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create an LLM with built-in default generation config.
# The generation config is set to None by default to keep
# the behavior consistent with the previous version.
# If you want to use the default generation config from the model,
# you should set the generation_config to "auto".
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", generation_config="auto")

# Load the default sampling parameters from the model.
sampling_params = llm.get_default_sampling_params()
# Modify the sampling parameters if needed.
sampling_params.temperature = 0.5

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
