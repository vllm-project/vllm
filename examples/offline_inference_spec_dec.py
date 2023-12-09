import numpy as np
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

# Create an LLM.
llm = LLM(model="lmsys/vicuna-13b-v1.5",
          draft_model="TinyLlama/TinyLlama-1.1B-Chat-v0.6",
          speculate_length=5)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Mean acceptance length={mean_num_accepted}"
    )
