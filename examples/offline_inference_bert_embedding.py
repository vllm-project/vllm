from vllm import LLM
from vllm.inputs import build_encoder_prompt

# Sample prompts.
prompts = [build_encoder_prompt("This is an example sentence.")]

# Create an LLM.
model = LLM(model="bert-base-uncased", enforce_eager=True)
outputs = model.encode(prompts)

# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of 768 floats
    print(len(output.outputs.embedding))
