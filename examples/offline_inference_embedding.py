from vllm import LLM
from vllm.inputs import build_decoder_prompts

# Sample prompts.
prompts = build_decoder_prompts([
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
])

# Create an LLM.
model = LLM(
    model="intfloat/e5-mistral-7b-instruct",
    enforce_eager=True,
    # NOTE: sliding_window is not supported by encoder_decoder_model
    disable_sliding_window=True,
    max_model_len = 2672,
    gpu_memory_utilization=0.95,
)
# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = model.encode(prompts)
# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of 4096 floats
