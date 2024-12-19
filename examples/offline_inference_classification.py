from vllm import LLM

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create an LLM.
# You should pass task="classify" for classification models
model = LLM(
    model="jason9693/Qwen2.5-1.5B-apeach",
    task="classify",
    enforce_eager=True,
)

# Generate logits. The output is a list of ClassificationRequestOutputs.
outputs = model.classify(prompts)

# Print the outputs.
for prompt, output in zip(prompts, outputs):
    probs = output.outputs.probs
    probs_trimmed = ((str(probs[:16])[:-1] +
                      ", ...]") if len(probs) > 16 else probs)
    print(f"Prompt: {prompt!r} | "
          f"Class Probabilities: {probs_trimmed} (size={len(probs)})")
