# SPDX-License-Identifier: Apache-2.0

from vllm import LLM

rope_theta = 1000
factor = 4.0
original_max_position_embeddings = 2048

# Use yarn to extend context
hf_overrides = {
    "rope_theta": rope_theta,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max_position_embeddings
    },
    "max_model_len": int(original_max_position_embeddings * factor)
}

llm = LLM(model="nomic-ai/nomic-embed-text-v1",
          trust_remote_code=True,
          task="embed",
          hf_overrides=hf_overrides)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.embed(prompts)

print("\nGenerated Outputs:\n" + "-" * 60)
for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    embeds_trimmed = ((str(embeds[:16])[:-1] +
                       ", ...]") if len(embeds) > 16 else embeds)
    print(
        f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})"
    )
    print("-" * 60)
