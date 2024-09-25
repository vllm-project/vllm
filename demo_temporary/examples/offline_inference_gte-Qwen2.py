from vllm.wde.entrypoints.llm import LLM

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Because gte-Qwen2 and Qwen2 use the same architecture name
# Qwen2ForCausalLM, So you need to manually switch to
# gte-Qwen2 using switch_to_gte_Qwen2.

# Output warning:
try:
    llm = LLM(model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
              switch_to_gte_Qwen2=False)
except ValueError:
    pass
    # wde does not yet have an integrated generation model

# You should use it like this
llm = LLM(model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
          switch_to_gte_Qwen2=True)

outputs = llm.encode(prompts)
for output in outputs:
    print(output.outputs)
