from vllm.wde.entrypoints.llm import LLM

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# You should use it like this
llm = LLM(model="Alibaba-NLP/gte-Qwen2-7B-instruct")

outputs = llm.encode(prompts)
for output in outputs:
    print(output.outputs.shape)
