from vllm.wde.entrypoints.llm import LLM

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

llm = LLM(model="Qwen/Qwen2-0.5B-Instruct", output_last_hidden_states=True)

outputs = llm.encode(prompts)
for output in outputs:
    print(output.outputs.shape)
