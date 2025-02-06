# NCCL_DEBUG=INFO python examples/parallelism_tests/test-qwen.py

from vllm import LLM
llm = LLM(
    model="Qwen/Qwen2-1.5B",
    task="generate",
    tensor_parallel_size=8
)
output = llm.generate("Hello, my name is")
print(output)