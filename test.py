from vllm import LLM

llm = LLM(model='gpt2')  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)