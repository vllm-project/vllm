from vllm.wde.entrypoints.llm import LLM

pairs = [['query', 'passage'], ['what is panda?', 'hi'],
         [
             'what is panda?', 'The giant panda (Ailuropoda melanoleuca), '
             'sometimes called a panda bear or simply panda, '
             'is a bear species endemic to China.'
         ]]

llm = LLM(model="BAAI/bge-reranker-v2-m3")

outputs = llm.reranker(pairs)
for output in outputs:
    print(output.score)
