from vllm import LLM, SamplingParams
from vllm.embedding_params import EmbeddingParams
import numpy as np
# Sample prompts.
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]
# Create a sampling params object.
embedding_params = EmbeddingParams()

# Create an LLM.
# llm = LLM(model="facebook/opt-125m")
llm = LLM(model="BAAI/bge-m3")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs1 = llm.encode(sentences_1, embedding_params)
lst1 = []
for output1 in outputs1:
    generated_text = output1.embedding.cpu()
    lst1.append(np.array(generated_text))
lst1 = np.array(lst1)

outputs2 = llm.generate(sentences_2, embedding_params)
lst2 = []
for output2 in outputs2:
    prompt = output2.prompt
    generated_text = output2.embedding.cpu()
    lst2.append(np.array(generated_text))
lst2 = np.array(lst2)
result = lst1 @ lst2.T

expected_result = np.array([[0.6265, 0.3477], [0.3499, 0.678 ]])
assert(np.isclose(result, expected_result, atol=1e-2).all())
print("Passed!")
