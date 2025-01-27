"""
Example of using the OpenAI entrypoint's rerank API which is compatible with
the Cohere SDK: https://github.com/cohere-ai/cohere-python

run: vllm serve BAAI/bge-reranker-base
"""
import cohere

# cohere v1 client
co = cohere.Client(base_url="http://localhost:8000", api_key="sk-fake-key")
rerank_v1_result = co.rerank(
    model="BAAI/bge-reranker-base",
    query="What is the capital of France?",
    documents=[
        "The capital of France is Paris", "Reranking is fun!",
        "vLLM is an open-source framework for fast AI serving"
    ])

print(rerank_v1_result)

# or the v2
co2 = cohere.ClientV2("sk-fake-key", base_url="http://localhost:8000")

v2_rerank_result = co2.rerank(
    model="BAAI/bge-reranker-base",
    query="What is the capital of France?",
    documents=[
        "The capital of France is Paris", "Reranking is fun!",
        "vLLM is an open-source framework for fast AI serving"
    ])

print(v2_rerank_result)
