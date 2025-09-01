# Retrieval-Augmented Generation

[Retrieval-augmented generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) is a technique that enables generative artificial intelligence (Gen AI) models to retrieve and incorporate new information. It modifies interactions with a large language model (LLM) so that the model responds to user queries with reference to a specified set of documents, using this information to supplement information from its pre-existing training data. This allows LLMs to use domain-specific and/or updated information. Use cases include providing chatbot access to internal company data or generating responses based on authoritative sources.

Here are the integrations:

- vLLM + [langchain](https://github.com/langchain-ai/langchain) + [milvus](https://github.com/milvus-io/milvus)
- vLLM + [llamaindex](https://github.com/run-llama/llama_index) + [milvus](https://github.com/milvus-io/milvus)

## vLLM + langchain

### Prerequisites

- Setup vLLM and langchain environment

```bash
pip install -U vllm \
            langchain_milvus langchain_openai \
            langchain_community beautifulsoup4 \
            langchain-text-splitters
```

### Deploy

- Start the vLLM server with the supported embedding model, e.g.

```bash
# Start embedding service (port 8000)
vllm serve ssmits/Qwen2-7B-Instruct-embed-base
```

- Start the vLLM server with the supported chat completion model, e.g.

```bash
# Start chat service (port 8001)
vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
```

- Use the script: <gh-file:examples/online_serving/retrieval_augmented_generation_with_langchain.py>

- Run the script

```python
python retrieval_augmented_generation_with_langchain.py
```

## vLLM + llamaindex

### Prerequisites

- Setup vLLM and llamaindex environment

```bash
pip install vllm \
            llama-index llama-index-readers-web \
            llama-index-llms-openai-like    \
            llama-index-embeddings-openai-like \
            llama-index-vector-stores-milvus \
```

### Deploy

- Start the vLLM server with the supported embedding model, e.g.

```bash
# Start embedding service (port 8000)
vllm serve ssmits/Qwen2-7B-Instruct-embed-base
```

- Start the vLLM server with the supported chat completion model, e.g.

```bash
# Start chat service (port 8001)
vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
```

- Use the script: <gh-file:examples/online_serving/retrieval_augmented_generation_with_llamaindex.py>

- Run the script

```python
python retrieval_augmented_generation_with_llamaindex.py
```
