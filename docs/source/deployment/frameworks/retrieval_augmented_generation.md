(deployment-retrieval-augmented-generation)=

# Retrieval-Augmented Generation

[Retrieval-augmented generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) is a technique that enables generative artificial intelligence (Gen AI) models to retrieve and incorporate new information. It modifies interactions with a large language model (LLM) so that the model responds to user queries with reference to a specified set of documents, using this information to supplement information from its pre-existing training data. This allows LLMs to use domain-specific and/or updated information. Use cases include providing chatbot access to internal company data or generating responses based on authoritative sources.

Here are the integrations:
- vLLM + [langchain](https://github.com/langchain-ai/langchain) + [milvus](https://github.com/milvus-io/milvus)
- vLLM + [llamaindex](https://github.com/run-llama/llama_index) + [milvus](https://github.com/milvus-io/milvus)

## vLLM + langchain

### Prerequisites

- Setup vLLM and langchain environment

```console
pip install -U vllm \
            langchain_milvus langchain_openai \
            langchain_community beautifulsoup4 \
            langchain-text-splitters
```

### Deploy

- Start the vLLM server with the supported embedding model, e.g.

```console
# Start embedding service (port 8000)
vllm serve ssmits/Qwen2-7B-Instruct-embed-base
```

- Start the vLLM server with the supported chat completion model, e.g.

```console
# Start chat service (port 8001)
vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
```

- Use the script: <gh-file:examples/online_serving/retrieval_augmented_generation_with_langchain.py>

```python
python retrieval_augmented_generation_with_langchain.py --help
usage: retrieval_augmented_generation_with_langchain.py [-h] [--vllm-api-key VLLM_API_KEY]
                                                        [--vllm-embedding-endpoint VLLM_EMBEDDING_ENDPOINT]
                                                        [--vllm-chat-endpoint VLLM_CHAT_ENDPOINT]
                                                        [--uri URI] [--url URL]
                                                        [--embedding-model EMBEDDING_MODEL]
                                                        [--chat-model CHAT_MODEL] [-i] [-k TOP_K]
                                                        [-c CHUNK_SIZE] [-o CHUNK_OVERLAP]

RAG Demo with vLLM and langchain

options:
  -h, --help            show this help message and exit
  --vllm-api-key VLLM_API_KEY
                        API key for vLLM compatible services
  --vllm-embedding-endpoint VLLM_EMBEDDING_ENDPOINT
                        Base URL for embedding service
  --vllm-chat-endpoint VLLM_CHAT_ENDPOINT
                        Base URL for chat service
  --uri URI             URI for Milvus database
  --url URL             URL of the document to process
  --embedding-model EMBEDDING_MODEL
                        Model name for embeddings
  --chat-model CHAT_MODEL
                        Model name for chat
  -i, --interactive     Enable interactive Q&A mode
  -k TOP_K, --top-k TOP_K
                        Number of top results to retrieve
  -c CHUNK_SIZE, --chunk-size CHUNK_SIZE
                        Chunk size for document splitting
  -o CHUNK_OVERLAP, --chunk-overlap CHUNK_OVERLAP
                        Chunk overlap for document splitting
```

- Run the script

```python
python retrieval_augmented_generation_with_langchain.py
```

## vLLM + llamaindex

### Prerequisites

- Setup vLLM and llamaindex environment

```console
pip install vllm \
            llama-index llama-index-readers-web \
            llama-index-llms-openai-like    \
            llama-index-embeddings-openai-like \
            llama-index-vector-stores-milvus \
```

### Deploy

- Start the vLLM server with the supported embedding model, e.g.

```console
# Start embedding service (port 8000)
vllm serve ssmits/Qwen2-7B-Instruct-embed-base
```

- Start the vLLM server with the supported chat completion model, e.g.

```console
# Start chat service (port 8001)
vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
```

- Use the script: <gh-file:examples/online_serving/retrieval_augmented_generation_with_llamaindex.py>

```python
python retrieval_augmented_generation_with_llamaindex.py --help
usage: retrieval_augmented_generation_with_llamaindex.py [-h] [--url URL]
                                                         [--embedding-model EMBEDDING_MODEL]
                                                         [--chat-model CHAT_MODEL]
                                                         [--vllm-api-key VLLM_API_KEY]
                                                         [--embedding-endpoint EMBEDDING_ENDPOINT]
                                                         [--chat-endpoint CHAT_ENDPOINT]
                                                         [--db-path DB_PATH] [-i]

RAG with vLLM and LlamaIndex

options:
  -h, --help            show this help message and exit
  --url URL             URL of the document to process
  --embedding-model EMBEDDING_MODEL
                        Model name for embeddings
  --chat-model CHAT_MODEL
                        Model name for chat
  --vllm-api-key VLLM_API_KEY
                        API key for vLLM compatible services
  --embedding-endpoint EMBEDDING_ENDPOINT
                        Base URL for embedding service
  --chat-endpoint CHAT_ENDPOINT
                        Base URL for chat service
  --db-path DB_PATH     Path to Milvus database
  -i, --interactive     Enable interactive Q&A mode
```

- Run the script

```python
python retrieval_augmented_generation_with_llamaindex.py
```
