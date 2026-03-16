# Score API

The Score API is designed to compute similarity scores between two input prompts. It supports three model architectures: `cross-encoder`, `late-interaction`, and `bi-encoder` models.

This functionality is supported through the offline `LLM.score(...)` API, along with several online endpoints: the `/score` endpoint and the Re-rank endpoints available at `/rerank`, `/v1/rerank`, and `/v2/rerank`.

!!! note
    vLLM handles only the model inference component of RAG pipelines (such as embedding generation and reranking). For higher-level RAG orchestration, you should leverage integration frameworks like [LangChain](https://github.com/langchain-ai/langchain).
