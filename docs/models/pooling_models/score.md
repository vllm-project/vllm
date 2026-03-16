# Score API

The score API is used to calculate similarity scores between two input prompts. It supports three types of models: cross-encoder, late-interaction, and bi-encoder.

The Score API includes the offline `LLM.score(...)` API, as well as the online `/score` scoring API and the online `/rerank`, `/v1/rerank`, `/v2/rerank` Re-rank APIs.

!!! note
    vLLM can only perform the model inference component (e.g. embedding, reranking) of RAG.
    To handle RAG at a higher level, you should use integration frameworks such as [LangChain](https://github.com/langchain-ai/langchain).
