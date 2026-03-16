# Embedding Models

Embedding model is a type of machine learning model designed to convert unstructured data—like text, images, or audio—into a structured, numerical format called an embedding.

Embedding corresponds to `embed` pooling task, offline `LLM.embed(...)`, `LLM.encode(..., pooling_task="embed")` API, and online `/v1/embeddings` API.

The difference between the (sequence) embedding task and the token embedding task is that (sequence) embedding outputs one embedding for each sequence, while token embedding outputs a embedding for each token.

Most embedding models support the token embedding task. See [this page](token_embed.md) for more information about token embedding.

## Typical Use Cases

### Get embedding

The most basic use case of embedding models is get embedding.

### Get similarity scores

Using the score API to compute similarity scores by calculating the cosine similarity of two input prompt's embeddings. See [Score API](score.md) for more information.
