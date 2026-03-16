# Embedding Models

Embedding models are a class of machine learning models designed to transform unstructured data—such as text, images, or audio—into a structured numerical representation known as an embedding.

This functionality is supported through the `embed` pooling task, the `offline LLM.embed(...)` and `LLM.encode(..., pooling_task="embed")` APIs, as well as the online `/v1/embeddings` endpoint.

The primary distinction between sequence embedding and token embedding lies in their output granularity: sequence embedding produces a single embedding vector for an entire input sequence, whereas token embedding generates an embedding for each individual token within the sequence.

Many embedding models support both sequence embedding and token embedding. For further details, please refer to [this page](token_embed.md).

## Typical Use Cases

### Get embedding

The most basic use case of embedding models is get embedding.

### Get similarity scores

Using the score API to compute similarity scores by calculating the cosine similarity of two input prompt's embeddings. See [Score API](score.md) for more information.
