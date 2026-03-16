# Token Embedding Models

Token embedding corresponds to `token_embed` pooling task, offline `LLM.encode(..., pooling_task="embed")` API, and online `/pooling` API.

The difference between the (sequence) embedding task and the token embedding task is that (sequence) embedding outputs one embedding for each sequence, while token embedding outputs a embedding for each token.

Most embedding models support the token embedding task. See [this page](embed.md) for more information about embedding.

## Typical Use Cases

### Multi vector retrieval

Please refer to [examples/pooling/token_embed/multi_vector_retrieval_offline.py](../../../examples/pooling/token_embed/multi_vector_retrieval_offline.py), [examples/pooling/token_embed/multi_vector_retrieval_offline.py](../../../examples/pooling/token_embed/multi_vector_retrieval_online.py).

### Late interaction

Using the score API to compute similarity scores by calculating the late interaction of two input prompts. See [Score API](score.md) for more information.

### Extract last hidden states

You can convert models of any architecture into an embedding model via `--convert embed` and then use token_embed to extract the last hidden states.
