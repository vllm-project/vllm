# Token Embedding Models

Token embedding is supported through the `token_embed` pooling task, the offline `LLM.encode(..., pooling_task="embed")` API, and the online `/pooling` endpoint.

The difference between the (sequence) embedding task and the token embedding task is that (sequence) embedding outputs one embedding for each sequence, while token embedding outputs a embedding for each token.

Many embedding models support both sequence embedding and token embedding. For further details on sequence embedding, please refer to [this page](embed.md).

## Typical Use Cases

### Multi-Vector Retrieval

For implementation examples, see:

Offline: [examples/pooling/token_embed/multi_vector_retrieval_offline.py](../../../examples/pooling/token_embed/multi_vector_retrieval_offline.py)

Online: [examples/pooling/token_embed/multi_vector_retrieval_online.py](../../../examples/pooling/token_embed/multi_vector_retrieval_online.py)

### Late interaction

Similarity scores can be computed using late interaction between two input prompts via the score API. For more information, see [Score API](score.md).

### Extract last hidden states

Models of any architecture can be converted into embedding models using `--convert embed`. Token embedding can then be used to extract the last hidden states from these models.
