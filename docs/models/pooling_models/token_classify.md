# Token Classification Models

Token classification corresponds to `token_classify` pooling task, offline `LLM.encode(..., pooling_task="token_classify")` API, and online `/pooling` API.

The difference between the (sequence) classification task and the token classification task is that (sequence) classification outputs one result for each sequence, while token classification outputs a result for each token.

Most classification models support the token classification task. See [this page](classify.md) for more information about (sequence) classification.

## Typical Use Cases

### Named Entity Recognition (NER)

Please refer to [examples/pooling/token_classify/ner_offline.py](../../../examples/pooling/token_classify/ner_offline.py), [examples/pooling/token_classify/ner_online.py](../../../examples/pooling/token_classify/ner_online.py).

### Sparse retrieval (lexical matching)

The `BAAI/bge-m3` model uses Token Classification to achieve sparse retrieval. See [this page](specific_models.md#baaibge-m3) for more information.

### Reward Models

See [Reward Models](reward.md) for more information.
