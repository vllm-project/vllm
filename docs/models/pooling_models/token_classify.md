# Token Classification Models

Token classification is supported through the `token_classify` pooling task, the offline `LLM.encode(..., pooling_task="token_classify")` API, and the online `/pooling` endpoint.

The key distinction between sequence classification and token classification lies in their output granularity: sequence classification produces a single result for an entire input sequence, whereas token classification yields a result for each individual token within the sequence.

Many classification models support both sequence classification and token classification. For further details on sequence classification, please refer to [this page](classify.md).

## Typical Use Cases

### Named Entity Recognition (NER)

For implementation examples, see:

Offline: [examples/pooling/token_classify/ner_offline.py](../../../examples/pooling/token_classify/ner_offline.py)

Online: [examples/pooling/token_classify/ner_online.py](../../../examples/pooling/token_classify/ner_online.py)

### Sparse retrieval (lexical matching)

The BAAI/bge-m3 model leverages token classification for sparse retrieval. For more information, see [this page](specific_models.md#baaibge-m3).

### Reward Models

For details on reward models, see [Reward Models](reward.md).
