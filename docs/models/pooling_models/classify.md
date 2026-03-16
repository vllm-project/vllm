# Classification Models

Classification is the task of predicting which of predefined categories or classes or labels the input data belongs to.

Classification corresponds to `classify` pooling task, offline `LLM.classify(...)`, `LLM.encode(..., pooling_task="classify")` API, and online `/classify` API.

The difference between the (sequence) classification task and the token classification task is that (sequence) classification outputs one result for each sequence, while token classification outputs a result for each token.

Most classification models support the token classification task. See [this page](token_classify.md) for more information about token classification.

## Typical Use Cases

### Classification

The most basic use case of Classification is to classify.

### Reward Models

See [Reward Models](reward.md) for more information.
