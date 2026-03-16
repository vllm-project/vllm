# Classification Models

Classification involves predicting which predefined category, class, or label best corresponds to a given input.

This functionality is supported through the `classify` pooling task, the offline `LLM.classify(...)` and `LLM.encode(..., pooling_task="classify")` APIs, as well as the online `/classify` endpoint.

The key distinction between sequence classification and token classification lies in their output granularity: sequence classification produces a single result for an entire input sequence, whereas token classification yields a result for each individual token within the sequence.

Many classification models support both sequence classification and token classification. For further details, please refer to [this page](token_classify.md).

## Typical Use Cases

### Classification

The most fundamental application of classification models is to categorize input data into predefined classes.

### Reward Models

For more information, see [Reward Models](reward.md).
