# Classification Models

Classification involves predicting which predefined category, class, or label best corresponds to a given input.

This functionality is supported through the `classify` pooling task, the offline `LLM.classify(...)` and `LLM.encode(..., pooling_task="classify")` APIs, as well as the online `/classify` endpoint.

The key distinction between (sequence) classification and token classification lies in their output granularity: (sequence) classification produces a single result for an entire input sequence, whereas token classification yields a result for each individual token within the sequence.

Many classification models support both (sequence) classification and token classification. For further details on token classification, please refer to [this page](token_classify.md).

## Typical Use Cases

### Classification

The most fundamental application of classification models is to categorize input data into predefined classes.

### Reward Models

For more information, see [Reward Models](reward.md).

## Supported Models

- Text only models

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `ErnieForSequenceClassification` | BERT-like Chinese ERNIE | `Forrest20231206/ernie-3.0-base-zh-cls` | | |
| `GPT2ForSequenceClassification` | GPT2 | `nie3e/sentiment-polish-gpt2-small` | | |
| `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2-based | `jason9693/Qwen2.5-1.5B-apeach` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

!!! note
    Most [cross-encoder models](score.md#cross-encoder) can also be used as classification models.

- Multimodal Models

!!! note
    For more information about multimodal models inputs, see [this page](../supported_models.md#list-of-multimodal-language-models).

| Architecture | Models | Inputs | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `Qwen2_5_VLForSequenceClassification`<sup>C</sup> | Qwen2_5_VL-based | T + I<sup>E+</sup> + V<sup>E+</sup> | `muziyongshixin/Qwen2.5-VL-7B-for-VideoCls` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using
[as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model]. By default, the class probabilities are extracted from the softmaxed hidden state corresponding to the last token.
