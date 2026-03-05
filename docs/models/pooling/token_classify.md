# Token Classification

## Supported Models

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|-----------------------------|-----------------------------------------|
| `BertForTokenClassification` | bert-based | `boltuix/NeuroBERT-NER` (see note), etc. |  |  |
| `ModernBertForTokenClassification` | ModernBERT-based | `disham993/electrical-ner-ModernBERT-base` |  |  |


## Supported Features

## Offline Inference

## Online Serving

## Usage scenarios

### Named Entity Recognition (NER)

    Named Entity Recognition (NER) usage, please refer to [examples/pooling/token_classify/ner_offline.py](../../examples/pooling/token_classify/ner_offline.py), [examples/pooling/token_classify/ner_online.py](../../examples/pooling/token_classify/ner_online.py).

