# Pooling models

## Cohere rerank usage

```bash
# vllm serve BAAI/bge-reranker-base
python examples/online_serving/pooling/cohere_rerank_client.py
```

## Embedding requests base64 encoding_format usage

```bash
# vllm serve intfloat/e5-small
python examples/online_serving/pooling/embedding_requests_base64_client.py
```

## Embedding requests bytes encoding_format usage

```bash
# vllm serve intfloat/e5-small
python examples/online_serving/pooling/embedding_requests_bytes_client.py
```

## Jinaai rerank usage

```bash
# vllm serve BAAI/bge-reranker-base
python examples/online_serving/pooling/jinaai_rerank_client.py
```

## Multi vector retrieval usage

```bash
# vllm serve BAAI/bge-m3
python examples/online_serving/pooling/multi_vector_retrieval_client.py
```

## Named Entity Recognition (NER) usage

```bash
# vllm serve boltuix/NeuroBERT-NER
python examples/online_serving/pooling/ner_client.py
```

## OpenAI chat embedding for multimodal usage

```bash
python examples/online_serving/pooling/openai_chat_embedding_client_for_multimodal.py
```

## OpenAI classification usage

```bash
# vllm serve jason9693/Qwen2.5-1.5B-apeach
python examples/online_serving/pooling/openai_classification_client.py
```

## OpenAI cross_encoder score usage

```bash
# vllm serve BAAI/bge-reranker-v2-m3
python examples/online_serving/pooling/openai_cross_encoder_score.py
```

## OpenAI cross_encoder score for multimodal usage

```bash
# vllm serve jinaai/jina-reranker-m0
python examples/online_serving/pooling/openai_cross_encoder_score_for_multimodal.py
```

## OpenAI embedding usage

```bash
# vllm serve intfloat/e5-small
python examples/online_serving/pooling/openai_embedding_client.py
```

## OpenAI embedding matryoshka dimensions usage

```bash
# vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
python examples/online_serving/pooling/openai_embedding_matryoshka_fy.py
```

## OpenAI pooling usage

```bash
# vllm serve internlm/internlm2-1_8b-reward --trust-remote-code
python examples/online_serving/pooling/openai_pooling_client.py
```

## Online Prithvi Geospatial MAE usage

```bash
python examples/online_serving/pooling/prithvi_geospatial_mae.py
```
