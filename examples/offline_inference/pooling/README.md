# Pooling models

## Convert llm model to seq cls

```bash
# for BAAI/bge-reranker-v2-gemma
# Caution: "Yes" and "yes" are two different tokens
python examples/offline_inference/pooling/convert_model_to_seq_cls.py --model_name BAAI/bge-reranker-v2-gemma --classifier_from_tokens '["Yes"]' --method no_post_processing --path ./bge-reranker-v2-gemma-seq-cls
# for mxbai-rerank-v2
python examples/offline_inference/pooling/convert_model_to_seq_cls.py --model_name mixedbread-ai/mxbai-rerank-base-v2 --classifier_from_tokens '["0", "1"]' --method from_2_way_softmax --path ./mxbai-rerank-base-v2-seq-cls
# for Qwen3-Reranker
python examples/offline_inference/pooling/convert_model_to_seq_cls.py --model_name Qwen/Qwen3-Reranker-0.6B --classifier_from_tokens '["no", "yes"]' --method from_2_way_softmax --path ./Qwen3-Reranker-0.6B-seq-cls
```

## Embed jina_embeddings_v3 usage

Only text matching task is supported for now. See <https://github.com/vllm-project/vllm/pull/16120>

```bash
python examples/offline_inference/pooling/embed_jina_embeddings_v3.py
```

## Embed matryoshka dimensions usage

```bash
python examples/offline_inference/pooling/embed_matryoshka_fy.py
```

## Multi vector retrieval usage

```bash
python examples/offline_inference/pooling/multi_vector_retrieval.py
```

## Named Entity Recognition (NER) usage

```bash
python examples/offline_inference/pooling/ner.py
```

## Prithvi Geospatial MAE usage

```bash
python examples/offline_inference/pooling/prithvi_geospatial_mae.py
```

## IO Processor Plugins for Prithvi Geospatial MAE

```bash
python examples/offline_inference/pooling/prithvi_geospatial_mae_io_processor.py
```

## Qwen3 reranker usage

```bash
python examples/offline_inference/pooling/qwen3_reranker.py
```
