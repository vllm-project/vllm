---
toc_depth: 3
---

# Engine Arguments

Engine arguments control the behavior of the vLLM engine.

- For [offline inference](../serving/offline_inference.md), they are part of the arguments to [LLM][vllm.LLM] class.
- For [online serving](../serving/online_serving/README.md), they are part of the arguments to `vllm serve`.

The engine argument classes, [EngineArgs][vllm.engine.arg_utils.EngineArgs] and [AsyncEngineArgs][vllm.engine.arg_utils.AsyncEngineArgs], are a combination of the configuration classes defined in [vllm.config][]. Therefore, if you are interested in developer documentation, we recommend looking at these configuration classes as they are the source of truth for types, defaults and docstrings.

## Sparse MLA ShardQ mode

`--attention-config.sparse_mla_shard_q_mode` accepts `default` (the default)
and `fuse_sp`. `default` uses the existing head-sharded projections and query
all-to-all. To keep SP token shards through the projections, use:

```bash
vllm serve MODEL --attention-config.sparse_mla_shard_q_mode fuse_sp
```

`fuse_sp` requires ShardQ to be enabled, NVIDIA DeepSeek-V3.2 / GLM DSA,
sequence-parallel MoE (EP with TP > 1 and DP > 1), and PP=1. It replicates Q
weights and gathers compact cache inputs, removing the hidden-state all-gather
and query all-to-all on eligible steps. Existing backend and token-threshold
checks still apply; unsupported configurations and prefill-graph capture ranges
keep the default path. Replicated Q/UK weights increase memory use.

--8<-- "gen:engine-args"
