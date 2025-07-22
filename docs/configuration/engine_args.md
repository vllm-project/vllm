---
toc_depth: 3
---

# Engine Arguments

Engine arguments control the behavior of the vLLM engine.

- For [offline inference](../serving/offline_inference.md), they are part of the arguments to [LLM][vllm.LLM] class.
- For [online serving](../serving/openai_compatible_server.md), they are part of the arguments to `vllm serve`.

The engine argument classes, [EngineArgs][vllm.engine.arg_utils.EngineArgs] and [AsyncEngineArgs][vllm.engine.arg_utils.AsyncEngineArgs], are a combination of the configuration classes defined in [vllm.config][]. Therefore, if you are interested in developer documentation, we recommend looking at these configuration classes as they are the source of truth for types, defaults and docstrings.

## `EngineArgs`

--8<-- "docs/argparse/engine_args.md"

## `AsyncEngineArgs`

--8<-- "docs/argparse/async_engine_args.md"
