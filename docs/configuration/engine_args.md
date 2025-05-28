---
title: Engine Arguments
---
[](){ #engine-args }

Engine arguments control the behavior of the vLLM engine.

- For [offline inference][offline-inference], they are part of the arguments to [LLM][vllm.LLM] class.
- For [online serving][openai-compatible-server], they are part of the arguments to `vllm serve`.

You can look at [EngineArgs][vllm.engine.arg_utils.EngineArgs] and [AsyncEngineArgs][vllm.engine.arg_utils.AsyncEngineArgs] to see the available engine arguments.

However, these classes are a combination of the configuration classes defined in [vllm.config][]. Therefore, we would recommend you read about them there where they are best documented.

For offline inference you will have access to these configuration classes and for online serving you can cross-reference the configs with `vllm serve --help`, which has its arguments grouped by config.

!!! note
    Additional arguments are available to the [AsyncLLMEngine][vllm.engine.async_llm_engine.AsyncLLMEngine] which is used for online serving. These can be found by running `vllm serve --help`
