# Offline Inference

Offline inference is possible in your own code using vLLM's [`LLM`][vllm.LLM] class.

!!! info
    [API Reference](../api/README.md#offline-inference)

## Generative APIs

For further details on generative models, please refer to [this page](../models/supported_models.md).

- `LLM.generate` - Generates completions for the given input prompts.
- `LLM.chat` - Generates responses for a chat conversation.

## Asynchronous Queue APIs

- `LLM.enqueue` - Enqueues prompts for generation without waiting for completion.
- `LLM.enqueue_chat` - Enqueues chat conversations for generation without waiting.
- `LLM.wait_for_completion` - Waits for all enqueued requests to complete and returns results.

## Pooling APIs

For further details on pooling models, please refer to [this page](../models/pooling_models/README.md).

- `LLM.classify` - Only applicable to [classification models](../models/pooling_models/classify.md).
- `LLM.embed` - Only applicable to [embedding models](../models/pooling_models/embed.md).
- `LLM.score` - Applicable to [score models](../models/pooling_models/scoring.md) (cross-encoder, bi-encoder, late-interaction).
- `LLM.encode` - Applicable to all [pooling models](../models/pooling_models/README.md).

## Profiling APIs

For further details on profiling, please refer to [this page](../contributing/profiling.md).

- `LLM.start_profile` - Starts profiling with an optional custom trace prefix.
- `LLM.stop_profile` - Stops the ongoing profiling session.

## Sleep Mode APIs

For further details on sleep mode, please refer to [this page](../features/sleep_mode.md).

- `LLM.sleep` - Puts the engine into sleep mode.
- `LLM.wake_up` - Wakes up the engine from sleep mode.

## Metrics APIs

For further details on metrics, please refer to [this page](../design/metrics.md).

- `LLM.get_metrics` - Returns a snapshot of aggregated metrics from Prometheus.

## Weight Transfer APIs (RL Training)

For further details on Weight Transfer, please refer to [this page](../training/weight_transfer/README.md).

- `LLM.init_weight_transfer_engine` - Initializes the weight transfer engine for RL training.
- `LLM.start_weight_update` - Starts a new weight update cycle.
- `LLM.update_weights` - Updates the model weights.
- `LLM.finish_weight_update` - Finishes the current weight update cycle.

## Additional APIs

- `LLM.collective_rpc` - Executes a method or callable collectively across all workers.
- `LLM.apply_model` - Applies a function directly to the model inside each worker.

## Ray Data LLM API

Ray Data LLM is an alternative offline inference API that uses vLLM as the underlying engine.
This API adds several batteries-included capabilities that simplify large-scale, GPU-efficient inference:

- Streaming execution processes datasets that exceed aggregate cluster memory.
- Automatic sharding, load balancing, and autoscaling distribute work across a Ray cluster with built-in fault tolerance.
- Continuous batching keeps vLLM replicas saturated and maximizes GPU utilization.
- Transparent support for tensor and pipeline parallelism enables efficient multi-GPU inference.
- Reading and writing to most popular file formats and cloud object storage.
- Scaling up the workload without code changes.

??? code

    ```python
    import ray  # Requires ray>=2.44.1
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

    config = vLLMEngineProcessorConfig(model_source="unsloth/Llama-3.2-1B-Instruct")
    processor = build_llm_processor(
        config,
        preprocess=lambda row: {
            "messages": [
                {"role": "system", "content": "You are a bot that completes unfinished haikus."},
                {"role": "user", "content": row["item"]},
            ],
            "sampling_params": {"temperature": 0.3, "max_tokens": 250},
        },
        postprocess=lambda row: {"answer": row["generated_text"]},
    )

    ds = ray.data.from_items(["An old silent pond..."])
    ds = processor(ds)
    ds.write_parquet("local:///tmp/data/")
    ```

For more information about the Ray Data LLM API, see the [Ray Data LLM documentation](https://docs.ray.io/en/latest/data/working-with-llms.html).
