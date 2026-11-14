# Scoped CPU threads for default weight loading

`VLLM_LOAD_TORCH_THREADS` is an optional positive integer cap on Torch intra-op
threads while `DefaultModelLoader.load_weights` runs. It defaults to `0`
(disabled). If the existing thread count is already at or below the cap, it is
left unchanged. The original count is restored on success and on exceptions.

```bash
VLLM_LOAD_TORCH_THREADS=4 vllm serve /path/to/model
```

This applies to the default loader only. It does not limit CPU threads during
inference and does not change weight values, attention or sampling. The thread
setting is process-wide: use this only during serialized initialization, not
concurrent loading/inference in the same process. Tune the cap for the hardware;
the example value of 4 is not a universal default.
Malformed non-integer values fail during initialization.
