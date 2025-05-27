# Reproducibility

vLLM does not guarantee the reproducibility of the results by default, for the sake of performance. You need to do the following to achieve
reproducible results:

- For V1: Turn off multiprocessing to make the scheduling deterministic by setting `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
- For V0: Set the global seed (see below).

!!! note

    Even with the above settings, vLLM only provides reproducibility
    when it runs on the same hardware and the same vLLM version.
    Also, the online serving API (`vllm serve`) does not support reproducibility
    because it is almost impossible to make the scheduling deterministic in the
    online setting.

!!! note

    Applying the above settings changes the random state in user code
    (i.e. the code that constructs [LLM][vllm.LLM] class).
    This may affect subsequent operations outside vLLM; see
    [this example](../examples/offline_inference/reproducibility.md).

## Setting the global seed

The `seed` parameter in vLLM is used to control the random states for various random number generators. This parameter can affect the behavior of random operations in user code, especially when working with models in vLLM.

### Default Behavior

In V0, the `seed` parameter defaults to `None`. When the `seed` parameter is `None`, the random states for `random`, `np.random`, and `torch.manual_seed` are not set. This means that each run of vLLM will produce different results if `temperature > 0`, as expected.

In V1, the `seed` parameter defaults to `0` which sets the random state for each worker, so the results will remain consistent for each vLLM run even if `temperature > 0`.

!!! note

    It is impossible to un-specify a seed for V1 because different workers need to sample the same outputs
    for workflows such as speculative decoding.
    
    For more information, see: <gh-pr:17929>

### Specifying a Seed

If a specific seed value is provided, the random states for `random`, `np.random`, and `torch.manual_seed` will be set accordingly. This can be useful for reproducibility, as it ensures that the random operations produce the same results across multiple runs.

### Locality of random state

By default, the random state in code outside of vLLM remains unaffected by vLLM.

- In V0: The seed is not specified by default.
- In V1: The workers are run in separate processes, unless `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
