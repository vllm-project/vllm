# Reproducibility

vLLM does not guarantee the reproducibility of the results by default, for the sake of performance. You need to do the following to achieve reproducible results:

- Turn off multiprocessing to make the scheduling deterministic by setting `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
- Optionally configure the global seed if you need to control random sampling (see below).

Example: <gh-file:examples/offline_inference/reproducibility.py>

!!! warning

    Applying the above settings [changes the random state in user code](#locality-of-random-state).

!!! note

    Even with the above settings, vLLM only provides reproducibility
    when it runs on the same hardware and the same vLLM version.
    Also, the online serving API (`vllm serve`) does not support reproducibility
    because it is almost impossible to make the scheduling deterministic in the
    online setting.

## Setting the global seed

The `seed` parameter in vLLM is used to control the random states for various random number generators.

If a specific seed value is provided, the random states for `random`, `np.random`, and `torch.manual_seed` will be set accordingly.

However, in some cases, setting the seed will also [change the random state in user code](#locality-of-random-state).

### Default Behavior

The `seed` parameter defaults to `0`, which sets the random state for each worker so the results remain consistent for each vLLM run even if `temperature > 0`.

!!! note

    It is impossible to un-specify a seed for V1 because different workers need to sample the same outputs
    for workflows such as speculative decoding.
    
    For more information, see: <gh-pr:17929>

### Locality of random state

The random state in user code (i.e. the code that constructs [LLM][vllm.LLM] class) is updated by vLLM when the workers run in the same process as user code, i.e.: `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

By default, this condition is not active so you can use vLLM without having to worry about accidentally making deterministic subsequent operations that rely on random state.
