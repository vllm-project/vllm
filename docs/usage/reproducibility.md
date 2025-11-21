# Reproducibility

vLLM does not guarantee the reproducibility of the results by default, for the sake of performance. To achieve
reproducible results, consider enabling [batch invariance](../features/batch_invariance.md) as the scheduling
cannot be made deterministic without using offline mode and setting `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

Example: [examples/offline_inference/reproducibility.py](../../examples/offline_inference/reproducibility.py)

!!! note

    Even with the above settings, vLLM only provides reproducibility
    when it runs on the same hardware and the same vLLM version.

## Setting the global seed

The `seed` parameter in vLLM is used to control the random states for various random number generators.

If a specific seed value is provided, the random states for `random`, `np.random`, and `torch.manual_seed` will be set accordingly.

### Default Behavior

In V1, the `seed` parameter defaults to `0` which sets the random state for each worker, so the results will remain consistent for each vLLM run even if `temperature > 0`.

It is impossible to un-specify a seed for V1 because different workers need to sample the same outputs
for workflows such as speculative decoding. For more information, see: <https://github.com/vllm-project/vllm/pull/17929>

!!! note

    The random state in user code (i.e. the code that constructs [LLM][vllm.LLM] class) is updated by vLLM 
    only if the workers are run in the same process as user code, i.e.: `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

    By default, `VLLM_ENABLE_V1_MULTIPROCESSING=1` so you can use vLLM without having to worry about
    accidentally making deterministic subsequent operations that rely on random state.
