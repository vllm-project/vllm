# Reproducibility

vLLM does not guarantee the reproducibility of the results by default, for the sake of performance. To achieve
reproducible results:

- In offline mode, you can either set `VLLM_ENABLE_V1_MULTIPROCESSING=0` which makes scheduling deterministic,
  or enable [batch invariance](../features/batch_invariance.md) to make the outputs insensitive to scheduling.
- In online mode, you can only enable [batch invariance](../features/batch_invariance.md).

Example: [examples/rl/batch_invariance/reproducibility_offline.py](../../examples/rl/batch_invariance/reproducibility_offline.py)

!!! warning

    Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0` will change the random state of user code 
    (i.e. the code that constructs [LLM][vllm.LLM] class).

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

## Identical results across tensor-parallel ranks

Triton autotune benchmarks candidate kernel configs on each GPU, so tensor-parallel ranks can choose different configs for the same kernel and shape, and compute results that differ in the last bits. Plain tensor parallelism hides this: the all-reduce after each block gives every rank the same sum. Code where ranks compute the same tensors on their own, with no all-reduce after them (for example blocks replicated on every rank), lets the ranks drift apart.

Set `VLLM_TRITON_SYNC_AUTOTUNE_TP=1` to make every rank use the config chosen by tensor-parallel rank 0, so all ranks choose the same configs. The first time a kernel meets a new shape, rank 0 resolves it as usual, from Triton's on-disk autotune cache or by benchmarking, and broadcasts the result; the other ranks adopt it without benchmarking or reading their own cache, so a cache that is warm on some ranks only cannot make them disagree. Syncing happens only inside model forward passes and costs one CPU broadcast per kernel and shape. It is turned off, with a warning, under expert parallelism, sequence-parallel MoE and context parallelism, where ranks may autotune different shapes. It does not make results reproducible across runs; for that, see the sections above.
