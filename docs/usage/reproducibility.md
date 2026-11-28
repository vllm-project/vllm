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

## NVSwitch (NVLS) non-determinism on Hopper

On Hopper nodes with NVSwitch (e.g. H100, H20, H800), NVLS all-reduce results
can differ between runs when the kernel driver or Fabric Manager is older than
550.144.03 (or 570+). This causes run-to-run output differences even for a
single request on an otherwise idle server. Upgrading CUDA or NCCL alone does
**not** fix this — the fix is in the kernel driver / Fabric Manager (see
[NCCL#2360](https://github.com/NVIDIA/nccl/issues/2360)).

If you only need the same request to produce the same output (fixed-batch
reproducibility) and do not need full batch invariance, you can set
`NCCL_NVLS_ENABLE=0` instead of enabling the full batch-invariance mode. This
resolves NVLS-induced divergence with approximately 1% latency overhead,
compared to ~76% for the full set of batch-invariance NCCL overrides.

| Setting | Divergence | Latency cost |
|---|---|---|
| default | ~25% of reruns diverge | 1× |
| `NCCL_NVLS_ENABLE=0` | 0% | ~1.01× |
| Full batch invariance (`VLLM_BATCH_INVARIANT=1`) | 0% | ~1.76× |

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
