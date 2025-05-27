# Reproducibility

vLLM does not guarantee the reproducibility of the results by default, for the sake of performance. You need to do the following to achieve
reproducible results:

- For V1: Turn off multiprocessing to make the scheduling deterministic by setting `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
- For V0: Set the global seed (see below).

!!! note

    Even with the above two settings, vLLM only provides reproducibility
    when it runs on the same hardware and the same vLLM version.
    Also, the online serving API (`vllm serve`) does not support reproducibility
    because it is almost impossible to make the scheduling deterministic in the
    online setting.

## Setting the global seed

The `seed` parameter in vLLM is used to control the random states for various random number generators. This parameter can affect the behavior of random operations in user code, especially when working with models in vLLM.

### Default Behavior

In V0, the `seed` parameter defaults to `None`. When the `seed` parameter is `None`, the random states for `random`, `np.random`, and `torch.manual_seed` are not set. This means that each run of vLLM will produce different results if `temperature > 0`, as expected.

In V1, the `seed` parameter defaults to `0` which sets the random state for each worker, so the results will remain consistent for each vLLM run even if `temperature > 0`.

!!! note

    Since V1 Engine is run in separate processes by default,
    the random state in user code (i.e. the code that constructs [LLM][vllm.LLM] class) remains unaffected.

    However, if you set `VLLM_ENABLE_V1_MULTIPROCESSING=0`,
    setting a seed does change the random state in user code.

    It is impossible to un-specify a seed for V1 because different workers need to sample the same outputs
    for workflows such as speculative decoding.
    
    For more information, see: <gh-pr:17929>

### Specifying a Seed

If a specific seed value is provided, the random states for `random`, `np.random`, and `torch.manual_seed` will be set accordingly. This can be useful for reproducibility, as it ensures that the random operations produce the same results across multiple runs.

!!! warning

    In V0, setting a seed changes the random state in user code which
    might affect subsequent operations outside vLLM.

### Example Usage

Without specifying a seed:

```python
import random
from vllm import LLM

# Initialize a vLLM model without specifying a seed
model = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

# Try generating random numbers
print(random.randint(0, 100))  # Outputs different numbers across runs
```

With a specific seed:

```python
import random
from vllm import LLM

# Initialize a vLLM model with a specific seed
model = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", seed=42)

# Try generating random numbers
print(random.randint(0, 100))  # Outputs the same number across runs
```

### Important Notes

- By default, the random state in the user code remains unaffected by vLLM.
- If a specific seed value is provided, the random states for `random`, `np.random`, and `torch.manual_seed` will be set to that value. This behavior can be useful for reproducibility but, in V0, may lead to non-intuitive behavior if the user is not explicitly aware of it.
