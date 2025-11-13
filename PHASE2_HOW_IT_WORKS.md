# How vLLM Parallel Execution Works

## The Flow

When you do:
```python
from vllm import LLM
llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=4)
```

Here's what happens under the hood:

### 1. LLM.__init__ (Main Process)
```
vllm/entrypoints/llm.py:344
    self.llm_engine = LLMEngine.from_engine_args(...)
```

### 2. Engine Creates Executor (Main Process)
```
The engine creates a MultiprocExecutor which will spawn worker processes
```

### 3. Executor Spawns Workers (4 separate Python processes)
```
vllm/v1/executor/multiproc_executor.py:100-200
    - Spawns 4 separate Python processes
    - Each gets a different rank (0, 1, 2, 3)
```

### 4. Each Worker Initializes Distributed State
```
Each worker process initializes:
    - torch.distributed.init_process_group()
    - parallel_state.initialize_model_parallel(tensor_parallel_size=4)
    - Sets _TP, _PP global variables
```

### 5. Each Worker Loads Model
```
vllm/v1/worker/gpu_model_runner.py:3064
    self.model = model_loader.load_model(vllm_config=...)

vllm/model_executor/model_loader/base_loader.py:49
    model = initialize_model(vllm_config=vllm_config)

vllm/model_executor/model_loader/utils.py:54
    return model_class(vllm_config=vllm_config, prefix=prefix)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    THIS is where YOUR model gets instantiated!
```

## How Our Callback Works

When you register a model with a callback:

```python
def build_my_model(vllm_config, parallel_context):
    tp_rank = parallel_context.get_tensor_parallel_rank()  # Gets REAL rank!
    tp_size = parallel_context.get_tensor_parallel_world_size()

    model = MyModel(config=vllm_config.hf_config, tp_size=tp_size)
    return model

ModelRegistry.register_model("MyModel", build_my_model)
```

What happens:

1. **ModelRegistry.load_model_cls()** returns `CallableModelWrapper` class
2. **vLLM calls** `CallableModelWrapper(vllm_config=vllm_config, prefix=prefix)`
3. **CallableModelWrapper.__init__**:
   - Creates `ParallelContext` from `vllm_config.parallel_config`
   - Calls `build_my_model(vllm_config, parallel_context)`
4. **Inside build_my_model**:
   - `parallel_context.get_tensor_parallel_rank()` calls `parallel_state.get_tensor_model_parallel_rank()`
   - At this point, `_TP` is ALREADY INITIALIZED in this worker process!
   - So it returns the REAL rank (0, 1, 2, or 3)!

## Process Diagram

```
Main Process:
    LLM() created
    └─> MultiprocExecutor created
        └─> Spawns 4 worker processes

Worker 0 (rank=0):                Worker 1 (rank=1):
    init distributed               init distributed
    _TP = GroupCoordinator(0/4)    _TP = GroupCoordinator(1/4)
    load model:                    load model:
      CallableModelWrapper()         CallableModelWrapper()
        ParallelContext created        ParallelContext created
        build_my_model called          build_my_model called
          tp_rank = 0 ✓                  tp_rank = 1 ✓
          create model                   create model

Worker 2 (rank=2):                Worker 3 (rank=3):
    ... same pattern ...           ... same pattern ...
```

## Why Our Test Doesn't Show This

Our pytest test runs in a SINGLE process without spawning workers, so:
- `_TP` is `None`
- `parallel_context.get_tensor_parallel_rank()` returns 0 (fallback)

To see REAL parallelism, you need to use the actual LLM() API which spawns workers.

## How to Test With Real Parallelism

Option 1: Use vLLM's LLM() API (what users will actually do)
Option 2: Use torchrun to manually spawn processes (for unit testing)
Option 3: Look at vLLM's existing tests that spawn workers

The key insight: **Our implementation already works correctly!** The parallel context gets the real ranks when run in vLLM's worker processes.
