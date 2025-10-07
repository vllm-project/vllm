# CUDA Graphs

This write-up introduces the new CUDA Graphs modes in vLLM v1 beyond previous [torch.compile integration](torch_compile.md). To summarize, we:

1. Added flexible `cudagraph_mode` configuration
2. Made full CUDA Graphs support orthogonal to compilation
3. Introduced a CUDA Graphs dispatcher as a central controller that picks the desired runtime mode and CUDA Graphs per batch automatically

In this document we will discuss the:

* [Motivation](#motivation)
* [CUDA Graphs modes](#cudagraphmodes)
* [Detailed design](#detailed-design)
* [Example usage of the different CUDA Graphs modes](#usage-guide)

!!! note
    In this document, we refer to pure decode (`max_query_len=1`) or speculative decode (`max_query_len =1+num_spec_tokens`) as **uniform decode** batches, and the opposite would be **non-uniform** batches (i.e., prefill or mixed prefill-decode batches).

!!! note
    The following contents are mostly based on the last commit of <gh-pr:20059>.

## Motivation

In the past, using the [torch.compile integration](torch_compile.md), we achieved a balance between performance and attention operation compatibility using piecewise compilation (+piecewise CUDA Graphs). However, when users enabled full CUDA Graphs, which relies on no-splitting compilation, the experience was all-or-nothing. CUDA Graphs was tightly coupled to compilation so lost the flexibility to choose supported attention backends (i.e., cascade attention is incompatible with CUDA Graphs). Many attention backends also weren’t ready for unified "full" CUDA Graphs capture (e.g., only FlashAttention 3 supports it currently) or only support CUDA Graphs for pure decode batches (e.g., Flashinfer, FlashMLA, and Mamba, etc.). That may lead to confusing performance/compatibility tradeoffs, inconsistent CUDA Graphs supports, and increasingly complex code structures.

So we seek a more fine-grained CUDA Graphs solution that can:

* Be explicitly aware of whether a batch is prefill/uniform decode/mixed and should capture/replay the CUDA Graphs accordingly. It should do this because a unified full CUDA Graph for different cases of the same batchsize is usually infeasible (e.g., for many attention backends).
* Capture full CUDA Graphs while maintaining the ability to use piecewise CUDA Graphs. i.e, can dispatch CUDA Graph-incompatible routines (e.g., cascade attention or mixed prefill/decode batches for some attention backends) into piecewise CUDA Graphs.
* Achieve centralized control of the CUDA Graphs behavior via a dispatcher, which makes CUDA Graphs dispatching easier to understand and more extensible.
* Add CUDA Graphs support to models that do not fit vllm's torch.compile integration system design in v1.

We also found that when a batch cannot hit a full CUDA Graph, the host-side eager execution of the flattened compiled FX graph (previous behavior) can be slower than the piecewise compiled FX graph in Python (see <gh-pr:20059>). So, we prefer maintaining the piecewise compilation when enabling full CUDA Graphs to reduce host-side overhead. We can safely do this as full CUDA Graphs and compilation are actually orthogonal to each other.

## `CudagraphModes`

[CUDAGraphMode][vllm.config.compilation.CUDAGraphMode] is the single knob you tune in `CompilationConfig.cudagraph_mode`:

* `NONE` — turn CUDA Graphs off. Good for debugging.
* `PIECEWISE` —  a single-mode strategy (and past default). It is the most flexible: attention or other CUDA Graphs-incompatible operations stay eager, everything else goes into CUDA Graphs.
* `FULL` — a single-mode strategy, which only captures full CUDA Graphs for non-uniform batches, then uniform-decode batches reuse the CUDA Graph of non-uniform batch of the same batch_size, since they are compatible; can be good for small models or workloads with small prompts.
* `FULL_DECODE_ONLY` — full CUDA Graph for uniform decode, eager run for prefill/mixed etc; suitable for decode instances in a P/D setup where prefill is not as important, so we can save some memory.
* `FULL_AND_PIECEWISE` — (default mode) full CUDA Graph for uniform decode, piecewise CUDA Graphs for others; generally the most performant setting, especially for low latency with small models or MoEs.

Defaults: If you’re on v1 with piecewise compilation, we default to `FULL_AND_PIECEWISE` for better performance, (for pooling models, it's still `PIECEWISE`). Otherwise, e.g. if piecewise compilation unavailable, we default to `NONE`.

While `NONE` , `PIECEWISE`, and `FULL` are single-mode configurations and simply equivalent to past implementations of eager execution, piecewise CUDA Graphs, and full CUDA Graphs respectively, `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE` are newly appended dual-mode configurations, which require dispatching to switch between concrete runtime modes according to runtime batches dynamically.

!!! note
    We also fuse the subset `NONE`, `PIECEWISE`, and `FULL` modes as the concrete runtime modes for CUDA Graphs dispatching, which means they are treated as one of the modes for prefill/mixed or uniform-decode phase at runtime.

With the new feature, cascade attention is supported by all Cudagraph modes now (if the attention backend supports it), though cascade attention itself is not cudagraph compatible.  Batches that use cascade attention will be dispatched to `PIECEWISE` runtime mode if we have a corresponding piecewise cudagraph at runtime; otherwise, they will be dispatched to `NONE` runtime mode.

!!! note
    Not all CUDA Graphs modes are compatible with every attention backend. For convenience, we alias `FULL` mode to `FULL_AND_PIECEWISE` (`-O 3`) or `FULL_DECODE_ONLY` (`-O 0`) for attention backends that support CUDA Graphs for only pure decode or uniform decode.

## Detailed Design

### Overview

The new CUDA Graphs logic is built on top of piecewise compilation and supports dual CUDA Graphs runtime mode switching. The system contains the following core components:
* [CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper]: wrapper that handles CUDAGraph capture & replay on the wrapped callable
* [CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher]: the central controller that contains the single source of truth about CUDA Graphs and handles dispatching between them.
* [CUDAGraphMode][vllm.config.compilation.CUDAGraphMode]: enum describing the supported and runtime modes (introduced above).
* [BatchDescriptor][vllm.forward_context.BatchDescriptor], serving as a unique representation of the runtime batch used for dispatching.

See the following figures for a quick comparison between the previous and current design patterns of CUDA Graphs with inductor compilation. We can see that previously the CUDA Graphs logic and compilation logic were tightly coupled into the vllm `PiecewiseBackend`, and CUDA Graphs was implicitly dispatched by `batch_size` idly. Now the CUDA Graphs logic is separated into the `CUDAGraphWrapper` class, responsible for both full and piecewise CUDA Graphs abilities, and dispatching is **explicitly** done via **runtime mode** plus the `BatchDescriptor` as the **dispatch key** via `CudagraphDispatcher`.

**Before:**

![previous_design](../assets/design/cuda_graphs/previous_design.png)

**After:**

![new_design](../assets/design/cuda_graphs/current_design.png)

### `BatchDescriptor`

[BatchDescriptor][vllm.forward_context.BatchDescriptor] is a component within `ForwardContext`, alongside the CUDA Graphs runtime modes, serving as the core structure for dispatching keys at runtime. The prototype is:

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    uniform_decode: bool = False
```

where `num_tokens` can be the padded token length, and `uniform_decode` is determined by if `max_query_len` of a batch is equal to the desired `max_query_len` of a uniform_decode, and the num_scheduled_tokens is divisible by that desired `max_query_len`.

The goal of this structure is to uniquely identify a (padded) batch with minimal possible items corresponding to a CUDA Graphs item. We are safe to exclude items like `uniform_query_len` because it is a constant at runtime for a certain setup currently. For example, it should be either `1` for a commonly pure decode or `1+num_spec_tokens` for a validation phase of speculative decode.

!!! note
    The prototype of `BatchDescriptor` may be extended for more general situations in the future, e.g., include more items, like `uniform_query_len` to support multiple different uniform decode lengths settings (<gh-pr:23679>), or other modifications needed to support CUDA Graphs for models whose inputs are not necessarily token length aware (for example, some multi-modal inputs).

### `CudagraphDispatcher`

The [CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher] takes responsibility for maintaining two sets of valid dispatching keys, one set for `FULL` runtime mode and one set for `PIECEWISE` runtime mode, and dispatches the correct runtime mode and the dispatching keys before executing the model's forwards. It will take in the initial key (a rough batch_descriptor for the padded input) and return the selected runtime mode and the final batch_descriptor, then tell the CUDAGraphWarpper instances that decision through forward contexts.  We should notice that CudagraphDispatcher is the only source of truth for available CUDA Graphs keys, and the CUDAGraphWrapper instances could have less logic and unquestioningly trust the forward context on what CUDA Graphs to dispatch to.

The dispatching keys are initialized through the dispatcher's `initialize_cudagraph_keys` method, which is called by the gpu_model_runner after all possible attention backends are initialized. This is where we can get much fancier in the future and “prepare” all kinds of CUDA Graphs combinations. For now, we just append available keys based on the valid combos of `decode_mode`/`mixed_mode` of `cudagraph_mode` and `cudagraph_capture_sizes` in the compilation config.

The dispatch code looks like:

```python
batch_descriptor=BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=...)
runtime_mode, batch_descriptor = cudagraphdispatcher.dispatch(batch_descriptor)
# execution
with set_forward_context(..., 
            cudagraph_runtime_mode=runtime_mode, 
            batch_descriptor=batch_descriptor):
     output = self.model(...)
```

Inside the `dispatch()` method, the dispatcher will search the proper CUDA Graphs runtime mode and existing dispatching keys for a return. We basically search the existing keys following the priority: `FULL`>`PIECEWISE`>`None`. If the dispatching key does not exist, default to return `NONE` mode for eager execution. The implementations can be found [here](https://github.com/vllm-project/vllm/blob/main/vllm/v1/cudagraph_dispatcher.py#L91).

Here is a simplified illustration of the workflow at runtime in the model executor:
![executor_runtime](../assets/design/cuda_graphs/executor_runtime.png)

### `CUDAGraphWrapper`

A [CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper] instance wraps a runnable and simply mimics the runnable with appended CUDA Graphs abilities. Each wrapper instance is bound to a specific `runtime_mode`, which is restricted to `PIECEWISE` and `FULL` mode, and takes responsibility for capturing/replaying and passing through (directly calling) the runnable.  At runtime, each wrapper would:

1. inspect the runtime_mode and batch_descriptor(dispatching key) from the global forward context.
2. If runtime_mode is `NONE` or runtime_mode does not match the mode of the wrapper, just call the runnable directly.
3. Otherwise, i.e., the runtime_mode matches the mode of the wrapper, the wrapper will perform CUDA Graphs capture (if key does not exist, create
a new entry and cache it) or replay (if key exists in the cache).

The above steps are based on the assumption that the CUDA Graphs wrapper would directly trust what’s in the forward context (controlled by the dispatcher) without any fallback behavior. See the implementation [here](https://github.com/vllm-project/vllm/blob/f751e50b7a2aae3110d83ed0d88202fc91b3e78a/vllm/compilation/cuda_graph.py#L106).

#### Nested Wrapper design

The core mechanism of making a full CUDA Graphs and piecewise CUDA Graphs coexist and compatible is the nested CUDA Graphs wrapper design, building on top of piecewise compilation with only a single piecewise FX graph.  We wrap a FULL mode wrapper outside the entire model for the full CUDA Graphs functionality; meanwhile, each piecewise backend is wrapped via a `PIECEWISE` mode wrapper inside the compilation.

The flow chart below should clearly describe how it works.
![wrapper_flow](../assets/design/cuda_graphs/wrapper_flow.png)

Therefore, for a `FULL` runtime mode, it is safe to capture/replay a full CUDA Graph since the piecewise wrapper is not activated. The situation is similar for `PIECEWISE` mode, as there are no conflicts between the `FULL` mode wrapper and `PIECEWISE` mode wrappers.  For the `NONE` runtime mode, both `FULL` and `PIECEWISE` wrappers would not be activated, so an eager execution is passed.

### Full CUDA Graph capturing & warm-up

The CUDA Graphs capturing happens on the first call runner's `dummy_run` in a non-`NONE` runtime mode. For full CUDA Graphs capture (pass `FULL` runtime mode), the core idea of explicitly capturing different cases (i.e., prefill/mixed batch or uniform_decode batch ) is to tell the underlying attention backend to launch the desired kernel routines (i.e., may launch different kernels or combos for different cases) via carefully crafting the attn_metadatas. To distinguish prefill/mixed batch or uniform_decode batch, the most important property is the `max_query_len` in attn_metadata (true for most attention backends). we set it to the desired uniform_query_len for uniform_decode otherwise we make it just the `num_tokens` for a non-uniform_decode batch.

The CUDA Graphs wrapper no longer manages the warm-up logic. The warm-up process is now controlled directly by the GPU model runner, where the `NONE` runtime mode is assigned to play an eager execution for warm-up. When warming up for a full CUDA Graph, it is also important to pass `force_attention=True` to the `dummy_run` function to explicitly warm up the attention backends.

## CUDA Graphs Compatibility of Attention Backends

To signal the CUDA Graphs compatibility of the attention backends, we introduce a new enum type [AttentionCGSupport][vllm.v1.attention.backends.utils.AttentionCGSupport], which is an enum type that tracks the capability of the attention backend to support CUDA Graphs. The value is sorted in the order of the capability, i.e., `ALWAYS`> `UNIFORM_BATCH`> `UNIFORM_SINGLE_TOKEN_DECODE`> `NEVER`.

```python
class AttentionCGSupport(enum.Enum):
    """ Constants for the CUDA Graphs support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never CUDA Graphs supported."""

    ALWAYS = 3
    """CUDA Graphs always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """CUDA Graphs supported for batches the only contain query lengths that are
    the same, this can be used for spec-decode 
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """CUDA Graphs supported for batches the only contain query_len==1 decodes"""
    NEVER = 0
    """NO CUDA Graphs support"""
```

Suppose we have hybrid attention backends (e.g., in mamba mixer models). In that case, we seek the minimum capability of all backends to determine the final capability of the model, and we might resolve the incompatible CUDA Graphs mode by downgrading the mode to the best fit one. For example, downgrading `FULL` mode to `FULL_AND_PIECEWISE` mode if the minimum capability is `UNIFORM_BATCH`, or `PIECEWISE` mode if the minimum capability is `NEVER` for -O3 compilation level. For the complete fallback policy, please see the code of [initialize_cudagraph_capture][vllm.v1.worker.gpu_model_runner.GPUModelRunner.initialize_cudagraph_capture].

The following table lists backends that support full CUDA Graphs at the time of writing.

| Attention Backend | cudagraph_support | Comments |
|:---|:---|:---|
| FlashAttention v2 | `UNIFORM_BATCH` | Actually `ALWAYS` but workaround to fallback to `FULL_AND_PIECEWISE` for performance reason |
| FlashAttention v3 | `ALWAYS` | has unified routine for both batches, so `FULL` mode is good |
| Triton Attention | `ALWAYS` | prefer `FULL_AND_PIECEWISE` since it has different kernels for prefill/mixed and pure decode batches |
| AITER FlashAttention | `UNIFORM_BATCH`| |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| FlashMLA | `UNIFORM_BATCH` | |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Mamba attention| `UNIFORM_SINGLE_TOKEN_DECODE` | |

Unlisted backends are all declared as `NEVER`.

## Usage guide

Now the CLI is directly using the uppercase string of cudagraph_mode for compilation_config: `--compilation-config '{"cudagraph_mode": "..."}'`, where `...` should be one of `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, and `FULL_AND_PIECEWISE`. Note that all `PIECEWISE` related modes require piecewise compilation, and all `FULL` related modes need CUDA Graphs support of attention backends. For example:

```bash
vllm serve --model meta-llama/Llama-3.1-8B-Instruct --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

### Python examples

```python
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

import vllm
from vllm.config import CUDAGraphMode

compilation_config = {"level": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}
model = vllm.LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            dtype='auto',
            compilation_config = compilation_config,
        )
sampling_params = vllm.SamplingParams(
    temperature=0,  # greedy decoding
    max_tokens=1024,
)
outputs = model.generate(
    ["My name is John and"],
    sampling_params=sampling_params,
)
```

### Migration from legacy flags

Legacy `use_cudagraph` and `full_cuda_graph` are unified by `cudagraph_mode`:

* `use_cudagraph=False` → `NONE`.
* `use_cudagraph=True` and `full_cuda_graph=False` → `PIECEWISE`.
* `full_cuda_graph=True` → directly set `FULL` and account for the graceful fallback policy.

As they are deprecated and will be removed in the next major or minor release, i.e., v0.11.0 or v1.0.0, we recommend using cudagraph_mode instead.

### NOTE for attention ops fusion

Attention ops fusion is compatible with piecewise cudagraph only when using inductor graph partition, i.e., passing `--compilation_config '{"use_inductor_graph_partition":true}'` (Note: this is an experimental and will be available after Pytorch version>=2.9). Otherwise, piecewise cudagraph does not support attention fusion by vllm custom graph partition. In the later case (which is also current state), we should set `splitting_ops=[]` in compilation_config to retain an complete FX graph for custom pass, and use cudagraph_mode = "FULL" or "FULL_DECODE_ONLY" when enabling attention fusion, since the default behavior of cudagraph_mode != `NONE` is always keeping the attention ops in the splitting_ops to get a piecewise FX graph. The good news is that the above tuning is automatically settled when `pass_config.enable_attn_fusion==True` and no users' explicit configs.

## About the Performance

See the following links for examples:

* [20059#issuecomment-3160858458](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3160858458)
* [20059#issuecomment-3188735226](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3188735226)
* [20059#issuecomment-3219888738](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3219888738)
