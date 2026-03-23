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
* [Encoder (ViT) CUDA Graphs](#encoder-vit-cuda-graphs)

!!! note
    In this document, we refer to pure decode (`max_query_len=1`) or speculative decode (`max_query_len =1+num_spec_tokens`) as **uniform decode** batches, and the opposite would be **non-uniform** batches (i.e., prefill or mixed prefill-decode batches).

!!! note
    The following contents are mostly based on the last commit of <https://github.com/vllm-project/vllm/pull/20059>.

## Motivation

Initial piecewise compilation was built to allow piecewise cudagraph capture, excluding cudagraph-unsupported operations (mainly attention). This allowed some speedup from cudagraphs while maintaining compatibility with all attention backends. We later added support for "full cudagraphs" by not compiling piecewise, so that we could further reduce the latency in cases where attention supported cudagraphs. However, this tight coupling between compilation and cudagraph capture led to an all-or-nothing experience with little flexibility. Many attention backends also weren’t ready for unified "full" CUDA Graphs capture (e.g., only FlashAttention 3 supports it currently) or only support CUDA Graphs for pure decode batches (e.g., Flashinfer, FlashMLA, and Mamba, etc.). That led to confusing performance/compatibility tradeoffs, inconsistent CUDA Graphs support, and increasingly complex code structure.

This led us to seek a more fine-grained CUDA Graphs solution with the following features:

* Explicitly aware of CUDA Graphs for prefill/mixed or (uniform-)decode batch and capture them separately.
* Separate CUDAGraph capture logic from compilation (as much as feasible) for feature orthogonality, which suggest:
    * Capturing piecewise and full cudagraphs using the same compiled graph, and
    * Full cudagraph capture without compilation.
* Dispatch between full and piecewise cudagraph at runtime depending on batch composition.
* Centralized control of CUDAGraph behavior for reduced code complexity and allowed more extendibility.

These features allow the most flexibility for cudagraph capture and compilation for all kinds of startup/performance tradeoffs and feature support.

## `CudagraphModes`

[CUDAGraphMode][vllm.config.compilation.CUDAGraphMode] is the single knob you tune in `CompilationConfig.cudagraph_mode`:

* `NONE` — turn CUDA Graphs off. Good for debugging.
* `PIECEWISE` —  a single-mode strategy (and past default). It is the most flexible: attention or other CUDA Graphs-incompatible operations stay eager, everything else goes into CUDA Graphs. Requires piecewise compilation.
* `FULL` — a single-mode strategy, which only captures full CUDA Graphs for non-uniform batches, then uniform-decode batches reuse the CUDA Graph of non-uniform batch of the same batch_size, since they are compatible; can be good for small models or workloads with small prompts.
* `FULL_DECODE_ONLY` — full CUDA Graph for uniform decode, no cudagraph for prefill/mixed etc.; suitable for decode instances in a P/D setup where prefill is not as important, this way we can save the memory needed for `PIECEWISE` CUDA Graphs.
* `FULL_AND_PIECEWISE` — (default mode) full CUDA Graph for uniform decode, piecewise CUDA Graphs for others; generally the most performant setting, especially for low latency with small models or MoEs, but also requires the most memory and takes the longest to capture.

Defaults: If you’re on v1 with piecewise compilation, we default to `FULL_AND_PIECEWISE` for better performance, (for pooling models, it's still `PIECEWISE`). Otherwise, e.g. if piecewise compilation unavailable, we default to `NONE`.

While `NONE` , `PIECEWISE`, and `FULL` are single-mode configurations and simply equivalent to past implementations of eager execution, piecewise CUDA Graphs, and full CUDA Graphs respectively, `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE` are newly appended dual-mode configurations, which require dispatching to switch between concrete runtime modes according to runtime batches dynamically.

!!! note
    Here, the single-modes `NONE`, `PIECEWISE`, and `FULL` are treated as the runtime modes for CUDA Graphs dispatching. If using a dual-mode, the dispatcher will always dispatch to one of its member modes (plus a potential `NONE` if no suitable CUDA Graph available), depending on the batch composition.

While cascade attention is not cudagraph compatible, it is now compatible with all possible cudagraph mode configurations. If a batch uses cascade attention, it always gets dispatched to `PIECEWISE` mode if available (otherwise `NONE`).

!!! note
    Not all CUDA Graph modes are compatible with every attention backend. We automatically "downgrade" modes to the closest supported mode. For example, if a backend only supports CUDA Graphs for pure decode/uniform batches, we convert `FULL` to `FULL_AND_PIECEWISE` if piecewise compilation is enabled, and `FULL_DECODE_ONLY` otherwise.

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
    num_reqs: int
    uniform: bool = False
    has_lora: bool = False
```

where `num_tokens` can be the padded token length, and `uniform` indicates if all the requests have the same query lengths. Many attention backends only support full cudagraphs when the batches are uniform; pure decode batches are uniform but may not be query length 1 (i.e. `num_tokens == num_reqs`), this occurs in the validation pass of spec-decode where "decode" batches will have a query length of  `1+num_spec_tokens`.

The goal of this structure is to uniquely identify a (padded) batch with minimal possible items corresponding to a CUDA Graphs item.

!!! note
    The prototype of `BatchDescriptor` may be extended for more general situations in the future, e.g., include more items, like `uniform_query_len` to support multiple different uniform decode lengths settings (<https://github.com/vllm-project/vllm/pull/23679>), or other modifications needed to support CUDA Graphs for models whose inputs are not necessarily token length aware (for example, some multi-modal inputs).

### `CudagraphDispatcher`

The [CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher] takes responsibility for maintaining two sets of valid dispatching keys, one set for `FULL` runtime mode and one set for `PIECEWISE` runtime mode, and dispatches the correct runtime mode and the dispatching keys before executing the model's forwards. It will take in the initial key (a rough batch_descriptor for the padded input) and return the selected runtime mode and the final batch_descriptor, then tell the CUDAGraphWrapper instances that decision through forward contexts. Notice that `CudagraphDispatcher` is the only source of truth for available CUDA Graph keys and `CUDAGraphWrapper` instances can blindly trust the forward context on what CUDA Graphs to dispatch to. This lets us simplify the wrapper code and centralize the logic in the dispatcher.

The dispatching keys are initialized through the dispatcher's `initialize_cudagraph_keys` method, which is called by the gpu_model_runner after all possible attention backends are initialized. This is where we can get much fancier in the future and “prepare” all kinds of CUDA Graphs combinations. For now, we just append available keys based on the valid combos of `decode_mode`/`mixed_mode` of `cudagraph_mode` and `cudagraph_capture_sizes` in the compilation config.

The dispatch code looks like:

```python
batch_descriptor=BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=...)
runtime_mode, batch_descriptor = cudagraphdispatcher.dispatch(batch_descriptor)
# execution
with set_forward_context(
    ..., 
    cudagraph_runtime_mode=runtime_mode, 
    batch_descriptor=batch_descriptor,
):
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

The above steps are based on the assumption that the CUDA Graphs wrapper would directly trust what’s in the forward context (controlled by the dispatcher). This lets us simplify and centralize the logic, reducing the complexity as well as the risk of mismatched state between the wrappers and the dispatcher. It also allows reusing the wrapper class for both `FULL` and `PIECEWISE` runtime modes. See the implementation [here](https://github.com/vllm-project/vllm/blob/f751e50b7a2aae3110d83ed0d88202fc91b3e78a/vllm/compilation/cuda_graph.py#L106).

#### Nested Wrapper design

The core mechanism of making a full CUDA Graphs and piecewise CUDA Graphs coexist and compatible is the nested CUDA Graphs wrapper design, building on top of piecewise compilation with only a single piecewise FX graph.  We wrap a FULL mode wrapper outside the entire model for the full CUDA Graphs functionality; meanwhile, each piecewise backend is wrapped via a `PIECEWISE` mode wrapper inside the compilation.

The flow chart below should clearly describe how it works.
![wrapper_flow](../assets/design/cuda_graphs/wrapper_flow.png)

Therefore, for a `FULL` runtime mode, it is safe to capture/replay a full CUDA Graph since the piecewise wrapper is not activated. The situation is similar for `PIECEWISE` mode, as there are no conflicts between the `FULL` mode wrapper and `PIECEWISE` mode wrappers.  For the `NONE` runtime mode, both `FULL` and `PIECEWISE` wrappers would not be activated, so we simply fall through to eager execution.

### Full CUDA Graph capturing & warm-up

The CUDA Graphs capturing happens when the runner first calls the model forward (using `_dummy_run`) with a non-`NONE` runtime mode. For full CUDA Graph capture, we explicitly capture different cases (i.e., prefill/mixed batch or uniform_decode batch) by properly setting attention metadata to make sure the underlying attention backends launch the desired kernel routines. To distinguish prefill/mixed batch or uniform_decode batch, the most important property is the `max_query_len` in attn_metadata (true for most attention backends). We set it to the desired `uniform_query_len` for uniform_decode otherwise we make it just the `num_tokens` for a non-uniform_decode batch.

The CUDA Graphs wrapper no longer manages the warm-up logic. The warm-up process is now controlled directly by the GPU model runner, where the `NONE` runtime mode is assigned to play an eager execution for warm-up. When warming up for a full CUDA Graph, it is also important to explicitly run attention during the warmup `dummy_run` call.

## CUDA Graphs Compatibility of Attention Backends

To signal the CUDA Graphs compatibility of the attention backends, we introduce a new enum type [AttentionCGSupport][vllm.v1.attention.backend.AttentionCGSupport], which is an enum type that tracks the capability of the attention backend to support CUDA Graphs. The value is sorted in the order of the capability, i.e., `ALWAYS`> `UNIFORM_BATCH`> `UNIFORM_SINGLE_TOKEN_DECODE`> `NEVER`.

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

Suppose we have hybrid attention backends (e.g., in mamba mixer models). In that case, we seek the minimum capability of all backends to determine the final capability of the model, and we might resolve the incompatible CUDA Graphs mode by downgrading the mode to the best fit one. For example, downgrading `FULL` mode to `FULL_AND_PIECEWISE` mode if the minimum capability is `UNIFORM_BATCH`, or `PIECEWISE` mode if the minimum capability is `NEVER` for -O3 compilation mode. For the complete fallback policy, please see the code for [this][vllm.v1.worker.gpu_model_runner.GPUModelRunner._check_and_update_cudagraph_mode].

The following table lists backends that support full CUDA Graphs at the time of writing.

| Attention Backend | cudagraph_support | Comments |
| :---------------- | :---------------- | :------- |
| FlashAttention v2 | `UNIFORM_BATCH` | Actually `ALWAYS` but workaround to fallback to `FULL_AND_PIECEWISE` for performance reason |
| FlashAttention v3 | `ALWAYS` | has unified routine for both batches, so `FULL` mode is good |
| Triton Attention | `ALWAYS` | prefer `FULL_AND_PIECEWISE` since it has different kernels for prefill/mixed and pure decode batches |
| AITER FlashAttention | `UNIFORM_BATCH` | |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | Will be set to `UNIFORM_BATCH` when using TRTLLM attention on Blackwell |
| FlashMLA | `UNIFORM_BATCH` | |
| FlashInferMLA | `UNIFORM_BATCH` | |
| FlashInferMLASparse | `UNIFORM_BATCH` | |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Mamba attention | `UNIFORM_SINGLE_TOKEN_DECODE` | |

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

compilation_config = {"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}
model = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    compilation_config=compilation_config,
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

### Piecewise compilation and full graph custom passes (attention fusion, sequence parallelism)

Unfortunately, some custom compile passes have to see the whole graph to be effective and hence aren't compatible with piecewise compilation. This includes `AttnFusionPass` and `SequenceParallelismPass`. As a short-term solution, we automatically disable piecewise compilation (by setting `splitting_ops=[]`) when attention fusion is enabled. We use CUDA Graph modes `FULL` or `FULL_DECODE_ONLY` (depending on backend support). However, this leads to another optimization incompatibility and confusing performance tradeoffs.

Long term, we've added the ability to partition the graph in Inductor instead of right after Dynamo. It can be enabled with `CompilationConfig.use_inductor_graph_partition=True` but is currently experimental and only available with `torch>=2.9`. This also increases compilation time as it has to compile the whole graph and cannot reuse piecewise compilation artifacts. Once vLLM supports 2.9, we plan to make this the default approach as it will also speed up piecewise cudagraph capture.

## About the Performance

See the following links for examples:

* [20059#issuecomment-3160858458](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3160858458)
* [20059#issuecomment-3188735226](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3188735226)
* [20059#issuecomment-3219888738](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3219888738)

## Encoder (ViT) CUDA Graphs

The CUDA Graphs infrastructure described above targets the **decoder** (language model) forward pass. vLLM also supports capturing the **encoder** (vision transformer) forward pass as CUDA Graphs, independently from the decoder. This is based on <https://github.com/vllm-project/vllm/pull/35963>.

!!! note
    Encoder CUDA Graphs are orthogonal to decoder CUDA Graphs — both can be enabled simultaneously. Encoder graphs capture the vision encoder execution (e.g., ViT in Qwen3-VL), while decoder graphs capture the language model execution as described in the sections above.

### Motivation

Vision encoder inference incurs CUDA kernel launch overhead on the host side. The overhead is more significant when the batch size is small or image size is small.

Encoder CUDA Graphs eliminate this overhead by pre-capturing the full encoder forward pass at multiple token budget levels during model initialization, then replaying the appropriate graph at runtime.

### Design

The encoder CUDA Graph system uses a **budget-based capture/replay** strategy, managed by [EncoderCudaGraphManager][vllm.v1.worker.gpu.mm.encoder_cudagraph.EncoderCudaGraphManager]. The system contains the following core components:

* [EncoderCudaGraphManager][vllm.v1.worker.gpu.mm.encoder_cudagraph.EncoderCudaGraphManager]: orchestrates capture, replay, greedy packing, and data-parallel execution for encoder CUDA Graphs.
* [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph]: a runtime-checkable protocol that models implement to opt-in to encoder CUDA Graphs.
* [BudgetGraphMetadata][vllm.v1.worker.gpu.mm.encoder_cudagraph.BudgetGraphMetadata]: holds the captured CUDA Graph and its associated I/O buffers for a single token budget level.

#### Budget-based graph capture

Multiple CUDA Graphs are pre-captured at different **token budget** levels (e.g., `[2048, 4096, 8192, 13824]`). Each budget defines a fixed token capacity, and all budgets share the same maximum batch size (number of images). The `BudgetGraphMetadata` for each level stores the graph along with pre-allocated input, metadata, and output buffers:

```python
@dataclass
class BudgetGraphMetadata:
    token_budget: int
    max_batch_size: int
    graph: torch.cuda.CUDAGraph
    input_buffer: torch.Tensor       # e.g. pixel_values
    metadata_buffers: dict[str, torch.Tensor]  # e.g. embeddings, seq metadata
    output_buffer: torch.Tensor      # encoder hidden states
```

Budgets are auto-generated as power-of-2 levels from a model-provided range via `get_encoder_cudagraph_budget_range()`, with the maximum budget always included even if it does not fall on a power-of-2 boundary. Budgets can also be explicitly specified by the user via `encoder_cudagraph_token_budgets` in `CompilationConfig`.

#### Greedy bin-packing at runtime

When a batch of images arrives, the manager sorts images by output token count (smallest first) and greedily packs as many images as possible into each sub-batch while staying within the **largest** token budget and the maximum batch size. Once a sub-batch is finalized (the next image would overflow either constraint), the manager finds the **smallest** budget that fits the sub-batch's total tokens and replays the corresponding CUDA Graph. This repeats until the batch is exhausted. Images that exceed all budgets fall back to eager execution.

For each graph replay:

1. Zero the pre-allocated `input_buffer`, then copy input tensors (e.g., `pixel_values`) into it.
2. Zero `metadata_buffers`, then slice-copy precomputed values (e.g., rotary embeddings, sequence metadata).
3. Replay the CUDA Graph.
4. Clone outputs from `output_buffer` (cloning is necessary since the buffer is reused across replays).

#### Data-parallel support

When `mm_encoder_tp_mode="data"`, the manager distributes images across TP ranks using load-balanced assignment via `get_load_balance_assignment`, executes locally on each rank, then gathers results back in the original order via `tensor_model_parallel_all_gather`.

### Model integration via `SupportsEncoderCudaGraph`

Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph] protocol. This protocol encapsulates all model-specific logic so that the manager remains model-agnostic. The protocol defines the following methods:

* `get_encoder_cudagraph_config()` — returns static configuration (supported modalities, input key, buffer keys, output hidden size).
* `get_encoder_cudagraph_budget_range(vllm_config)` — returns `(min_budget, max_budget)` for auto-inference of token budgets.
* `get_encoder_cudagraph_num_items(mm_kwargs)` — returns the number of items (e.g. images) in the batch.
* `get_encoder_cudagraph_per_item_output_tokens(mm_kwargs)` — returns per-item output token counts, used for greedy packing.
* `get_encoder_cudagraph_per_item_input_sizes(mm_kwargs)` — returns per-item input sizes (e.g. patch counts), used for DP load balancing.
* `select_encoder_cudagraph_items(mm_kwargs, indices)` — extracts a sub-batch of items by index, used during greedy packing and DP sharding.
* `prepare_encoder_cudagraph_capture_inputs(...)` — creates dummy inputs for graph capture.
* `prepare_encoder_cudagraph_replay_buffers(...)` — computes new buffer values from actual batch inputs before replay.
* `encoder_cudagraph_forward(...)` — forward pass using precomputed buffers (called during capture and replay).
* `encoder_eager_forward(...)` — fallback eager forward when no graph fits.

Currently supported: **Qwen3-VL** (see `vllm/model_executor/models/qwen3_vl.py`).

!!! note
    The `SupportsEncoderCudaGraph` protocol is designed to be model-agnostic. New vision encoder models can opt-in by implementing the protocol methods without modifying the manager.

!!! note
    Encoder CUDA Graphs have currently been tested with `--mm-encoder-attn-backend=FLASH_ATTN` and `--mm-encoder-attn-backend=FLASHINFER` on Blackwell GPUs.

### Configuration

Three fields in `CompilationConfig` control encoder CUDA Graphs:

* `cudagraph_mm_encoder` (`bool`, default `False`) — enable CUDA Graph capture for multimodal encoder. When enabled, captures the full encoder forward as a CUDA Graph for each token budget level.
* `encoder_cudagraph_token_budgets` (`list[int]`, default `[]`) — token budget levels for capture. If empty (default), auto-inferred from model architecture as power-of-2 levels. User-provided values override auto-inference.
* `encoder_cudagraph_max_images_per_batch` (`int`, default `0`) — maximum number of images per batch during capture. If 0 (default), auto-inferred as `max_budget // min_budget`.

### Usage guide

Enable encoder CUDA Graphs via `compilation_config`:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

With explicit budgets:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_images_per_batch": 8}'
```

Python example:

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # Optional: override auto-inferred budgets
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_images_per_batch": 8,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    compilation_config=compilation_config,
)
```

The manager tracks hit/miss statistics and logs them periodically. A "hit" means an image was processed via CUDA Graph replay; a "miss" means eager fallback (image exceeded all budgets).

### About the Performance

The following benchmarks were run on Blackwell GPUs (GB200) using `vllm bench mm-processor`. See [#35963](https://github.com/vllm-project/vllm/pull/35963) for full details.

#### Single GPU (1x GB200)

Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`, dataset: `lmarena-ai/VisionArena-Chat` (3000 prompts, 300 warmup), `max_model_len=32768`.

| Backend | Mean latency improvement | P99 latency improvement |
| :------ | :----------------------- | :---------------------- |
| FLASH_ATTN | +11.8% (5.13→4.52ms) | +31.6% (9.16→6.26ms) |
| FLASHINFER | +19.6% (5.42→4.36ms) | +40.3% (10.87→6.49ms) |

To reproduce:

```bash
vllm bench mm-processor \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 3000 --num-warmups 300 \
  --max-model-len 32768 --seed 42 \
  --mm-encoder-attn-backend FLASH_ATTN \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_images_per_batch": 8}'
```

#### Multi-GPU (4x GB200, TP=4, DP=4)

Model: `Qwen/Qwen3-VL-32B-Instruct`, dataset: `random-mm` (1000 prompts, 200 warmup, 20 images/request at 336x336), `max_model_len=8192`.

| Backend | Mean latency improvement | P99 latency improvement |
| :------ | :----------------------- | :---------------------- |
| FLASH_ATTN | +18.4% (28.39→23.16ms) | +14.0% (238.78→205.28ms) |
| FLASHINFER | +44.4% (23.24→12.91ms) | +84.9% (172.41→26.05ms) |

To reproduce:

```bash
vllm bench mm-processor \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --dataset-name random-mm \
  --random-mm-base-items-per-request 20 \
  --random-mm-num-mm-items-range-ratio 0.0 \
  --random-mm-bucket-config '{"(336,336,1)": 1.0}' \
  --num-prompts 1000 --num-warmups 200 \
  --max-model-len 8192 --seed 42 \
  --mm-encoder-attn-backend FLASHINFER \
  --tensor-parallel-size 4 --mm-encoder-tp-mode data \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_images_per_batch": 8}'
```
