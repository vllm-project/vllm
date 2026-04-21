# Vision Encoder (ViT) CUDA Graphs

The [CUDA Graphs](cuda_graphs.md) infrastructure in vLLM primarily targets the **decoder** (language model) forward pass. vLLM also supports capturing the **encoder** (vision transformer) forward pass as CUDA Graphs, independently from the decoder. This is based on <https://github.com/vllm-project/vllm/pull/35963>.

!!! note
    Encoder CUDA Graphs are orthogonal to decoder CUDA Graphs — both can be enabled simultaneously. Encoder graphs capture the vision encoder execution (e.g., ViT in Qwen3-VL), while decoder graphs capture the language model execution as described in the [CUDA Graphs design document](cuda_graphs.md).

## Motivation

Vision encoder inference incurs CUDA kernel launch overhead on the host side. The overhead is more significant when the batch size is small or image size is small.

Encoder CUDA Graphs eliminate this overhead by pre-capturing the full encoder forward pass at multiple token budget levels during model initialization, then replaying the appropriate graph at runtime.

## Design

The encoder CUDA Graph system uses a **budget-based capture/replay** strategy, managed by [EncoderCudaGraphManager][vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager]. The system contains the following core components:

* [EncoderCudaGraphManager][vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager]: orchestrates capture, replay, greedy packing, and data-parallel execution for encoder CUDA Graphs.
* [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph]: a runtime-checkable protocol that models implement to opt-in to encoder CUDA Graphs.
* [BudgetGraphMetadata][vllm.v1.worker.encoder_cudagraph.BudgetGraphMetadata]: holds the captured CUDA Graph and its associated I/O buffers for a single token budget level.

### Budget-based graph capture

Multiple CUDA Graphs are pre-captured at different **token budget** levels (e.g., `[2048, 4096, 8192, 13824]`). Each budget defines a fixed token capacity, and all budgets share the same maximum batch size (number of images). The `BudgetGraphMetadata` for each level stores the graph along with pre-allocated input, metadata, and output buffers:

```python
@dataclass
class BudgetGraphMetadata:
    token_budget: int
    max_batch_size: int
    max_frames_per_batch: int
    graph: torch.cuda.CUDAGraph
    input_buffer: torch.Tensor       # e.g. pixel_values
    metadata_buffers: dict[str, torch.Tensor]  # e.g. embeddings, seq metadata
    output_buffer: torch.Tensor      # encoder hidden states
```

Budgets are auto-generated as power-of-2 levels from a model-provided range via `get_encoder_cudagraph_budget_range()`, with the maximum budget always included even if it does not fall on a power-of-2 boundary. Budgets can also be explicitly specified by the user via `encoder_cudagraph_token_budgets` in `CompilationConfig`.

### Greedy bin-packing at runtime

When a batch of images arrives, the manager sorts images by output token count (smallest first) and greedily packs as many images as possible into each sub-batch while staying within the **largest** token budget and the maximum batch size. Once a sub-batch is finalized (the next image would overflow either constraint), the manager finds the **smallest** budget that fits the sub-batch's total tokens and replays the corresponding CUDA Graph. This repeats until the batch is exhausted. Images that exceed all budgets fall back to eager execution.

For each graph replay:

1. Zero the pre-allocated `input_buffer`, then copy input tensors (e.g., `pixel_values`) into it.
2. Zero `metadata_buffers`, then slice-copy precomputed values (e.g., rotary embeddings, sequence metadata).
3. Replay the CUDA Graph.
4. Clone outputs from `output_buffer` (cloning is necessary since the buffer is reused across replays).

### Data-parallel support

When `mm_encoder_tp_mode="data"`, the manager distributes images across TP ranks using load-balanced assignment via `get_load_balance_assignment`, executes locally on each rank, then gathers results back in the original order via `tensor_model_parallel_all_gather`.

### Video inference support (experimental)

Following <https://github.com/vllm-project/vllm/pull/35963> (ViT full CUDA graph support for image inference), <https://github.com/vllm-project/vllm/pull/38061> extends the encoder CUDA graph framework to support video inference for Qwen3-VL. Previously, the CUDA graph capture/replay path only handled image inputs (`pixel_values` + `image_grid_thw`). Video inputs use different keys (`pixel_values_videos` + `video_grid_thw`) and require larger `cu_seqlens` buffers because each video item contributes multiple frames (`T` attention sequences). This PR generalizes the protocol and manager to handle both modalities through a single shared graph manager.

!!! note
    Video CUDA graphs are automatically disabled when EVS (Efficient Video Sampling) pruning is enabled, since EVS makes the token count data-dependent and incompatible with CUDA graph capture.

    Currently, we only support image-only or video-only inputs when enabling CUDA graph, mixed inputs (image + video) are not supported yet (we will work on it in the near future). Thus, it's recommended to turn off the image modality by `--limit-mm-per-prompt '{"image": 0}'` for video-only inputs.

## Model integration via `SupportsEncoderCudaGraph`

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
* `get_input_modality(...)` - return the modality of the inputs.

!!! note
    The `SupportsEncoderCudaGraph` protocol is designed to be model-agnostic. New vision encoder models can opt-in by implementing the protocol methods without modifying the manager.

**Supported models:**

| Architecture | Models | CG for Image | CG for Video |
| ------------ | ------ | ------------ | ------------ |
| `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |

!!! note
    Encoder CUDA Graphs have currently been tested with `--mm-encoder-attn-backend=FLASH_ATTN` and `--mm-encoder-attn-backend=FLASHINFER` on Blackwell GPUs.

## Configuration

Three fields in `CompilationConfig` control encoder CUDA Graphs:

* `cudagraph_mm_encoder` (`bool`, default `False`) — enable CUDA Graph capture for multimodal encoder. When enabled, captures the full encoder forward as a CUDA Graph for each token budget level.
* `encoder_cudagraph_token_budgets` (`list[int]`, default `[]`) — token budget levels for capture. If empty (default), auto-inferred from model architecture as power-of-2 levels. User-provided values override auto-inference.
* `encoder_cudagraph_max_vision_items_per_batch` (`int`, default `0`) — maximum number of images/videos per batch during capture. If 0 (default), auto-inferred as `max_budget // min_budget`.
* `encoder_cudagraph_max_frames_per_batch` (`int`, default `0`) — maximum number of video frames per batch during capture. If 0 (default), auto-inferred as `encoder_cudagraph_max_vision_items_per_batch * 2` (to be optimized).

## Usage guide

### Image inference

Enable encoder CUDA Graphs via `compilation_config`:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

With explicit budgets:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

Python example:

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # Optional: override auto-inferred budgets
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_vision_items_per_batch": 8,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    compilation_config=compilation_config,
)
```

The manager tracks hit/miss statistics and logs them periodically. A "hit" means an image was processed via CUDA Graph replay; a "miss" means eager fallback (image exceeded all budgets).

### Video inference

Enable encoder CUDA Graphs via `compilation_config`:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --limit-mm-per-prompt '{"image": 0}' \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

With explicit budgets:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --limit-mm-per-prompt '{"image": 0}' \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_vision_items_per_batch": 8, "encoder_cudagraph_max_frames_per_batch": 64}'
```

Python example:

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # Optional: override auto-inferred budgets
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_vision_items_per_batch": 8,
    # "encoder_cudagraph_max_frames_per_batch": 64,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    limit_mm_per_prompt='{"image": 0}',
    compilation_config=compilation_config,
)
```

## About the Performance

The following benchmarks were run on Blackwell GPUs (GB200) using `vllm bench mm-processor`. See [#35963](https://github.com/vllm-project/vllm/pull/35963) for full details.

### Single GPU (1x GB200)

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
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

### Multi-GPU (4x GB200, TP=4, DP=4)

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
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

!!! note
    Find more details about benchmarks on GPUs (A100) for video inference at [#38061](https://github.com/vllm-project/vllm/pull/38061).
