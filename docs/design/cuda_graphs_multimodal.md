# Vision Encoder (ViT) CUDA Graphs

The [CUDA Graphs](cuda_graphs.md) infrastructure in vLLM primarily targets the **decoder** (language model) forward pass. vLLM also supports capturing the **encoder** (vision transformer) forward pass as CUDA Graphs, independently from the decoder. This is based on <https://github.com/vllm-project/vllm/pull/35963>.

For two-tower vision encoders (e.g., DeepSeek-OCR's SAM + CLIP with dynamic tiling), a **dual-path graph** mode captures two independent sets of CUDA graphs — one for the global image path and one for the local patch path — enabling independent budget selection and partial eager fallback per path. This is based on <https://github.com/vllm-project/vllm/pull/43586>.

!!! note
    Encoder CUDA Graphs are orthogonal to decoder CUDA Graphs — both can be enabled simultaneously. Encoder graphs capture the vision encoder execution (e.g., ViT in Qwen3-VL), while decoder graphs capture the language model execution as described in the [CUDA Graphs design document](cuda_graphs.md).

## Motivation

Vision encoder inference incurs CUDA kernel launch overhead on the host side. The overhead is more significant when the batch size is small or image size is small.

Encoder CUDA Graphs eliminate this overhead by pre-capturing the full encoder forward pass at multiple token budget levels during model initialization, then replaying the appropriate graph at runtime.

For two-tower vision encoders such as DeepSeek-OCR (SAM + CLIP with dynamic tiling), the global image path and local patch path have independent token profiles (272 tokens per global image vs. 100 tokens per local patch). Capturing a single monolithic graph for both paths would significantly reduce packing efficiency. The dual-path graph mode captures each path as a separate set of budgets, allowing the manager to pack and replay each path independently.

## Design

The encoder CUDA Graph system uses a **budget-based capture/replay** strategy, managed by [EncoderCudaGraphManager][vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager]. The system contains the following core components:

* [EncoderCudaGraphManager][vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager]: orchestrates capture, replay, greedy packing, and data-parallel execution for encoder CUDA Graphs.
* [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph]: a runtime-checkable protocol that models implement to opt-in to encoder CUDA Graphs.
* [EncoderItemSpec][vllm.v1.worker.encoder_cudagraph_defs.EncoderItemSpec]: describes a single encoder input item (image or video) with its input size and output token count.
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
    input_buffers: dict[str, torch.Tensor]  # e.g. pixel_values, embeddings, seq metadata
    output_buffer: torch.Tensor      # encoder hidden states
```

Budgets are auto-generated as power-of-2 levels from a model-provided range via `get_encoder_cudagraph_budget_range()`, with the maximum budget always included even if it does not fall on a power-of-2 boundary. Budgets can also be explicitly specified by the user via `encoder_cudagraph_token_budgets` in `CompilationConfig`.

When `EncoderCudaGraphConfig.enable_dual_path_graph` is `True`, the manager generates two independent budget lists — `global_token_budgets` (multiples of `global_token_per_image`) and `local_token_budgets` (multiples of `local_token_per_patch`) — and stores captured graphs under `budget_graphs["global"]` and `budget_graphs["local"]` respectively.

### Greedy bin-packing at runtime

When a batch of images arrives, the manager sorts images by output token count (smallest first) and greedily packs as many images as possible into each sub-batch while staying within the **largest** token budget and the maximum batch size. Once a sub-batch is finalized (the next image would overflow either constraint), the manager finds the **smallest** budget that fits the sub-batch's total tokens and replays the corresponding CUDA Graph. This repeats until the batch is exhausted. Images that exceed all budgets fall back to eager execution.

For dual-path models, the manager routes to `_execute_local_dual_path()`, which constrains both global and local token budgets simultaneously during packing (see [Dual-Path graph capture](#dual-path-graph-capture)).

For each graph replay:

1. Call `prepare_encoder_cudagraph_replay_buffers()` to compute buffer values (including `pixel_values` and precomputed metadata) from actual batch inputs.
2. Zero the pre-allocated `input_buffers`, then slice-copy the replay values into them.
3. Replay the CUDA Graph.
4. Clone outputs from `output_buffer` (cloning is necessary since the buffer is reused across replays).

### Dual-Path graph capture

For two-tower vision encoders (e.g., DeepSeek-OCR), the `EncoderCudaGraphConfig` sets `enable_dual_path_graph=True` and provides `global_token_per_image` / `local_token_per_patch`. The manager captures two independent sets of CUDA graphs — one for the **global** image path and one for the **local** patch path — stored under `budget_graphs["global"]` and `budget_graphs["local"]` respectively.

**Budget generation.** Two separate budget lists are generated:

* `global_token_budgets` — power-of-2 multiples of `global_token_per_image` (e.g., `[272, 544, 1088, 2176, 4352, 8704, 13824]` for DeepSeek-OCR).
* `local_token_budgets` — power-of-2 multiples of `local_token_per_patch` (e.g., `[0, 100, 200, 400, 800, 1600, 3200, 6400, 12800]` for DeepSeek-OCR). A budget of `0` is always included to handle images with no local patches (images ≤ 640×640 that produce only global features).

Both lists are capped at the same `max_budget`.

**Dual-path greedy packing.** Each `EncoderItemSpec` provides both `global_output_tokens` (constant per image) and `local_output_tokens` (proportional to the patch count). The dual-path packing algorithm constrains both budgets simultaneously:

* Sort images by total output tokens (global + local), smallest first.
* Greedily pack images: an image is added to the current sub-batch only if both the accumulated global tokens ≤ `max_global_budget` **and** the accumulated local tokens ≤ `max_local_budget`, with the image count ≤ `max_batch_size`.
* Once either constraint would overflow, finalize the sub-batch and find the smallest fitting budget **independently** for each path.
* Repeat until all images are packed.

**Partial graph fallback.** After packing, each sub-batch falls into one of four execution scenarios:

| Global budget | Local budget | Execution |
| :---: | :---: | --- |
| Found | Found | Both paths use CUDA graph replay |
| Found | `None` | Global graph replay + local path skipped (no patches) |
| `None` | Found | Global eager fallback + local graph replay |
| `None` | `None` | Both paths fall back to eager execution |

Note that the `0`-budget graph is never actually replayed for local — it signals that local patch processing should be skipped entirely.

**Buffer keys per path.** Global and local paths use different buffer keys. For DeepSeek-OCR, the global path uses `pixel_values` (full images, shape `[B, 3, 1280, 1280]`) while the local path uses `images_crop` (patches, shape `[P, 3, 1024, 1024]`). The manager iterates over each captured graph's own `input_buffers.keys()` rather than a shared `buffer_keys` list, so both paths can use different buffers.

**Post-processing.** The `postprocess_encoder_output` method receives a `local_output` parameter (a tensor or `None`) containing the local-path encoder output. The model is responsible for assembling global and local features into the final per-image embedding. For DeepSeek-OCR, this means reshaping the global output into `[B, 272, n_embed]`, the local output into `[P, 100, n_embed]`, assembling patch grids with newline tokens, and concatenating `[patches_grid, global, view_separator]` for each image.

!!! note
    The dual-path design enables partial CUDA graph coverage — one path can hit while the other falls back to eager. This avoids wasted compute on zero-padded patch buffers for untiled images and avoids graph invalidation caused by variable `crop_shape` per image.

### Data-parallel support

When `mm_encoder_tp_mode="data"`, the manager distributes images across TP ranks using load-balanced assignment via `get_load_balance_assignment`, executes locally on each rank, then gathers results back in the original order via `tensor_model_parallel_all_gather`.

### Video inference support

Following <https://github.com/vllm-project/vllm/pull/35963> (ViT full CUDA graph support for image inference), <https://github.com/vllm-project/vllm/pull/38061> extends the encoder CUDA graph framework to support video inference for Qwen3-VL. Previously, the CUDA graph capture/replay path only handled image inputs (`pixel_values` + `image_grid_thw`). Video inputs use different keys (`pixel_values_videos` + `video_grid_thw`) and require larger `cu_seqlens` buffers because each video item contributes multiple frames (`T` attention sequences). This PR generalizes the protocol and manager to handle both modalities through a single shared graph manager.

!!! note
    Video CUDA graphs are automatically disabled when EVS (Efficient Video Sampling) pruning is enabled, since EVS makes the token count data-dependent and incompatible with CUDA graph capture.

    Mixed inputs (image+video) per prompt are also supported now.

## Model integration via `SupportsEncoderCudaGraph`

Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph] protocol. This protocol encapsulates all model-specific logic so that the manager remains model-agnostic. The protocol defines the following methods:

* `get_encoder_cudagraph_config()` — returns static configuration (supported modalities, buffer keys, output hidden size, padding logics, max frames per video).
* `get_encoder_cudagraph_budget_range(vllm_config)` — returns `(min_budget, max_budget)` for auto-inference of token budgets.
* `get_encoder_cudagraph_item_specs(mm_kwargs)` — returns `list[EncoderItemSpec]` describing each item with its input size, total output token count (`output_tokens`), and optionally per-path token counts (`global_output_tokens`, `local_output_tokens`) for dual-path models.
* `select_encoder_cudagraph_items(mm_kwargs, indices)` — extracts a sub-batch of items by index, used during greedy packing and DP sharding.
* `prepare_encoder_cudagraph_capture_inputs(..., path="default")` — creates dummy inputs for graph capture. The `path` parameter (`"global"` or `"local"`) tells the model which path to generate dummy inputs for. Returns `EncoderCudaGraphCaptureInputs` with a single `values: dict[str, torch.Tensor]` that contains all buffers to be recorded into the graph.
* `prepare_encoder_cudagraph_replay_buffers(mm_kwargs, max_batch_size, max_frames_per_batch, path="default")` — computes buffer values from actual batch inputs. The `path` parameter selects which modality keys to extract from `mm_kwargs`. Returns `EncoderCudaGraphReplayBuffers` with a `values` dict whose keys match the captured graph's `input_buffers.keys()`.
* `encoder_cudagraph_forward(inputs: dict[str, torch.Tensor], path="default")` — forward pass accepting only fixed-shaped input tensors (the captured `values` dict). Called during both capture and replay. The `path` parameter dispatches to the correct encoder sub-module (e.g., global vs. local path for DeepSeek-OCR).
* `encoder_eager_forward(mm_kwargs, path="default")` — fallback eager forward when no graph fits. When `path` is `"global"` or `"local"`, runs only that encoder path without graph capture.
* `postprocess_encoder_output(..., local_output=None)` — post-process encoder output. The `local_output` parameter receives the local-path encoder output tensor (or `None`), enabling dual-path models to assemble global and local features into the final per-image embedding.

!!! note
    The `SupportsEncoderCudaGraph` protocol is designed to be model-agnostic. New vision encoder models can opt-in by implementing the protocol methods without modifying the manager.

**Supported models:**

| Architecture | Models | CG for Image | CG for Video | Dual-Path Graph |
| ------------ | ------ | ------------ | ------------ | --------------- |
| `DeepseekOCRForCausalLM` | `DeepSeek-OCR` | ✅︎ | ❌︎ | ✅︎ |
| `Glm4vForConditionalGeneration` | `GLM-4.1V, GLM-4.6V-Flash` | ✅︎ | ✅︎ | ❌︎ |
| `InternVLChatModel` | `InternVL3.5`, `InternVL3`, `InternVL2.5`, `InternVL2` | ✅︎ | ✅︎ | ❌︎ |
| `KimiVLForConditionalGeneration` | `Kimi-VL` | ✅︎ | ❌︎ | ❌︎ |
| `Llama4ForConditionalGeneration` | `Llama 4` | ✅︎ | ❌︎ | ❌︎ |
| `Qwen2VLForConditionalGeneration` | `Qwen2-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3_5ForConditionalGeneration` | `Qwen3.5`, `Qwen3.6` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3_5MoeForConditionalGeneration` | `Qwen3.5-MoE`, `Qwen3.6-MoE` | ✅︎ | ✅︎ | ❌︎ |
| `Step3VLForConditionalGeneration` | `Step3-VL` | ✅︎ | ❌︎ | ✅︎ |

!!! note
    Encoder CUDA Graphs have currently been tested with `--mm-encoder-attn-backend=FLASH_ATTN` and `--mm-encoder-attn-backend=FLASHINFER` on Blackwell GPUs.
    For Qwen2-VL and Qwen2.5-VL only FA2 and FA3 has been tested.

## Configuration

Three fields in `CompilationConfig` control encoder CUDA Graphs:

* `cudagraph_mm_encoder` (`bool`, default `False`) — enable CUDA Graph capture for multimodal encoder. When enabled, captures the full encoder forward as a CUDA Graph for each token budget level.
* `encoder_cudagraph_token_budgets` (`list[int]`, default `[]`) — token budget levels for capture. If empty (default), auto-inferred from model architecture as power-of-2 levels. User-provided values override auto-inference.
* `encoder_cudagraph_max_vision_items_per_batch` (`int`, default `0`) — maximum number of images/videos per batch during capture. If 0 (default), auto-inferred as `max_budget // min_budget`.
* `encoder_cudagraph_max_frames_per_batch` (`int`, default `None`) — maximum number of video frames per batch during capture. If `None` (default), auto-inferred as `encoder_cudagraph_max_vision_items_per_batch * max_frames_per_video` (`max_frames_per_video` is a model-specific value from `EncoderCudaGraphConfig`, computed by `get_max_frames_per_video()` on the model). If we limit the video count per prompt to `0`, it will also be set to `0` (i.e., fall back to image-only mode).

Dual-path mode is configured at the model level via `EncoderCudaGraphConfig` fields (`enable_dual_path_graph`, `global_token_per_image`, `local_token_per_patch`) — no additional user configuration is required. The manager automatically generates separate budget lists and routes to dual-path execution when the model opts in.

## Usage guide

### Image inference

Enable encoder CUDA Graphs via `compilation_config`:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

For `Llama 4` (image only):

```bash
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --limit-mm-per-prompt '{"image": 1}' \
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
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

With explicit budgets:

```bash
vllm serve Qwen/Qwen3-VL-32B \
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
