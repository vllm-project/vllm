# PaddleOCR-VL Encoder CUDA Graph Support

## Purpose

This change makes PaddleOCR-VL use the vLLM `SupportsEncoderCudaGraph` path for
image inputs. It follows the same integration shape as the DeepSeek-OCR PR:
the model advertises encoder CUDA graph support, the shared manager packs image
items into fixed token budgets, and the model provides capture inputs, replay
buffers, graph forward, and eager fallback methods.

The captured graph includes both parts of the PaddleOCR-VL image encoder:

- the SigLIP vision tower, and
- the PaddleOCR-VL spatial-merge projector (`mlp_AR`).

The graph output is already in the text hidden size, so the shared manager can
scatter it exactly like the regular multimodal embedding path.

The scope is intentionally narrow: image encoder CUDA graph support for
PaddleOCR-VL on FA-style ViT attention backends, using the fixed-shape `values`
contract from PR #42288. The graph forward does not receive full `mm_kwargs`,
and this patch does not add PaddleOCR-specific SDPA or FlashInfer graph support.

## Why This Is Needed

The original PaddleOCR-VL eager image path is correct, but it is not friendly to
CUDA graph capture because several pieces of work depend on the exact image
shape at runtime:

- per-image position-embedding interpolation,
- rotary embedding index construction,
- `cu_seqlens` and `max_seqlen` construction,
- per-image projector reshaping with `einops.rearrange`, and
- per-request splitting and concatenation of variable-size patch tensors.

CUDA graph replay requires stable tensor objects and stable shapes in the
captured region. This change moves the dynamic work into metadata preparation
before capture or replay, then runs a fixed-shape forward during the graph.

## Runtime Flow

1. The multimodal processor produces `pixel_values` and `image_grid_thw` for
   each image.
2. `EncoderCudaGraphManager` asks the model for item specs with
   `get_encoder_cudagraph_item_specs()`, then greedily packs images into the
   smallest fitting token budget.
3. `select_encoder_cudagraph_items()` extracts the packed sub-batch by image
   index and keeps the matching patch tensors and grids together.
4. During graph capture, `prepare_encoder_cudagraph_capture_inputs()` creates
   dummy `pixel_values` plus fixed-shape metadata buffers for the budget.
5. During graph replay, `prepare_encoder_cudagraph_replay_buffers()` computes
   the real metadata for the selected images. The manager copies those values
   into the captured buffers, using PaddleOCR-specific padding for `cu_seqlens`.
6. `encoder_cudagraph_forward()` runs the batched SigLIP tower and `mlp_AR`
   projector from the precomputed metadata.
7. If a batch does not fit any captured budget, `encoder_eager_forward()` uses
   the original per-image eager path as the fallback.

## Function Map

These are the generated or materially changed functions and why each one is
necessary.

| Function | Necessity |
| --- | --- |
| `_pad_paddleocr_cu_seqlens_buffer()` | Pads replay `cu_seqlens` for FlashAttention-style backends so the captured buffer shape stays fixed while padded entries point at the captured dummy tail. |
| `Projector.forward_packed()` | Replaces per-image `rearrange` in `mlp_AR` with a precomputed `merge_index` gather plus reshape, which lets the projector live inside the CUDA graph. |
| `SiglipVisionEmbeddings.forward(..., position_embeddings=...)` | Allows the graph path to consume precomputed position embeddings instead of running per-image interpolation during capture/replay. |
| `SiglipEncoder.forward(..., rotary_pos_emb=..., max_seqlen=...)` | Allows the graph path to consume precomputed RoPE and attention metadata instead of creating indices or syncing `.max()` inside the graph. |
| `SiglipVisionTransformer.forward(..., encoder_metadata=...)` | Threads the precomputed metadata through embeddings and encoder layers with one model-internal entry point. |
| `SiglipVisionModel.prepare_encoder_metadata()` | Central precompute step. It builds position embeddings, RoPE embeddings, padded `cu_seqlens`, optional `max_seqlen`, and projector `merge_index` outside the graph. |
| `get_encoder_cudagraph_config()` | Tells the shared manager which modality is supported, which buffers are captured, what output hidden size to allocate, and how to pad `cu_seqlens`. |
| `get_input_modality()` | Declares that this integration only handles image inputs. |
| `get_max_frames_per_video()` | Returns `1` so the shared image/video manager can keep using one protocol for all models. |
| `get_encoder_cudagraph_budget_range()` | Provides automatic budget bounds when users do not pass explicit encoder graph token budgets. |
| `_get_image_grid_thw()` | Normalizes tensor/list `image_grid_thw` inputs into plain integer tuples for item specs and metadata generation. |
| `_get_pixel_values_list()` | Splits concatenated patch tensors back into per-image tensors when the processor provides a packed tensor. |
| `_get_cat_pixel_values()` | Produces the concatenated patch tensor expected by the packed graph forward. |
| `get_encoder_cudagraph_item_specs()` | Gives the manager each image's input patch count and output token count for packing and DP load balancing. |
| `select_encoder_cudagraph_items()` | Extracts only the images selected for one graph replay batch, preserving variable patch ranges correctly. |
| `prepare_encoder_cudagraph_capture_inputs()` | Creates the dummy fixed-shape capture buffers for one token budget and batch size. |
| `prepare_encoder_cudagraph_replay_buffers()` | Computes real replay buffer values from the selected request images before the manager copies them into captured buffers. |
| `encoder_cudagraph_forward()` | Defines the exact fixed-shape forward that is captured and replayed. |
| `encoder_eager_forward()` | Keeps the original path available for oversized inputs and for parity tests. |

## Test Plan

Run these from the repo root on a CUDA machine.

Existing MRoPE regression tests:

```bash
.venv/bin/python -m pytest tests/model_executor/test_paddleocr_vl_mrope.py -q
```

VLM CUDA graph generation test:

```bash
.venv/bin/python -m pytest \
  'tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_image[paddleocr_vl]' \
  -q
```

The shared ViT CUDA graph generation test is the PR-facing coverage point, in
the same style as the InternVL and DeepSeek-OCR encoder CUDA graph PRs. It
validates that the model can initialize, capture, replay, and generate with the
encoder graph path enabled.

E2E functional test, matching the style of the DeepSeek-OCR PR:

```bash
.venv/bin/python examples/generate/multimodal/vision_language_offline.py \
  -m paddleocr_vl \
  --modality image \
  --enable-vit-cuda-graph
```

For a PR test-result section, paste the generated text output under a heading
like this:

```text
E2E functional test:

--------------------------------------------------
<generated PaddleOCR-VL answer 1>
--------------------------------------------------
<generated PaddleOCR-VL answer 2>
--------------------------------------------------
```

## Benchmark Plan on an FA2 / FA3 GPU

Use `vllm bench mm-processor` and compare the same workload with
`cudagraph_mm_encoder` off and on. The commands below use dummy weights and
random image inputs, so they measure the vLLM encoder path without depending on
model quality.

Set up the environment:

```bash
export PATH="$PWD/.venv/bin:$PATH"
```

For FA2, use:

```bash
FA_VERSION=2
```

For FA3, use:

```bash
FA_VERSION=3
```

Run the eager baseline:

```bash
.venv/bin/python -m vllm.entrypoints.cli.main bench mm-processor \
  --model PaddlePaddle/PaddleOCR-VL \
  --trust-remote-code \
  --load-format dummy \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-encoder-attn-backend FLASH_ATTN \
  --attention-config.flash_attn_version="${FA_VERSION}" \
  --dataset-name random-mm \
  --random-mm-base-items-per-request 1 \
  --random-mm-num-mm-items-range-ratio 0 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(392,392,1): 1.0}' \
  --random-input-len 32 \
  --random-output-len 1 \
  --num-prompts 100 \
  --num-warmups 10 \
  --disable-tqdm \
  --seed 42 \
  --output-json /tmp/paddleocr_vl_encoder_eager.json \
  --compilation-config '{"cudagraph_mm_encoder": false}'
```

Run the encoder CUDA graph path:

```bash
.venv/bin/python -m vllm.entrypoints.cli.main bench mm-processor \
  --model PaddlePaddle/PaddleOCR-VL \
  --trust-remote-code \
  --load-format dummy \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-encoder-attn-backend FLASH_ATTN \
  --attention-config.flash_attn_version="${FA_VERSION}" \
  --dataset-name random-mm \
  --random-mm-base-items-per-request 1 \
  --random-mm-num-mm-items-range-ratio 0 \
  --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --random-mm-bucket-config '{(392,392,1): 1.0}' \
  --random-input-len 32 \
  --random-output-len 1 \
  --num-prompts 100 \
  --num-warmups 10 \
  --disable-tqdm \
  --seed 42 \
  --output-json /tmp/paddleocr_vl_encoder_cg.json \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [196], "encoder_cudagraph_max_vision_items_per_batch": 1}'
```

Compare the JSON outputs:

```bash
.venv/bin/python - <<'PY'
import json

with open("/tmp/paddleocr_vl_encoder_eager.json") as f:
    eager = json.load(f)
with open("/tmp/paddleocr_vl_encoder_cg.json") as f:
    cg = json.load(f)

for key in ("mean_e2el_ms", "median_e2el_ms"):
    before = eager.get(key)
    after = cg.get(key)
    if before and after:
        print(f"{key}: eager={before:.3f}ms cg={after:.3f}ms "
              f"improvement={(before - after) / before * 100:.2f}%")

print("\nEncoder-related stats:")
for name in sorted(set(eager.get("mm_processor_stats", {})) |
                   set(cg.get("mm_processor_stats", {}))):
    if "encoder" not in name.lower():
        continue
    print(name)
    print("  eager:", eager["mm_processor_stats"].get(name))
    print("  cg:   ", cg["mm_processor_stats"].get(name))
PY
```

## Local T4 Notes

The FA2/FA3 benchmark above is the meaningful validation target. I could not
get a reliable throughput number on the local Tesla T4 because:

- T4 cannot run FA2/FA3.
- SDPA encoder graph capture fails because the SDPA wrapper reads split lengths from a CUDA tensor during capture.
- SDPA and FlashInfer graph coverage are intentionally not part of this minimal PaddleOCR-VL patch; the implementation only advertises PaddleOCR encoder graph capture for FA-style ViT attention backends.
