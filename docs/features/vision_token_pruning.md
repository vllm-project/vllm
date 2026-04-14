bash
python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen3-VL-32B-Instruct \
--image-pruning-rate 0.3
This configuration drops the least important ~30% of image tokens before fusion
into the Qwen VL decoder.
## Supported models
Attention-score-based image token pruning is currently implemented for:
- `Qwen2_5_VLForConditionalGeneration`
- `Qwen3VLMultiModalForConditionalGeneration`
- `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`
For other multimodal models, the engine will ignore `--image-pruning-rate` and
fall back to the default behavior.
## How it works (high level)
1. The ViT encoder runs as usual to produce per-patch hidden states.
2. At a configurable layer (`vit_attention_score_layer_index`), the model:
   - runs FlashAttention with a custom wrapper to obtain **softmax log-sum-exp**
     values (`softmax_lse`), and
   - uses a Triton kernel to reconstruct **per-token importance scores** from
     these values.
3. The scores are aggregated over spatial cells (following the VisionZip
   dominant-token selection scheme) to produce a scalar importance per token.
4. The lowest-scoring tokens are pruned according to `image_pruning_rate`.
5. mRoPE positions and multimodal metadata are updated to match the pruned
   sequence length, including data-parallel sharded execution.
The pruning happens entirely inside the **vision stack**; the LLM sees a
shorter sequence of image tokens with unchanged text tokens.
## Recommended configurations
Based on internal benchmarks on Qwen3‑VL‑32B‑Instruct:
- **Multi‑image tasks** (e.g. 30 images / sample, multi-image QA):
  - `--image-pruning-rate 0.3`–`0.4`  
    Preserves or slightly improves integrated accuracy, while significantly
    reducing prefill latency.
- **Non‑OCR single‑image tasks** (perception, reasoning, general VQA):
  - `--image-pruning-rate 0.3`–`0.5`  
    Typically <2% absolute accuracy drop across CC‑Bench, MMBench, AI2D,
    MMMU‑dev.
- **OCR‑heavy tasks** (OCRBench, DocVQA):
  - `--image-pruning-rate <= 0.3`  
    Higher pruning rates can noticeably hurt fine‑grained text recognition.
### Performance
On a multi‑image benchmark (Qwen3‑VL‑32B‑Instruct, 30 images / sample,
~43.5% vision tokens):
- Enabling score extraction adds ~30 ms to the ViT encoder.
- Reducing image tokens yields:
  - **−8% TTFT** at 30% pruning,
  - **−13% TTFT** at 40% pruning,
  - **−17% TTFT** at 50% pruning.
- Decode latency is effectively unchanged; savings come from a cheaper prefill.
## Limitations and notes
- This feature currently targets **image tokens** only. Video token pruning
  (e.g. EVS) is handled by a separate path; in Qwen3‑VL these mechanisms can
  coexist.
- Very aggressive pruning rates (e.g. ≥0.5 on hard multi‑image tasks or OCR)
  can noticeably degrade structured metrics (edit distance, alignment).
- The feature relies on the FlashAttention backend; non‑Flash backends
  currently fall back to the standard path without attention score extraction.