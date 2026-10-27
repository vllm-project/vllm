# Unlimited-OCR-2

Unlimited-OCR-2 uses DASH routing and page-aware two-axis RoPE. Its architecture
name is shared with Unlimited-OCR, but its attention and checkpoint are different.

Decode supports full CUDA graphs (`FULL_DECODE_ONLY`) without `torch.compile`.
Both prefill and decode support `FLASH_ATTN` (default) and `FLASHINFER`.
DASH gathers selected remote blocks and the local window, runs segmented Flash
attention and merges partial outputs using their LSEs plus the routing biases.
FlashInfer uses its FA2 ragged-prefill kernel for both phases; FlashAttention
selects FA2/FA3/FA4 according to hardware and `flash_attn_version`.
These paths require FP16 or BF16. Routing and paged storage remain DASH-specific;
the internal cache metadata backend is Triton for R-SWA reclamation.
`TRITON_ATTN` keeps the original Triton decode and PyTorch attention prefill.

Prefill supports optional piecewise CUDA graphs through main's breakable
capture runtime. Set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` before starting vLLM and
use `cudagraph_mode="FULL_AND_PIECEWISE"` (or leave the mode unset). QKV/router
projections, normalization, MoE and output projections are captured; dynamic
DASH routing, Flash attention and KV writes execute between graph segments.
Decode still uses a full graph. Whole-prefill `FULL` graphs are not supported.
With the environment variable unset, the default remains `FULL_DECODE_ONLY`.

Add prefill token buckets to `cudagraph_capture_sizes`, for example
`[1, 2, 1024, 2048, 4096]` for short documents. A bucket counts all scheduled
tokens, not pages or requests. Larger prefills fall back to eager execution;
large buckets consume additional graph memory and must fit the device budget.
The vision encoder has separate graph controls and is not captured by this
option. Use `PIECEWISE` to use segmented graphs for both phases.

Prefill still requires a complete prompt. The checkpoint must be
unquantized with the default KV dtype. Prefix caching is disabled. Pipeline
parallelism, context parallelism, speculative decoding, KV transfer and LoRA
are not supported. Use `enforce_eager=True` to disable graphs for debugging.

Decode reuses Unlimited-OCR's existing `RSWAAttention`, `RSWASpec` and
`RSWAManager`. Input KV is retained for DASH retrieval. Generated-token KV is
visible only in the 128-token local window; the manager frees whole output
blocks outside it, leaving null entries in the logical block table. Prefix and
window boundary blocks may retain extra physical tokens due to block alignment.
Decode masks the evicted gap before reading router keys or main KV.

Remote block summaries and fallback routing use only input-prefix tokens.
This applies R-SWA visibility to DASH, including mixed input/output blocks and
queries with no same-page image candidates. The published Unlimited-OCR-2
SGLang wheel uses full KV, so these edge paths can differ from that deployment.
Use `hf_overrides={"rswa_window": 0}` to retain the previous full-output-KV
behavior for comparison. The default window is derived from
`sparse_local_block * sparse_block_size`; other nonzero window sizes are rejected.

Decode caches FP32 router summaries for complete input blocks in unused value
lanes of the existing auxiliary KV head. Four exactly represented bytes per
FP32 value preserve the summary without BF16 rounding or additional KV blocks.
Partial prefix blocks and full-output-KV mode retain live summary computation.
Decode RoPE uses fused Triton kernels with cached frequencies and preserves the
reference intermediate FP16/BF16 rounding. Prefill uses page/block membership
lookups and linear-size coordinate scans instead of per-query token masks and
per-token/page cumulative tensors.

Router keys and integer page metadata share one auxiliary KV head per TP rank.
At TP=1 this uses 55% of the initial implementation's per-token KV storage.
BF16 with all 12 layers uses 67,584 bytes per token. For 1,000 input and 1,000
output tokens, logical retained KV is about 72.7 MiB (1,000 + 128 tokens), plus
block alignment and in-flight decode allocations. This excludes weights,
temporaries, the graph pool and unused space in the engine's reserved KV pool.

Each input page must have a `<PAGE><image>` pair in the prompt. Preserve the
output `<PAGE>` tokens with `skip_special_tokens=False`.
The reference Transformers API's optional page-count stopping guard is not
installed automatically; generation can repeat pages up to `max_tokens`.

```python
from PIL import Image
from vllm import LLM, SamplingParams

images = [Image.open(path).convert("RGB") for path in ["page1.png", "page2.png"]]
llm = LLM(
    model="baidu/Unlimited-OCR-2",
    enforce_eager=False,
    attention_config={"backend": "FLASH_ATTN"},  # or "FLASHINFER"
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    max_model_len=8192,
    max_num_batched_tokens=8192,
    max_num_seqs=2,
    limit_mm_per_prompt={"image": len(images)},
    mm_processor_kwargs={"crop_mode": False},
    mm_encoder_attn_backend="TORCH_SDPA",
    compilation_config={"cudagraph_capture_sizes": [1, 2]},
    kernel_config={"enable_jit_warmup": False},
)
outputs = llm.generate(
    {
        "prompt": "<PAGE><image>" * len(images) + "Multi page parsing.",
        "multi_modal_data": {"image": images},
    },
    SamplingParams(temperature=0, max_tokens=4096, skip_special_tokens=False),
)
print(outputs[0].outputs[0].text)
```

Set `max_num_batched_tokens` to at least `max_model_len` because chunked prefill
is disabled. The number of image pages and generated tokens must fit within
`max_model_len`. Dynamic cropping applies independently to every page, including
multi-page requests, matching the official Unlimited-OCR-2 Gundam preprocessing
with at most 16 local tiles per page.
Set `mm_processor_kwargs={"crop_mode": True}` for that mode; `crop_mode=False`
uses one 1024-pixel global view per page.

## Long documents

The [document example](../../../examples/generate/unlimited_ocr2.py) processes
bounded page windows, releasing each request's KV before submitting the next:

```bash
python examples/generate/unlimited_ocr2.py page*.png \
    --page-window 10 --overlap-pages 1 --attention-backend FLASHINFER
```

Each window contains at most ten input pages, including an optional overlapping
page. The overlap is re-encoded and omitted from the assembled output. Repeated
tail pages beyond the window's input page count are also omitted; missing page
markers raise an error. This bounds live per-request context independently of
the document's total page count. The engine's reserved KV pool remains fixed by
`gpu_memory_utilization` or `kv_cache_memory_bytes`.

Windows change cross-page context. This example does not reuse the preceding
window's KV tail or implement the customized SGLang document scheduler. Pass
`--page-window 0` to process the whole document in one request. Prefill plus the
output token budget for each window must fit within `max_model_len`.
