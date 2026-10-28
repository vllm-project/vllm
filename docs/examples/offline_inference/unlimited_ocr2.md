# Unlimited-OCR-2

[Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) shares its
architecture name with Unlimited-OCR, but uses DASH routing and page-aware RoPE.
The v2 checkpoint is detected automatically; v1 processing remains unchanged.

## Usage

```bash
VLLM_USE_V2_MODEL_RUNNER=1 python examples/generate/unlimited_ocr2.py page*.png \
    --attention-backend FLASH_ATTN --max-model-len 32768 --max-tokens 16384
```

The [example](../../../examples/generate/unlimited_ocr2.py) submits the complete
ordered document. In Python, use `<PAGE><image>` per input page followed by
`Multi page parsing.`, and preserve output markers with `skip_special_tokens=False`.
Gundam cropping applies independently to each page with at most 16 local tiles,
matching the official v2 preprocessing. Set `mm_processor_kwargs={"crop_mode": False}`
for one 1024-pixel global view per page.

Chunked prefill is supported: `max_num_batched_tokens` can be smaller than
`max_model_len`, including chunks within an image's token span. Page positions
are computed from the full prompt once; each chunk reads earlier input KV from
the paged cache. Input and generated tokens must fit within `max_model_len`. By default there is no page-count stopping
guard; generation can repeat pages up to `max_tokens`. Pass
`--page-count-guard` (or register
`vllm.model_executor.models.unlimited_ocr2:PageCountGuardLogitsProcessor`
via `--logits-processors`) to force exactly one `<PAGE>` section per prompt
page: EOS is masked until all pages are done, extra page markers force EOS,
and `max_page_tokens` (default 20000) caps each page. The guard reads
`page_token_id` / `eos_token_id` / `max_page_tokens` from
`SamplingParams.extra_args` when given.
With Model Runner V2, the guard counts prompt markers once and processes new
output tokens on the GPU, without copying token history back to the CPU.

## Attention and KV cache

Both prefill and decode support `FLASH_ATTN` (default) and `FLASHINFER`, with
FP16/BF16 weights. DASH selects remote blocks and a local window, runs segmented
attention and merges outputs using their LSEs and routing biases. FlashAttention
selects FA2/FA3/FA4 according to hardware and `flash_attn_version`; FlashInfer uses
its FA2 ragged-prefill kernel for both phases. Triton's paged-cache metadata is
used internally for R-SWA; it is not a selectable DASH compute backend.

The existing R-SWA manager retains all input KV for DASH and reclaims whole output
blocks outside the 128-token local window. Block boundaries may retain extra
physical tokens. BF16/TP=1 KV uses 67,584 bytes/token across all 12 layers; 1,000
input + 1,000 output tokens retain about 72.7 MiB, excluding alignment, weights,
temporaries, Graph memory and the unused portion of the reserved KV pool.
Complete input-block routing summaries reuse auxiliary KV lanes.

Remote summaries and fallback routing only use the input prefix. This enforces
output-window visibility even for mixed input/output blocks or queries without
same-page image candidates, which can differ from the published custom SGLang
wheel's full-KV behavior. Use `hf_overrides={"rswa_window": 0}` for full output KV;
other nonzero window overrides are unsupported.

## CUDA Graphs and limitations

The default is full decode Graphs (`FULL_DECODE_ONLY`), without `torch.compile`.
Use `enforce_eager=True` to disable Graphs. For opt-in segmented prefill Graphs,
set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` before startup and select
`cudagraph_mode="FULL_AND_PIECEWISE"` (or leave the mode unset).
Projections, normalization and MoE are captured; dynamic DASH routing, attention
and KV writes execute in eager breaks. `PIECEWISE` segments both phases.
Whole-prefill `FULL` Graphs are unsupported; prefill speedup is workload dependent.

Add token buckets to `cudagraph_capture_sizes`, e.g. `[1, 2, 1024, 2048, 4096]`.
Buckets count scheduled tokens, not pages or requests. Larger prefills fall back
to eager; larger buckets consume additional Graph memory. The vision encoder
has separate Graph controls.

Weights must be unquantized with default KV dtype. Prefix caching, pipeline and
context parallelism, LoRA, speculative decoding, dual batch overlap and KV
transfer are unsupported. Validated on GB200/BF16; other hardware is unverified.
