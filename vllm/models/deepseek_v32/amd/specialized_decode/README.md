# GLM-5.2 specialized decode (AMD)

An opt-in decode path for GLM-5.2 on MI355X. It runs each MoE decoder layer
(layers 3–77) as one persistent FlyDSL kernel inside vLLM's layer loop. Prefill,
mixed batches and batches above `MAX_BATCH_SIZE` run vLLM's decode unchanged. The
interface is the `SpecializedDecode` shape of #61867.

## Enabling

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <GLM-5.2 checkpoint> -tp 8 \
  --additional-config '{"decode_backend": "latency_specialized"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

`decode_backend` is `"auto"` (off) by default. If a deployment isn't supported,
startup fails with one error that lists every reason.

## Supported checkpoints

Attention precision follows the checkpoint. Nothing is re-quantized at load time.

| Checkpoint | Attention | Indexer layers | Validated |
| --- | --- | --- | --- |
| `amd/GLM-5.2-MXFP4` (MXFP4 experts) | BF16 | vLLM's indexer (`indexer_only`) | GPU: speed, GSM8K |
| `amd/GLM-5.2-Quark-MXFP4-AttnFP8` (MXFP4 experts, Quark per-channel FP8 attention), `max_model_len` ≤ 4096 | FP8, with the checkpoint's per-row scales | Fused into the kernel | GPU: speed, GSM8K |
| Same, `max_model_len` > 4096 | FP8, with the checkpoint's per-row scales | vLLM's indexer | CPU tests and compile only |

The fused indexer also needs an FP8 `indexer.wq_b` and a first mono layer that is not
an indexer layer; otherwise vLLM's indexer runs. The startup log names the path and,
when the fused indexer is off, why.

Any other attention quantization is refused at startup. That includes block or
per-tensor FP8, a non-Quark method, and FP8 on only some layers or matrices.

## Supported deployments

- **Hardware:** gfx950 (MI350X / MI355X), TP = 8, Model Runner V2.
- **Precision:** `dtype=bfloat16`; `kv_cache_dtype` auto or bfloat16.
- **Graphs:** FULL decode CUDA graphs or eager mode. Any capture sizes and
  `max_num_seqs` work: captured sizes up to `MAX_BATCH_SIZE` run the kernels, larger
  ones capture vLLM's decode. PIECEWISE-only graphs run every decode step through
  vLLM's decode.
- **Refused at startup:**
    - PP, DP, EP, DCP, PCP, DBO;
    - speculative decoding, LoRA, sleep mode, KV-transfer connectors, routed-expert
    capture;
    - AITER fused shared experts;
    - `STOCK_TORCH_COMPILE`;
    - MRV1.
- **Checked when the executor is built:** the MoE layout and backend
  (`AITER_MXFP4_MXFP4`), the attention weights and the RoPE table.
- **Checked at every KV bind:** the caches are bound and contiguous, use the MLA row
  layout, and are under 4 GiB.

## Batch sizes

Two constants in `runtime/__init__.py` set the limits:

- `BATCH_SIZES = (1, 2, 4, 5, 6, 8, 10, 12)` is every batch size the kernel can be
  built for.
- `MAX_BATCH_SIZE = 8` is the largest batch the kernels serve.

A batch pads up to the next built size, so 3 requests run as 4. Above 8 the kernel
switches to a second row group, and the gain over vLLM falls off quickly (BF16,
3000-token context, TPOT):

| Batch | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM (ms) | 13.71 | 14.71 | 15.52 | 16.59 | 17.38 | 18.44 | 19.51 |
| Specialized decode (ms) | 5.94 | 6.81 | 8.46 | 10.13 | 11.55 | 14.84 | 18.41 |
| Speedup | 2.31× | 2.16× | 1.83× | 1.64× | 1.50× | 1.24× | 1.06× |

FP8-attention checkpoint (`amd/GLM-5.2-Quark-MXFP4-AttnFP8`, fused indexer), against
vLLM on the same checkpoint:

| Batch | 1 | 2 | 4 | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vLLM (ms) | 14.48 | 15.24 | 15.85 | 16.68 | 17.40 | 18.23 | 19.39 |
| Specialized decode (ms) | 4.89 | 5.98 | 7.34 | 8.74 | 10.27 | 14.83 | 17.24 |
| Speedup | 2.96× | 2.55× | 2.16× | 1.91× | 1.69× | 1.23× | 1.12× |

Both variants fall off the same way above 8, so `MAX_BATCH_SIZE` is 8 for both. GSM8K
(5-shot, greedy, 1319 questions, strict): 1247 BF16; 1230 FP8, against 1226 for vLLM on
the same FP8 checkpoint.

## Options

These go in `additional_config["specialized_decode"]`. Unknown keys and invalid
values are refused at startup.

| Key | Default | Purpose |
| --- | --- | --- |
| `indexer_mode` | `"indexer_only"` | How non-fused indexer layers refresh the top-k: vLLM's indexer alone, or `"attn"` (vLLM's whole attention, slower) |
| `layers` | 3–77 | Decoder layers that get a kernel |
| `failstop` | `"raise"` | Expired cross-rank waits: `"raise"` (fail-stop), `"warn"` (log and continue) or `"off"` (no fail-stop watch). With `"warn"` / `"off"` the eager health check still disables the kernels after an expiry |
| `poll_limit` | 20,000,000 | Bound on each cross-rank wait (about 4 s) |
| `step_sync`, `check_every` | on in eager mode, off under FULL graphs | Eager-mode health check (a device sync and a CPU all-reduce). `step_sync=True` is refused under FULL graphs; `check_every` has no effect there |

Debug only:

- `debug_no_fused_indexer`;
- `device_nonfinite`, `failstop_nonfinite`;
- `enabled`;
- `poll_early_out`, `indexer_trim`, `cache_hoist`, `split_keys64`.

## Layout

| Path | Holds |
| --- | --- |
| `runtime/__init__.py` | `DecodeConfig`, `BATCH_SIZES` / `MAX_BATCH_SIZE`, `specialized_decode_config()`, `unsupported()` |
| `runtime/decode.py` | `SpecializedDecode`: `prepare_kv`, `plan`, `Step.layer` (layer mode) |
| `runtime/mono_layers.py` | Per-layer kernels, step inputs, attention path selection, KV-bind checks |
| `runtime/vllm_weights.py` | Views of vLLM's loaded weights (BF16, FP8 per-channel, MXFP4 experts) |
| `runtime/indexer_only.py` | The non-fused indexer refresh |
| `runtime/guards.py` | Build checks, fail-stop and the poll-error watch (`PollErrorWatch`) |
| `flydsl/` | The vendored FlyDSL kernel (JIT-compiled) |

The model hooks are in `../model.py` and `../model_state.py`. The tests are in
`tests/models/deepseek_v32/amd_specialized_decode/`.
