# DSpine

DSpine extends the DFlash backbone with per-layer code bridges: after each
active decoder layer, a gated write injects the predecessor row's features into
the next row, and a final last-write refinement scores every
(predecessor candidate, successor candidate) pair. It requires a compatible
trained checkpoint and Model Runner V2, which is selected automatically.

## Usage

For a checkpoint trained with block size 16:

```bash
vllm serve /path/to/target --dtype bfloat16 \
    --speculative-config '{"method":"dspine","model":"/path/to/dspine","num_speculative_tokens":15}'
```

`"method":"dflash"` with the same checkpoint also works: vLLM rewrites a draft
whose `dflash_config` carries `selected_prefix_code_bridge` to
`DSpineDraftModel` and dispatches the DSpine speculator. The `dspine` alias is
normalized to the shared `dflash` runtime.

`num_speculative_tokens` must be exactly the checkpoint's `block_size - 1`
(for example, `15` for `block_size=16`): the code bridge is trained for one
anchor row plus a fixed number of mask rows, and other lengths are rejected at
startup rather than silently degrading acceptance length.

## Proposal sampling

- `"draft_sample_method":"greedy"` (default) walks the transition table with
  argmax. The final RMSNorm denominator is omitted because its positive scale
  cannot change the argmax.
- `"draft_sample_method":"probabilistic"` keeps that denominator, samples from
  the request's temperature and filters over the K candidates, and supplies the
  realized sparse proposal distribution to the rejection sampler.

Both modes reuse the DFlash2 candidate walk and draft-logit cache. With request
temperature zero, probabilistic proposals also use argmax.

## Checkpoint requirements

The checkpoint must declare `DFlashDraftModel` or `DSpineDraftModel` and ship:

- `dflash_config.selected_prefix_code_bridge` with `code_size`, `rank`, and
  `eps`.
- `dflash_config.selected_prefix_feature_layers` (1-based, must include the
  last layer) and `selected_prefix_feature_writes = true` with
  `selected_prefix_feature_write_mode = "independent"`.
- `dflash_config.draft_block_causal = true`; DSpine requires causal draft
  attention and full attention layers.
- A refinement width in `code_bridge_second_pass` (mode `serial_table`) and/or
  `dflash_config.selected_prefix_last_write_refine` (`kind: exact_sender`), a
  power of two in `[2, 64]`.

Bridge tensors live under `selected_prefix_bridge.*` and include `codebook`,
`read_weight`, `early_read_weights`, `feature_write_weights`, `gate`,
`predecessor_gate`, `prev`, `out`, and the two version scalars. The loader
requires every non-optional parameter to come from the checkpoint and rejects a
nonzero terminal projection or an unsupported version.

The draft shares the target input embedding and LM head; both must be
unquantized, row-contiguous, and on the draft rank. Only tensor parallel size 1
is supported, and FP4 KV caches are rejected.

## Numerics

The refinement table is computed in fp32 from the last write's pre-write
residual stream, RMS, and receiver query, so the draft model never materializes
`[batch, steps, K, hidden]` refined states. The bridge has an eager path and a
fused Triton path; both share rounding points, and the fused path is used when
activations are BF16 on CUDA.

Greedy outputs match non-speculative decoding token for token. As with any
speculative method, a BF16 target can still diverge on near-ties because
verification batches have a different shape than decode batches; run with
`VLLM_BATCH_INVARIANT=1` when exact equality across configurations is required.
