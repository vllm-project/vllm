# #69306: Whisper encoder admission -> eager decoder fallback

**Status: diagnostic only, no CUDA Graph correctness patch.**

## Existing behavior to verify

`vllm/v1/worker/gpu/model_runner.py` sets `skip_compiled=True` if
`self.is_encoder_decoder` and `scheduler_output.scheduled_encoder_inputs`
are both true. Encoder-decoder graph capture is constrained by cross-attention
`kv_proj` work performed inside the decoder forward. Removing the eager gate
without relocating/updating the cross-attention cache is **not safe**.

## Measurement

Run `openai/whisper-large-v3` (or the report's checkpoint) under a high concurrency
transcription workload on an NVIDIA GPU with `--cudagraph-metrics --enable-logging-iteration-details` and keep
request statistics enabled. Use a single API server for iteration logs. Collect complete per-step engine logs,
then execute:

```bash
python benchmarks/issues/69306/analyze_whisper_steps.py whisper_engine.log \
  --output-json whisper_admission_stats.json
```

The analyzer classifies admission steps and reports their share of iteration
time when every iteration is timed. Current main records eligible GPU worker
timings with CUDA events. Report timing coverage alongside the fraction and
use the CUDA graph metrics to distinguish replay from eager execution. Compare outputs and
latency against `--enforce-eager` for a baseline; use a reproducible audio
corpus and record sample rate, duration, concurrency, decoder output,
WER (word-error rate), TPOT, GPU time, and activation/KV usage.

## Implementation acceptance gate

1. Factor encoder cross-attention projection / cache writes out of the
   graph-replayed decoder path, with event/stream dependencies correctly ordered.
2. Ensure the captured graph sees the correct per-request cache and respects
   preemption, mixed prefill/decode, LoRA, and varying encoder lengths.
3. Add correctness tests against the current eager path including joining
   encoder inputs while earlier requests decode.
4. Prove actual throughput improvement on H100 (or an appropriately comparable
   NVIDIA GPU); the reported `+29%` is an estimate, not observed speedup.

Do not submit a performance-claim PR before this validation.
