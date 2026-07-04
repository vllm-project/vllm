# Voxtral realtime benchmarks

Scripts to reproduce the realtime sliding-window / re-anchoring numbers
(KV-bounded streaming + unbounded duration). Hardware-agnostic; the numbers
quoted in the PR discussion were collected on a single RTX 4090 Laptop 16 GiB.

## Tools

- **`ws_load.py`**: drives N concurrent, real-time-paced WebSocket streams
  against a running vLLM server. Records per-stream wall vs audio seconds, the
  final transcript, and frame-to-partial latency.
  ```
  python ws_load.py -n 4 --audio clip.wav --out results.json
  python ws_load.py -n 1 --audio clip.wav --loops 7 --out long.json  # long session
  ```
- **`vram_probe.py`**: samples `nvidia-smi` VRAM, `vllm:kv_cache_usage_perc`, and
  the engine-core RSS at a fixed interval into a CSV, to show VRAM/KV stay flat
  over a run.
- **`test_reanchor_math.py`**: standalone numerical check that re-rotating a
  cached key by `R(-D)` and shifting the query position by `-D` leaves the
  attention score unchanged (the correctness basis for RoPE re-anchoring).
  ```
  python test_reanchor_math.py
  ```

## Example: KV-bounded serve on a single 16 GiB GPU

```
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --tokenizer-mode mistral \
  --hf-overrides '{"text_config":{"sliding_window":512},"audio_config":{"sliding_window":256}}' \
  --max-model-len 16384 --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

The decoder `sliding_window` controls concurrency: a smaller window means less
KV per stream, so you can run more streams at once. Add
`--enable-realtime-unbounded` for unbounded-duration sessions (experimental;
re-anchors the RoPE position clock so a session never reaches `max_model_len`).

## Reproducing the endurance run

The 4.7 h, 4-stream endurance result (VRAM spread 54 MiB under 0.4 %, `kv_usage`
bounded, wall/audio = 1.000 on all 4 streams, no crash across ~284 re-anchor
events) was collected with `--max-model-len 4096`, window 512/256 and
`--enable-realtime-unbounded`, sampling with `vram_probe.py` alongside
`ws_load.py -n 4 --loops 7`. (Raw CSV/JSON probe output is git-ignored by repo
policy, `.gitignore` `*.csv` / `benchmarks/**/*.json`; regenerate locally with
`vram_probe.py`.)
