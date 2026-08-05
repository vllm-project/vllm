# vllm-startup-bench

`vllm-startup-bench` measures and compares process **startup latency**
("time-to-ready": wall clock from process spawn until an HTTP readiness
endpoint first responds successfully) across one or more `vllm serve`-style
commands.

It exists to give concrete, reproducible before/after numbers for changes on
vLLM's startup path (e.g. Python frontend vs the Rust frontend, or two
candidate implementations of a piece of startup logic), instead of relying on
anecdotal impressions of "faster". It is not a load-testing tool — for
steady-state serving throughput, use `vllm-bench` instead.

## How it works

For each `--variant NAME=COMMAND`, each run:

1. Spawns `COMMAND` via `sh -c` in its own process group.
2. Polls `--health-url` (default `http://127.0.0.1:8000/health`, matching
   both the Python and Rust frontends' readiness endpoint) until it returns a
   successful HTTP status, the process exits, or `--ready-timeout-secs`
   elapses.
3. Records the elapsed time, then terminates the whole process group
   (SIGTERM, escalating to SIGKILL after `--shutdown-timeout-secs`) so the
   next run starts from a clean slate.

Runs are interleaved round-robin across variants (run 1 of A, run 1 of B, run
2 of A, ...) rather than run back-to-back per variant, so that any
systematic drift over the session (thermal throttling, filesystem cache
state, etc.) is spread evenly across variants instead of confounding it with
run order. `--warmup-runs` runs (default 1) execute first and are discarded,
to avoid one-time cold-cache effects unrelated to the code under test
skewing the timed samples.

The first `--variant` given is treated as the baseline; the report shows
each other variant's median time-to-ready relative to it, and explicitly
flags a variant as a **regression** if it's slower, rather than only ever
reporting speedups.

## Usage

Compare plain Python `vllm serve` against the Rust frontend for the same
model (both bind the same port, so only one runs at a time):

```bash
vllm-startup-bench \
  --variant "python=vllm serve Qwen/Qwen3-0.6B" \
  --variant "rust=VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B" \
  --runs 5 --warmup-runs 1
```

Illustrative output (actual numbers depend on model, hardware, and what's
being compared):

```text
=== Startup time-to-ready (http://127.0.0.1:8000/health) ===

variant             runs  failures    mean(s)  median(s)     min(s)     max(s)  stddev(s)
python                 5         0     12.481     12.402     12.011     13.055      0.412
rust                   5         0     11.803     11.769     11.520     12.244      0.271

=== Comparison vs. baseline "python" (median) ===

rust             1.05x faster  (-0.633s, -5.1%)
```

Use `--log-dir` to capture each run's stdout/stderr for debugging a variant
that fails to become ready, and `--save-result` to write the full sample set
and summary statistics as JSON for later analysis:

```bash
vllm-startup-bench \
  --variant "python=vllm serve Qwen/Qwen3-0.6B" \
  --variant "rust=VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B" \
  --runs 10 --log-dir /tmp/startup-bench-logs --save-result results.json
```

Any two commands that expose an HTTP readiness endpoint work, not just
`vllm serve` — e.g. to isolate just the frontend layer against an
already-running headless engine (see `rust/README.md`'s "External Engine"
section), or to compare two Rust implementations of some startup-path logic
against each other directly.

## CLI Reference

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--variant NAME=COMMAND` | — | One variant to benchmark (repeatable, required, first = baseline) |
| `--health-url` | `http://127.0.0.1:8000/health` | URL polled until it returns a successful status |
| `--runs` | `5` | Timed runs per variant |
| `--warmup-runs` | `1` | Untimed runs per variant, discarded, run before timed runs |
| `--ready-timeout-secs` | `300` | Max time to wait for readiness per run |
| `--poll-interval-ms` | `50` | Interval between readiness poll attempts |
| `--shutdown-timeout-secs` | `15` | Grace period after SIGTERM before SIGKILL |
| `--cooldown-secs` | `2` | Delay after a run's process exits before the next run starts |
| `--log-dir` | — | Directory for per-run `<variant>-run<N>.log` files |
| `--save-result` | — | Path to write the full result set as JSON |

## Architecture

- `src/main.rs` — Entry point, tokio runtime
- `src/cli.rs` — clap derive CLI args
- `src/lib.rs` — Orchestration: interleaved warmup/timed rounds, summarization, report printing
- `src/runner.rs` — Spawn one command, poll for readiness, tear down the process group
- `src/process_group.rs` — Process-group signal helpers (adapted from `vllm-managed-engine`)
- `src/stats.rs` — Mean/median/min/max/stddev over a variant's samples
- `src/variant.rs` — `NAME=COMMAND` parsing
