# `lmcache query` CLI Command Design

**Status:** Proposal  |  **Date:** 2026-03-20

## Goal
Provide a formal single-shot query interface for both the serving engine and KV cache worker, with metrics output. besides normal request query to serving engine, offers the feature to query the detailed KV cache info by the request prompt.
 

---

## Design Principles

### Single-shot, metrics-first command

`lmcache query` performs exactly one request and reports latency + result metrics
through the shared metrics framework (`BaseCommand.create_metrics()`), so users
can choose `--format terminal` or `--format json`.

### Two targets with one verb

`query` has two second-level targets:

- `query engine`: run one inference request and measure TTFT/TPOT/throughput.
- `query kvcache`: inspect cache coverage for one prompt (lookup).
 
### Script-friendly output and behavior

- `--format json` produces machine-readable metrics.
- `--output` writes the same formatted result to file.
- Exit codes: `0` success, `1` error.
- Errors go to stderr, metrics go to stdout.

### Prompt corpora support

Both subcommands accept prompt templates like `{ffmpeg}` and `{paul_graham}`,
using the shared corpora expansion mechanism described in `commands.md`.

---

## Command Overview

```text
lmcache query
├── engine    # Single inference query with latency/token metrics
└── kvcache   # Single request cache lookup or round-trip verification
```

```bash
$ lmcache query -h
usage: lmcache query [-h] {engine,kvcache} ...

Run one query and report metrics.

subcommands:
  engine      Run one inference request and report TTFT/TPOT metrics
  kvcache     Query KV cache coverage or run store-retrieve round-trip
```

---

## Commands in Detail

### `query engine`

Send one inference request to an engine HTTP endpoint and report token/latency metrics; ``--prompt`` supports placeholders, where ``{lmcache}`` loads ``lmcache/cli/documents/lmcache.txt`` and custom documents use ``--documents NAME=PATH``.
 

```bash
# Single inference query
$ lmcache query engine --url http://localhost:8000/v1 \
     --prompt "{lmcache} Summarize LMCache usage." \
     --format terminal \
     --max-tokens 128
   
================= Query Engine =================
Model:                         facebook/opt-125m
Prompt documents lmcache:                    608
Prompt query:                                  9
--------------- Latency Metrics ----------------
Input tokens:                             618.00
Output tokens:                              9.00
TTFT (ms):                                 26.88
TPOT (ms/token):                            0.91
Total latency (ms):                        35.05
Throughput (tokens/s):                   1100.64
================================================
```

#### Proposed flags besides native engine query flags

| Flag | Description |
|------|-------------|
| `--url` | Engine HTTP endpoint (`http://host:port`) |
| `--prompt` | Prompt text, supports `{documents}` templates |
| `--timeout` | Request timeout in seconds (default: 30) |
| `--documents name=path` | Register custom documents template |



#### Output metrics
 
- `prompt_tokens`, `output_tokens`, `model`
- `ttft_ms`, `tpot_ms_per_token`, `total_latency_ms`, `throughput_tokens_per_s`

 

### `query kvcache`

Two modes under one command:

1. **Lookup mode (default):** tokenize prompt and query cache coverage.
 
```bash
# Lookup mode
$ lmcache query kvcache --url http://localhost:5555 \
    --prompt "{ctx} What is the example usage of lmcache?" \
    --documents ctx=LMCache/lmcache/cli/documents/lmcache.txt  \
    --model meta-llama/Llama-3.1-8B-Instruct

======== Query KV Cache Result ==========
Prompt tokens:                           8192
Cached chunks:                       30/32 (93.8%)
Cache locations:               [cpu=12, disk=0, ...]
Cached tokens:                         7680/8192
Cache status:                       HIT (partial)
=========================================
```
 

#### Proposed flags

| Flag | Description |
|------|-------------|
| `--url` | KV cache HTTP endpoint (`http://host:port`) |
| `--prompt` | Prompt for tokenization + lookup |
| `--model` | Tokenizer/model used to derive token IDs |
| `--documents name=path` | Register custom documents template |

#### Output metrics (lookup mode)

- `prompt_tokens`
- `cached_chunks_hit`
- `cached_chunks_total`
- `cached_chunk_location`
- `cached_tokens_hit`
- `cached_tokens_total`
- `cache_status` (`HIT`, `MISS`, `HIT (partial)`)


---

## API Surface and Dependencies

### `query engine`

Uses inference engine HTTP APIs (OpenAI-compatible or engine-native endpoint),
then computes CLI-side metrics from the single response stream/non-stream result.

No new dependencies required: use stdlib `urllib.request` and existing helpers.

### `query kvcache`

All `lmcache query kvcache` CLI operations go through HTTP, using either the
per-instance HTTP server or the controller HTTP server.
 

---

## Implementation

- **Single `QueryCommand`** (`BaseCommand` subclass) with second-level
  subparsers (`engine`, `kvcache`) in `lmcache/cli/commands/query.py`.
- **`query engine`:** `PromptBuilder` (`lmcache/cli/prompt.py`) expands `{name}`
  placeholders from `--documents`; top-level metrics include model plus per-slot
  token estimates (e.g. prompt documents, prompt query). `Request`
  (`lmcache/cli/request.py`) streams an OpenAI-compatible `/v1/chat/completions`
  or `/v1/completions` request; **Latency Metrics** repeats server usage (labeled
  **Input tokens**, not a duplicate client-side total).
- **`query kvcache`:** stub; no handler yet.
- **Errors:** `query_engine` catches `RuntimeError` / `ValueError`, prints the
  message to stderr, exits `1`; unknown `query_target` prints to stderr and exits
  `1`.

---

## Phasing

| Phase | Work |
|-------|------|
| **1a** | `query engine` with prompt, max-tokens, TTFT/TPOT/throughput metrics |
| **1b** | `query kvcache` lookup mode (prompt tokenization + cache coverage) |
| **future** | richer query diagnostics (per-chunk detail) |

