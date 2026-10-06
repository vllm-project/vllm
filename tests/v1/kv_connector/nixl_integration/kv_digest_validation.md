# KV digest validation harnesses

Prototype validation for the NIXL pull-mode KV digest (checksum) prototype.
All scenarios run on a single node with 2 GPUs and the toy proxy. Scripts log
to `~/pd1p1d/{p,d,proxy}.log`.

## Coverage matrix

Mapping the key corruption risks (production: a request silently reads
another request's KV, so users see responses unrelated to their prompt) to
the scenario that exercises it:

| Key validation | What it is in production | Scenario | Script | Status |
| --- | --- | --- | --- | --- |
| Baseline (no fault) | Normal operation; digest must never false-positive | A, D | `kv_digest_smoke.sh` stage 1, `kv_digest_concurrency.sh` | Covered |
| Silent transport corruption | Bit/content divergence the transport does not report | B | `kv_digest_smoke.sh` stage 2 | Covered (detector self-test) |
| Failure policy on detection | Mismatch must fail the request, not serve corrupt output | C | `kv_digest_smoke.sh` stage 3 | Covered (`fail` policy; `recompute` untested, Phase 2 backlog) |
| kv read after release | P's lease expires early (heartbeat loss, clock skew) and frees blocks before D reads | E | `kv_digest_race.sh` | Covered |
| kv read after re-assign | The freed block is reallocated and overwritten by another request before D reads | E | `kv_digest_race.sh` | Covered (5/12 victims on 2026-10-05) |
| Partial read via D prefix cache | D holds the prompt prefix locally and reads only the suffix; digest alignment must follow | F | `kv_digest_edge_paths.sh` stage 1 | Covered |
| Chunked prefill | Multi-step prefill must digest once, after the last chunk, over the full block list | G | `kv_digest_edge_paths.sh` stage 2 | Covered |
| kv transfer while RDMA failed mid-transfer | NIC/link failure mid-transfer leaves partially written blocks on D | - | - | Pending: needs cross-node (P/D on different hosts over IB; same-node transfers use NVLink P2P). Reported transport errors already route through `_handle_failed_transfer`; the digest covers the silent partial-write case |

## Prerequisites

- Editable vLLM install with `nixl` (`uv pip install "nixl[cu13]"`), `ninja`
  on `PATH` (FlashInfer TRTLLM JIT needs the executable).
- `Qwen/Qwen3-0.6B` in the HF cache.
- Layout: P on GPU 0 port 8100, D on GPU 1 port 8200, toy proxy on 8192.
- Feature flag on BOTH sides:
  `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"...","kv_connector_extra_config":{"enable_kv_digest":true}}'`

## Scenario A: normal path, no injection

`kv_digest_smoke.sh` stage 1 launches P/D/proxy with digests on and sends
requests.

Expected: all 200, coherent output, zero `KV digest mismatch` in d.log,
zero `omitting remote_block_digests` warnings in p.log. With
`VLLM_LOGGING_LEVEL=DEBUG`, D logs show the digests riding in:

```text
kv_transfer_params={... 'remote_block_digests': [['e45d45d7e4ddb557_af0c969f1359eba9']]}
```

## Scenario B: byte-corruption injection

Script: `kv_digest_smoke.sh` stage 2 (`VLLM_NIXL_DIGEST_CORRUPT=1` on D):
flips one byte of the first received block before verification.

Expected: a per-block mismatch ERROR naming group/block/expected/got, e.g.

```text
ERROR ... KV digest mismatch for request ...: group 0, block 1, expected 8be6448f16d78d1f_..., got 8be6448f16d78dea_...
```

A note on purpose: scenario B does not simulate a realistic fault. A literal
1-bit flip mid-transfer is rare on InfiniBand (link-layer ICRC catches
those), and the realistic corruption classes are semantic, not bit-level:
stale blocks, reassigned blocks (scenario E), partial writes (RDMA failure
mid-transfer), misaddressing bugs. B exists to prove the detection machinery
itself works - that the digest comparison really compares content and really
fires when it diverges. It is a deterministic detector self-test, like a unit
test for the detector: if an injected one-byte difference is not caught, no
other scenario can be trusted.

## Scenario C: failure routing

Script: `kv_digest_smoke.sh` stage 3 - scenario B's fault plus
`VLLM_NIXL_DIGEST_FAIL=1`. The task requirement is fail-closed ("a request
must never use another request's KV"), so detection alone is insufficient:
this stage proves a mismatch kills the request through the existing
`kv_load_failure_policy` path instead of generating from corrupt KV.
Log-only mode (scenario B) still has a role: gray rollout - observe first,
turn on fail once false-positive confidence is established.

Expected: `Failing 1 request(s) due to KV load failure (failure_policy=fail)`
in the D scheduler log, client gets an internal error, and
`Num failed transfers` increments in the KV Transfer metrics. Note the
`recompute` policy is untested (Phase 2 backlog).

Under `VLLM_NIXL_DIGEST_FAIL=1`, verification SKIPS are also fail-closed: an
unsupported layout, a missing producer digest, or a missing per-rank entry
fails the request (an unverified request must not pass in fail-closed mode).
With fail routing off, skips stay silent at DEBUG level.

## Scenario D: concurrency false-positive control

`kv_digest_concurrency.sh`: 240 concurrent unique-prompt requests with
digests on. This is the no-fault baseline under load: interleaved requests,
batched digest computation, many requests finishing in the same step.
Expected: 0 mismatches.

(History: this used to have a second arm with a pathological 0.1s lease. As
executed that arm expired zero leases - a NIXL transfer takes ~2ms, far under
100ms - so it tested exactly the same thing as the control arm. Early
release done right is scenario E's job.)

## Scenario E: read-after-release / read-after-reassign race

`kv_digest_race.sh` forces the real race. Three ingredients, all required:

1. P lease (50ms) much shorter than D's time-to-read (2s, via
   `VLLM_NIXL_DEBUG_RECV_DELAY_MS`) - P releases the blocks first.
2. Tiny P KV pool (`--num-gpu-blocks-override 24`, `--max-model-len 256`) so
   the allocator wraps around in ~12 prefills (~1.2s), before D reads.
3. Direct prefill-only churn against P (bypassing D's delay-throttled
   pipeline) plus `--no-enable-prefix-caching`, so freed victim blocks are
   reallocated and overwritten before D's READ lands.

Expected: some victims caught, e.g. (note BOTH digest halves differ - the
whole block was overwritten by another request, not a byte flip):

```text
ERROR ... KV digest mismatch for request ...: group 0, block 1, expected 2ef4e75e..., got de7cb827...
```

Nondeterminism is inherent: victims whose READ lands before the overwrite
pass legitimately. A run on 2026-10-05 (h200-1) caught 5 of 12 victims.

## Scenario F: partial read via D-side prefix cache

`kv_digest_edge_paths.sh` stage 1. No fault injected. D's local prefix cache
is warmed by a direct-to-D request; a PD request with the same prompt prefix
then makes D read only the missing suffix blocks from P. Digest verification
compares only the trailing digest entries for the blocks actually read.

Expected: 0 mismatches, and d.log shows the partial read as
`num_external_tokens` smaller than the full prompt length.

## Scenario G: chunked prefill

`kv_digest_edge_paths.sh` stage 2. No fault injected. P runs with
`--max-num-batched-tokens 32` so a ~120-token prompt prefills in multiple
steps. Digests must be computed once, after the last chunk, and cover the
full accumulated block list.

Expected: 200 with coherent output, digests present, 0 mismatches; p.log
shows the request's `num_computed_tokens` advancing in 32-token steps.

## Gotchas

- Never `pkill -f "vllm serve"` from a shell whose own command line contains
  that string (it kills the shell). Split kill and launch into separate
  commands.
- `num_gpu_blocks_override` must still fit `--max-model-len`
  (max_len/block_size blocks per request) or the engine refuses to start.
- Run under `salloc` on shared clusters; unprotected GPU processes may be
  reaped by the cluster.
