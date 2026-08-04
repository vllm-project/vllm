# Encryption Serde End-to-End Example

This example demonstrates the `aesgcm` serde: the L2 disk adapter encrypts KV
cache with **AES-GCM** before writing to disk, keyed **per `cache_salt`**, and
decrypts it back on prefetch. It protects KV bytes at rest against anyone who
can read the L2 storage.

## What it does

1. Generates a random **master key** file (16 bytes for AES-128)
2. Starts an `lmcache server` with:
   - **L1**: 20 GB CPU memory cache, LRU eviction
   - **L2**: filesystem (disk) adapter
   - **Serde**: `aesgcm` (AES-128-GCM), key derived per `cache_salt` via
     HKDF-SHA256 from the master key
3. Starts vLLM connected via `LMCacheMPConnector`
4. Sends an inference request **with a `cache_salt`** — KV is computed, written
   to L1, then asynchronously encrypted and stored to L2 disk
5. Calls the lmcache HTTP API to **force-clear L1** (CPU cache)
6. Re-sends the same request — L1 misses, L2 prefetch fires, the encrypted
   bytes are loaded from disk and **decrypted** back into KV, then vLLM resumes
   from cache

## Files

- `run_serde_aesgcm_example.sh` — full end-to-end: `lmcache server` +
  `vllm serve` + real inference, then clear L1 and re-infer to hit the L2 path.

## Quick sanity check (no vLLM required)

The pytest suite exercises the aesgcm transform (round-trip, per-tenant key
isolation, tamper detection) without needing vLLM:

```bash
pytest tests/v1/distributed/serde/test_aesgcm.py -xvs
```

## Requirements

- vLLM installed (`vllm serve` works)
- `lmcache` CLI installed (`lmcache server --help` works)
- 1 GPU (default `CUDA_VISIBLE_DEVICES=0`)

## Run

```bash
./run_serde_aesgcm_example.sh
```

Override defaults via environment variables:

```bash
MODEL="meta-llama/Llama-3.1-8B-Instruct" \
GPU_DEVICE=0 \
CACHE_SALT=tenant-a \
AES_BITS=128 \
./run_serde_aesgcm_example.sh
```

Logs and the generated master key live under
`/tmp/lmcache_serde_aesgcm_example/` (override with `TMP_DIR`).

## What to check

- **Round-trip works.** Step 5's server log shows the prompt served from L2
  (e.g. `Prefetch request completed (L1+L2): 4/4 retained keys (0 L1, 4 L2)`)
  with **no `InvalidTag`**, and the completion shares step 3's cached prefix —
  proving encrypt-on-store + decrypt-on-load work on the real MP path. The L2
  hit plus a clean decrypt is the real proof: AES-GCM is authenticated, so a
  byte-inexact decrypt cannot pass the tag check. (The two completions may
  diverge in their freshly generated tail — with async scheduling and
  floating-point non-associativity, a cached-KV forward pass vs. a full
  recompute can flip a token under greedy decoding. That is normal vLLM
  KV-reuse behavior, independent of the serde.)
- **Data is encrypted at rest.** The files under the disk path begin with the
  `0x01` frame version byte followed by a 12-byte IV, not raw KV (step 3 prints
  the first byte).

## Fail-safe check (optional): tamper / wrong key → miss, not corruption

Corrupt a stored object or restart the server pointed at a **different** master
key, then repeat the clear-L1 + re-request steps:

```bash
# after step 3 has stored objects:
truncate -s -1 "$(find /tmp/lmcache_serde_aesgcm_example/disk -type f | head -1)"
curl -s -X POST localhost:8080/cache/clear
# re-send the same request
```

Decryption fails the AES-GCM tag check (`InvalidTag`), the load is treated as a
cache miss, and vLLM **recomputes** — correct output, no crash or garbage KV.

## L2 adapter config syntax

The serde is attached per-adapter via a `serde` sub-dict in the `--l2-adapter`
JSON:

```json
{
  "type": "fs",
  "base_path": "/tmp/lmcache_serde_disk",
  "serde": {
    "type": "aesgcm",
    "key_provider": "hkdf",
    "master_key_path": "/etc/lmcache/keys/master",
    "aes_bits": 128
  }
}
```

Swap the `fs` block for `{"type": "s3", "bucket": "...", ...}` to encrypt an
S3-backed L2. A request with no `cache_salt` still works (encrypted under the
empty-salt key). Provide the master key via a file — e.g. a mounted Kubernetes
`Secret`; it is read once at startup and never written to L2.

## Notes

- **Scope:** encrypts the **L2 tier only** — L1 (host RAM) and L0 (GPU) hold
  plaintext, so this protects readers of the remote storage, not a party with
  access to a running server. See
  [`docs/design/v1/distributed/serde/aesgcm.md`](../../../docs/design/v1/distributed/serde/aesgcm.md).
- **Trust model:** the `hkdf` provider derives every tenant's key from one
  shared master key (any server with the master can decrypt any tenant). It is
  not per-tenant access isolation.
