# Patches running in production — what is mounted where, and what is missing

Four machines serve this fork, and **none of them runs the same set**. A patch is invisible
by construction: a file mounted over vLLM produces no log line, no warning, and no error
when it is absent. This page is the only place that difference is written down.

Measured 9-sep-2026. **Re-measure after any image change** — see the last section.

## The four machines

| | ainode1 | ainode2 | local `vllm-vllm1-1` | Mac Studio `.76` |
|---|---|---|---|---|
| how | Docker, `:ro` volume mounts | Docker, `:ro` volume mounts | Docker, no patch volume | venv, files copied in place |
| image / venv | `…01_09_26_mtp_segint8-yarnprobe` | `…01_09_26_mtp_segint8-yarnprobe` | `…29_07_26_mtp_segsint8_2` | `~/.venv-vllm-metal` + `vllm_metal` |
| patches | **5** | **8** | **0** | **3** |

## What each one carries

| patch | destination | ainode1 | ainode2 | local | Mac |
|---|---|---|---|---|---|
| `qwen3.py` | `vllm/parser/qwen3.py` | ✅ | ✅ | ⛔ | ✅ |
| `utils.py` | `vllm/v1/worker/utils.py` | ✅ | ✅ | ⛔ | ⛔ |
| `single_type_kv_cache_manager.py` | `vllm/v1/core/…` | ✅ | ✅ | ⛔ | ⛔ |
| `triton_attn.py` | `vllm/v1/attention/backends/…` | ✅ | ✅ | ⛔ | ⛔ |
| `triton_per_token_head_attention.py` | `vllm/v1/attention/ops/…` | ✅ | ✅ | ⛔ | ⛔ |
| `sampler.py` (zero-temp tripwire) | `vllm/v1/sample/sampler.py` | ⛔ | ✅ | ⛔ | ⛔ |
| `routed_experts.py` | `vllm/model_executor/layers/fused_moe/…` | ⛔ | ✅ | ⛔ | ⛔ |
| `qwen2_moe.py` | `vllm/model_executor/models/…` | ⛔ | ✅ | ⛔ | ⛔ |
| `MITIGACION` in `align.py` | `vllm_metal/attention/state/…` | — | — | — | ✅ |
| `GUARDA` in `topk_topp_sampler.py` | `vllm/v1/sample/ops/…` | — | — | — | ✅ |

Sources: `/docker/vllm_router/docker-compose-4.yml` (ainode1),
`/docker/vllm/docker-compose.4gpu.yml` (ainode2), `docker inspect` (local), and md5 on the
Mac. The last two rows are Metal-plugin-specific and have no counterpart on ROCm.

## The three that bite

### `qwen3.py` — the parser that ate your arguments

md5 `80df0af4185631e4f913995e7c213511` when patched, `e5b15950ac9f25686f77ae0c7810f4c5`
when not.

An unterminated `<parameter=` kept reading past the end of the value. Measured over 64
rejected tool-calls from the router log, two distinct damages:

- the value read straight past `</think>` and into the *next* tool call:
  `{"path": "/tmp/probe.py\n</think>\n\n<tool_call>\n<function=edit>"}`;
- if the **last** parameter closed with `</function>` instead of `</parameter>` — which the
  template's own example invites — that parameter **vanished, with no error**. The call
  arrived one required argument short.

The fix adds `</think>`, `</function>` and `<tool_call>`/`</tool_call>` as extra terminators,
as lookaheads so they are not consumed.

⛔ **The local container does not have this.** It runs the July image with no patch volume
at all. Anything reproduced there against tool-calling is reproducing a *different* parser.

### `sampler.py` — the zero-temperature tripwire (ainode2 only)

Fixes the `!!!!` output. The divide-by-zero guard was **conditional** — it only ran when the
batch contained a greedy request. A 0.0 carried over from an earlier step landing in an
all-random batch divided by zero → NaN → `argmax` returns index 0 → token id 0 → `!`.

The patch makes it unconditional and counts every trip. If
`[TRIPWIRE temp-cero] N filas …` shows up in the log, the condition is **real in live
traffic** and the patch is neutralising it.

⚠️ **ainode1 does not mount it.** The upstream fix is in this branch as `0aa0085946`, so its
image may well carry it already — but that has not been verified on ainode1, and «probably
in the image» is not the same as «measured present».

### `align.py` — the GDN state manager that killed the whole engine (Mac)

A request can be scheduled with **zero tokens to compute** — for instance when the prompt is
100% covered by the prefix cache. The Metal plugin raised on that, which **killed the
EngineCore and took every session with it**.

The mitigation appends the current block instead of raising, keeping `dst_ids` positionally
aligned (`group_mappings` indexes it per request).

⚠️ This matters more now than when it was written: the Mac serves with
`--enable-prefix-caching`, which is exactly what produces 100%-covered prompts.

## ⚠️ Changing the image silently reverts everything

A mount puts *this* file on top of whatever the new image ships. If vLLM has moved on — and
these are hot files — you are serving an old patch over a new vLLM, **with nothing to warn
you**: no error, just odd behaviour weeks later.

Recipe, with the new image running and the mount removed:

```bash
docker exec vllm-vllm1-1 python3 /app/parche_tripwire.py
docker exec vllm-vllm1-1 cat /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/sampler.py \
  > /docker/vllm/patches/sampler.py
```

And to check a mounted file still matches its image (they should differ **only** in the
patch block):

```bash
docker run --rm --entrypoint cat <IMAGE> \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/sampler.py > /tmp/img.py
diff /tmp/img.py /docker/vllm/patches/sampler.py
```

On the Mac there are no mounts — the files were copied into the venv, so **`pip install`
overwrites them**. Backups sit next to each one (`*.bak.sinparche`, `*.orig`).

## Related

- `tools/rdna3/perdim_rope/` — per-dimension RoPE factors, the answer to «the YaRN patch».
- `tools/rdna3/INT8_GDN_PAGEFAULT.md` — the int8/GDN page fault investigation.

---

## 21-sep-2026 — what the two serving boxes actually run now

The table above is from 9-sep and is about *mounted files*. The two biggest wins since then
are **not patches at all**: one is a kernel, the other is two command-line flags.

| | ainode1 (production) | ainode2 (lab) |
|---|---|---|
| GQA decode kernel (`VLLM_RDNA3_GQA_DECODE=1`) | ✅ | ✅ |
| `--prefix-match-unit 16` | ✅ | ✅ |
| `--enable-mamba-fine-grained-prefix-cache` | ✅ | ✅ |
| MTP `k=3` + `--prefix-cache-retention-interval 1584` | ✅ | ✅ |
| `VLLM_RDNA3_CUSTOM_AR_CAP` | ⛔ | 49152 |
| `VLLM_RDNA3_TINY_GEMV` | ⛔ | ⛔ **retired** |
| rocBLAS prefill override | ⛔ | ⛔ **retired** |

### The two flags are the largest single win of the line

They cut the prefill of every follow-up turn by **73×** — not because of any kernel, but
because the prefix-cache hit granularity was **1.584 tokens**: the attention block is
inflated to cover the GDN state page (`interface.py:933`), and with MTP a whole block is
dropped from the tail to protect against **3** lookahead tokens
(`kv_cache_coordinator.py:107`). Measured on ainode1:

    new tokens per turn        1.645-2.778  ->  37
    TTFT of a follow-up turn   1,24 s       ->  0,32 s
    a repeated 163k prompt     ~113 s       ->  0,53 s
    short decode               110-113      ->  113-116 tok/s
    163,7k decode              110-117      ->  117-119 tok/s

⚠️ `--prefix-match-unit` **alone does nothing** — without
`--enable-mamba-fine-grained-prefix-cache` there is no GDN state checkpoint at the fine
boundary. It had been written off as "inert" in an earlier attempt for exactly that reason,
and sat commented out in both composes. ⚠️ And note the **int8 KV cache doubles the
granularity**: fewer bytes per token means more tokens are needed to match the GDN page.

### Two patches were retired the same night

`tiny_gemv` and the rocBLAS prefill override measure well in a single-GPU microbenchmark
(3,7× / 2,0×, and TTFT −5,9%) and cost **−22/27% of decode under concurrency** at TP4, plus
they pin one GPU at 100% and 120 W **with the server idle**. Their headers carry the
numbers. Same trap as always here: a microbenchmark in eager does not predict the cost
inside `torch.compile` at TP4 — measure the **step** with 4 and 6 sessions, never a kernel
on its own.

Full write-up, with the bisection that separates each piece: `rdna3_p2p/14` §25-27.
