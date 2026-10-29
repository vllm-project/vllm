# `perdim` — per-dimension RoPE factors, or: how not to pay the YaRN tax

Serving Qwen3.5 past its native 262 144-token window costs accuracy *everywhere*, not just
past the window. This is an environment-level patch (no weights touched) that recovers most
of that cost. Measured on gfx1100, August 2026.

**It is not YaRN.** It is what replaced YaRN here, after measuring what YaRN actually does.

## The finding

YaRN derives every dimension's scaling factor from a closed formula. Two instruments said
it was free, and both were blind:

- **needle-in-a-haystack** — naive YaRN finds the needle at 300k. No signal.
- **perplexity** — +0.06% over 11 355 tokens, and the sign flipped depending on the source
  corpus. No signal.

What does see it is comparing the **argmax token, position by position**, against the
un-scaled model. At 480k:

| condition | needles | divergence vs native |
|---|---|---|
| identity (all factors 1.0) | 4/5 | 0.097% ← noise floor |
| **`perdim`, cutoff=21 factor=2** | **5/5** | **0.132%** |
| YaRN factor 2 (vLLM built-in) | 5/5 | **1.576%** |

**The YaRN tax is present from the very first token and does not accumulate with length**
— 1.465% over the first 64 positions, the same rate as globally. That is exactly why
perplexity misses it: the changes land on near-ties that barely move the mean logprob.

Touching only the **11 long-wavelength pairs** — the ones whose angle is ≈ 0 at short
positions, so they contribute nothing there — drops the perturbation **12×**, and to
**0.000% over the first 64 tokens**.

⚠️ Read the last row honestly: **identity already scores 4/5.** The patch is not what gets
you to 480k — the model nearly gets there on its own. The patch is what stops you paying a
toll on every short request along the way.

## Why not vLLM's own `longrope`

vLLM already accepts per-dimension factors through `rope_type: "longrope"`. It is unusable
here: that branch ignores `mrope_section`, so it **drops mRoPE and takes the vision tower
with it**. This model is `Qwen3_5ForConditionalGeneration` — the vision tower is not
optional.

So the factors go in through `MRotaryEmbedding` instead, which keeps mRoPE intact.

## How it works

`patch_perdim_rope.py` wraps three things on `MRotaryEmbedding`:

- `_compute_inv_freq` → divides the inverse frequencies element-wise by the factor vector.
  A factor of 1.0 is an exact identity (verified: delta `0.000e+00`).
- `__init__` → registers each instance so they can all be retuned at once.
- `set_perdim_factors` → recomputes the table and writes it back with **`copy_` in place**.
  The pointer never changes, so **captured CUDA graphs stay valid** and you can retune a
  live server.

A daemon thread watches the JSON file and reapplies on change, writing what it actually did
to a status file — so you can tell «applied» from «file written».

```
factor > 1  ⇒  longer wavelength for that pair  ⇒  interpolated
factor = 1  ⇒  untouched
```

With `head_dim=256` and `partial_rotary_factor=0.25` there are **32 frequency pairs**. At
`theta=1e7`, pair ~21 is the first whose wavelength does not complete a full rotation within
256k — those are the under-trained ones, and the only ones worth moving.

## Use it

```bash
export PYTHONPATH=/tmp/vllm_merged:/app        # /app must hold sitecustomize.py
export VLLM_PERDIM_ROPE_FILE=/app/perdim_450k.json
export VLLM_PERDIM_ROPE_STATUS=/app/perdim_status.json
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
vllm serve ... --max-model-len 450000
```

`sitecustomize.py` only fires when `VLLM_PERDIM_ROPE_FILE` is set, and swallows its own
errors — a broken patch must never take the server down with it.

Retune without restarting: write a new file, then read `perdim_status.json` and check
`instancias > 0`. **If it says 0, nothing was applied** — the file being on disk proves
nothing.

### The files here

| file | what |
|---|---|
| `patch_perdim_rope.py` | the patch |
| `sitecustomize.py` | the hook that activates it from `PYTHONPATH` |
| `perdim_450k.json` | production factors for 450k: 1.0 on pairs 0-14, a ramp, then `s = 1.71661` (= 450000/262144) from pair 22 on |

The serving side is just the four environment variables above plus
`--max-model-len 450000`; the rest of the command line is whatever you already serve with
(here: TP2, `int8_per_token_head`, prefix caching on).

⚠️ Do not confuse `perdim_450k.json` (production factors) with `perdim_factors.json` (the
search scratch file, currently sitting at identity).

## Three traps that already cost time here

1. ⚠️ **Prefix caching silently invalidates an A/B.** Retuning RoPE on a live server does
   **not** invalidate the KV cache, so vLLM keeps serving states computed with the old
   factors — 89.6% cache hit rate, and a whole sweep came out identical to the control. It
   was called a result before being checked. Redo it with **a fresh server per candidate**;
   turning prefix caching off is not a workaround, it trips the gfx11 all-reduce bug at
   startup.
2. ⚠️ **`--hf-overrides` for `rope_parameters` must be nested inside `text_config`.** This
   model uses the Transformers v5 layout. Put it at the root and **vLLM ignores it without
   a word** — the server comes up looking fine and scaling nothing.
3. ⚠️ **Zero tax in short context is worthless on its own** — doing nothing also scores
   zero. Any candidate has to be measured *past* the native window. A 256k test proves
   nothing: 256k is *inside* 262 144.

## What was ruled out

**Dual Chunk Attention** — Qwen's own training-free route to 1M. Half-present in vLLM:
`DualChunkRotaryEmbedding` exists (217 lines) and `qwen3_next.py` passes the config, but
`qwen3_5.py` has zero references and — the part that kills it — **no attention backend
consumes it**, neither in the container nor in the tree. It is not a flag; it is writing the
Triton kernel for RDNA3.

## Status

The local 450k `perdim` deployment has been **stopped since
30-aug-2026**. Restarting it is item 6 of `RETOMAR_05SEP_NOCHE.md` in the `w4a16_trainer`
repo.

Neither ainode1 nor ainode2 run `perdim`: they serve 400k on plain
`VLLM_ALLOW_LONG_MAX_MODEL_LEN` with no rescaling at all — i.e. the *identity* row above.
Same for the Mac Studio deployment (see `docs/MAC_Q38V4_400K.md` in `w4a16_trainer`).

⚠️ The image tag `…-yarnprobe` is a **leftover label** from that experiment. There is no
YaRN code in those images. Do not read the tag as documentation.
