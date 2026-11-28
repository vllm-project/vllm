# Writing a ROCm Decode Megakernel — a top‑down guide, with a Gemma‑4‑31B FP8 worked example

> **Audience:** a kernel/systems engineer who wants to implement a *decode megakernel*
> ("monokernel") for a dense transformer on AMD Instinct (MI300X / MI355X, CDNA3/CDNA4)
> inside vLLM, the same way the AMD team did it for DeepSeek‑V4.1, Kimi‑K3, GLM‑5.2 and
> Qwen3.8‑27B.
>
> **How to read it:** Part 1 is the *mental model* and the top‑level architecture. Part 2
> is *what to fuse* for Gemma. Part 3 is the *toolchain / hardware contract*. Part 4 is the
> *step‑by‑step build*. Part 5 is *test & benchmark*. Part 6 is a *task breakdown you can
> execute top‑to‑bottom*. Part 7 is *pitfalls*. The appendices are a reference file map, a PR
> index, and external links.
>
> Everything here is grounded in real, in‑flight vLLM work. The canonical references are the
> **merged** DeepSeek‑V4.1 mono decode (PR #60397, #61082), the **dense** Qwen3.8‑27B mono
> decode (PR #61573 — the closest template to a dense Gemma), the Kimi‑K3 monokernel
> (PR #60683, #61435), the GLM‑5.2 monokernel (PR #60889), the NVIDIA CuTeDSL specialized
> decode that started the pattern (PR #61867), and the shared‑abstractions RFC (#60904) /
> PR (#61311). See [Appendix B](#appendix-b--pr-index) for the full index.

---

## Part 0 — TL;DR

A **decode megakernel** replaces the ~40–600 tiny GPU kernels that vLLM launches per decoder
layer at *decode time* with **one or two persistent kernel launches per layer**. All compute
units (CUs) are launched once, stay resident, and hand off to each other through
**global‑memory mailboxes** instead of returning to the host between ops. On a single‑token
decode step this removes launch latency, keeps weights/activations resident, and hides HBM
latency — real measured wins are **−5 % to −27 % TPOT / +9 % to +34 % output tok/s** at low
concurrency (Qwen3.8‑27B dense, Kimi‑K3, GLM‑5.2 numbers below).

The trick that makes it *tractable* and *reviewable*:

1. It only runs on a **narrow, opt‑in fast path** — pure decode, small row count (≤ 8–64),
   FULL CUDA graph, specific TP, specific GPU arch, specific quant. Everything else falls
   back to stock vLLM unchanged.
2. Each launch is **bit‑exact (or cosine‑bounded) against the stock vLLM ops** it replaces,
   stage by stage, so it is testable without a model eval.
3. It is written in **FlyDSL** (AMD's Python‑embedded GPU DSL over MLIR/ROCDL), not raw HIP,
   so one file both *generates the kernel* and *plans the host launch*.
4. It plugs into vLLM through a **thin, model‑agnostic abstraction** (`SpecializedDecode` /
   `MonoKernel`), so the model file barely changes.

For a **dense** model like Gemma‑4‑31B you follow the **Qwen3.8‑27B dense** template almost
exactly: **K1** = the attention block, **K2** = the FFN (MLP). Gemma adds *sandwich norms* and
*QK‑norm* and *interleaved sliding/global attention*, which change *what* K1/K2 compute but not
*how* the megakernel is structured.

---

## PART 1 — TOP‑LEVEL ARCHITECTURE

### 1.1 Why a decode megakernel exists

At decode you process **one new token per request** (plus a few speculative draft tokens). A
27–31 B model is ~60 layers; each layer is a chain of small ops (norm, q/k/v GEMV, qk‑norm,
RoPE, attention, o_proj, norm, gate/up GEMV, activation, down GEMV, norm, all‑reduces). In the
stock path **every one of these is its own kernel launch**. At batch‑1 the GEMVs are tiny and
*memory‑bound*, so the dominant cost is not math — it is:

- **Launch overhead / bubbles**: between kernels the memory pipeline stalls; no weight load is
  in flight during a kernel's prologue/epilogue. The Stanford *"Look Ma, No Bubbles"* megakernel
  work measured stock vLLM/SGLang at ~50 % of peak HBM bandwidth vs ~78 % for a megakernel.
- **HBM round‑trips**: each op writes its output to HBM and the next reads it back, even though
  on a persistent grid it could stay in LDS/registers.
- **Tail effects**: a tiny kernel can't fill the GPU; CUs idle.

A **megakernel** launches the grid once (one CTA per CU, all co‑resident), keeps the whole
layer on‑chip, overlaps the next op's weight loads with the current op's compute, and
synchronizes CUs through global‑memory flags — never returning to the host mid‑layer.

> **Key references** (full links in Appendix C): Stanford Hazy Research *"Look Ma, No Bubbles"*
> (megakernel for Llama‑1B, grid‑persistent + on‑GPU instruction interpreter + global‑counter
> sync + software pipelining) and its TP follow‑up; Mirage Persistent Kernel (MPK), an
> automatic megakernel compiler. The AMD vLLM "mono decode" family is the production, FlyDSL,
> hand‑planned realization of the same idea, specialized per model.

### 1.2 The AMD "mono decode" family (what you're joining)

| Model | Kind | PR | State | Launches | Headline result |
| --- | --- | --- | --- | --- | --- |
| DeepSeek‑V4.1‑Flash | MoE | #60397, #61082 | **merged** | K1 + K2 per layer (FFN‑only launch on index/compressor layers) | +3.9–4.5 % P90/P50 interactivity, accuracy neutral |
| Qwen3.8‑27B | **dense** | #61573 | open | K1 (GDN mixer) + K2 (MXFP4 FFN) | +34 % / +21 % output tok/s @ TP1 conc 4/8 |
| Kimi‑K3 | MoE | #60683, #61435 | open | K1 (KDA attn) + K2 (o_proj→latent MoE) | −22.9 % TPOT, +27.3 % tok/s |
| GLM‑5.2 | MoE+DSA | #60889 | open | 1 persistent kernel per MoE layer | 2.3× TPOT @ batch1, 1.5× @ batch8 |
| MiniMax‑M3, Qwen3.8‑Flash‑Next, GLM‑5.3‑Flash, DSv4‑Pro | various | many | draft | — | — |

The **abstractions** these all share are being standardized by RFC **#60904** and two competing
implementations — **#61311** (`vllm/models/common/specialized_decode/`) and **#60683/#61435**'s
`vllm/models/common/mono/`. You target one of these; this guide uses both so you can pick.

> **For Gemma‑4 you clone the Qwen3.8‑27B dense path (#61573).** It is the only *dense* member,
> and its K2 already contains a **Gemma‑style RMSNorm** (the `dense_ffn.py` header literally
> says *"post_attention_layernorm (Gemma, 1 + w)"*). Gemma's *attention* (full MHA + QK‑norm +
> sliding window) differs from Qwen's GDN, so K1 is where your real new work is.

### 1.3 The execution model (memorize this)

Every mono kernel is written for one fixed execution model. From the merged DSv4.1 README and
`common/plan.py`:

```text
BLOCKS   = 256          # one CTA per CU of an MI355X; ALL co-resident
THREADS  = 512          # 8 wavefronts (wavefront = 64 lanes on CDNA)
WAVES    = 8
LDS_BYTES = 160 * 1024  # a whole CU's LDS (CDNA4). CDNA3 (gfx942) has 64 KiB -> smaller tiles
```

- **Persistent, co‑resident grid.** Because a spin‑wait on another CTA only makes progress if
  that CTA is *running*, the runner **refuses a GPU whose CU count != BLOCKS** (a partitioned
  GPU). This is why mono kernels are arch‑pinned (gfx950 first; gfx942 ports are separate PRs,
  e.g. #60731, #60997).
- **A kernel is a sequence of *stages*; a stage is a *task table*.** CTA `b` runs tasks
  `b, b+256, b+512, …`, and the stages execute in order. A consumer stage only ever waits on
  **earlier** stages, so there is **no deadlock while the grid is resident**. (`first_task(bid,
  base)` in `plan.py` computes a CTA's first task of a stage placed from an arbitrary base CTA.)
- **Software pipelining is explicit.** A GEMV task **issues its weight loads before it polls its
  activations** (`batched_rounds` in `common/ops.py` issues `LOAD_BATCH=4` rounds of loads before
  any use; weights a layer reads once are streamed `CM_NT` nontemporal so they don't evict reused
  data).
- **Hand‑offs are tagged mailboxes, not barriers.** See §1.4.

### 1.4 Synchronization: the tagged mailbox + peer memory

This is the heart of the design (`common/sync.py`, `common/ops.py`).

**Within one GPU (device scope).** Every 32‑bit value is stored next to a **tag** in one 8‑byte
`(value, tag)` pair. A consumer *polls the payload until every tag matches the expected tag*.
That means a hand‑off costs **one round trip** — no separate "data ready" flag, no store‑drain,
no clearing between steps.

```python
# common/sync.py  (Mailbox)
class Mailbox:
    def __init__(self, layer, ...):
        self.tag = layer + 1          # distinguishes the layers of one step
    def put(self, addr, i, v):        # store (bits(v), tag) as one i32x2 vector
        ...
    def poll(self, specs, batch=12):  # re-load all pairs while ANY tag is stale
        ...                           # returns payloads once all tags land
```

- The **tag** encodes the step/epoch and the launch slot (`helpers.step_tag(epoch, slot)` →
  `(epoch % 2^22) * 256 + slot + 1`). The runner zeroes mailboxes at a step start
  (`begin_step`) so a stale pair from an earlier step can never carry a live tag. **Every
  argument is a fixed device pointer** → the whole launch is **CUDA‑graph safe**.
- `publish()` is the "make my plain stores visible, then raise the flag" primitive: it issues
  `s_waitcnt(vmcnt=0)` + a workgroup barrier, *then* one thread writes the mailbox pair, so a
  flag never overtakes the data it announces.
- Cache scope is chosen per hand‑off via CPol bits: `CM_DEV=16` (SC1, device scope — past the
  per‑XCD L2) for CTA↔CTA, `CM_SYS=17` (SC0|SC1, system scope — over XGMI) for GPU↔GPU.

**Across GPUs (system scope) — the in‑kernel TP all‑reduce.** Each rank owns one **symmetric,
uncached peer buffer** (AITER `UncachedIpcHeap`); every rank holds every peer's base address.
A "push" is a **system‑scope store of `(value, tag)` into every peer's region**. Because every
rank runs the *same* launches, their epochs agree; peer regions alternate by epoch parity, and
a rank cannot run two steps ahead of a peer (each step's reduce waits on every rank's push).
The reduce is **exact fp32 in rank order → bf16**, matching vLLM's own small‑batch all‑reduce
(which, below its INT4 quick‑reduce threshold, is also exact).

> This is why the mono path can fold the tensor‑parallel all‑reduce *into* the kernel at TP1/
> small M — a huge part of the win at TP≥2. (At TP2 the Qwen3.8‑27B dense path *keeps* the
> all‑reduce on the stock collective instead; see §4.8.)

### 1.5 The two‑launch split: K1 and K2

Dense layer = **attention block** then **FFN block**. The mono split mirrors that:

```text
            ┌───────────────────────── one decoder layer at decode ─────────────────────────┐
  residual ─┤  K1: input_norm → Wqkv → q/k norm → RoPE → attention → (o_proj)   →  core    ├─► K2 …
            │  K2: (o_proj→AR)→ post_attn_norm → pre_mlp_norm → gate_up → act → down → norm  │
            └───────────────────────────────────────────────────────────────────────────────┘
```

- **K1** = everything from the residual up to the attention output. For Gemma that is:
  `fused_add_rms_norm(input)` → `Wqkv` GEMV → **q_norm/k_norm** (per‑head RMSNorm) → **RoPE**
  (θ depends on local/global layer) → **attention** (sliding‑window for local layers, full for
  global) → `o_proj`.
- **K2** = the FFN: the attention all‑reduce (if folded), `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `gate_up` GEMV, GeGLU activation, `down` GEMV,
  `post_feedforward_layernorm`, and the FFN all‑reduce.

You can also stack variants (Kimi‑K3 #61435 adds "M1" MLA prologue/epilogue launches). Start
with **just K2** (the FFN — it's bit‑exact and self‑contained), then add **K1**. This is exactly
how the dense PR is structured (`VLLM_ROCM_QWEN38_DENSE_MONO_K2` default on; K1 optional).

#### Why two launches, not one

Nothing stops you from writing a single persistent launch for the whole layer — a resident stage
can wait on an earlier stage through the mailbox (§1.4), so dependency alone does not force a
kernel boundary. The split exists for three concrete reasons:

1. **It cuts the layer at the tensor‑parallel all‑reduce boundaries.** A layer has two cross‑rank
   reductions — after `o_proj` (attention output) and after `down_proj` (FFN output). A reduce
   needs *every* rank to have finished producing its partial, so it is the one place every rank
   must line up. Ending K1 at the attention output and letting K2 open with the attention
   all‑reduce (`K2: (o_proj→AR)→ …` in the diagram above) puts each launch boundary exactly on a
   rank‑synchronization point, which keeps the in‑kernel peer all‑reduce (or the stock collective
   at TP2) clean and graph‑safe. *Example:* DSv4.1 (#60397) ends K1 right after the attention
   front and has K2 do "the attention back, the attention TP all‑reduce, the FFN seam, the MoE and
   its TP all‑reduce" — i.e. K2 **starts** by consuming every rank's pushed `wo_b` partial.
2. **It makes K2 reusable on layers that keep stock attention.** Because K2 does not care *how* its
   input was produced, the very same K2 launch serves both a fully‑fused layer **and** a layer
   whose attention ran on the stock path. A single monolithic launch could not be reused that way.
   *Examples:* DSv4.1's "FFN‑only launch" runs stock attention on its compressor/index layers and
   still uses K2 for the FFN (#60397); #61568 adds "K2 in plain decode" so KDA layers run vLLM's
   attention up to `o_proj`'s input, then K2; and the guide's own recommendation for Gemma is to
   fuse local layers as K1+K2 but run **stock attention + your K2** on the global layers (§4.7).
3. **It bounds each launch's working set and keeps its stage graph simple.** One kernel doing the
   norms, QKV, attention, and the whole FFN together would need far more live registers and LDS,
   hurting occupancy — and recall a mono kernel that spills to scratch is *refused* (Pitfall #2).
   Two smaller launches keep register/LDS pressure down and let each kernel's stages form a simple
   "only wait on earlier stages" DAG. *Example:* Kimi‑K3 #61435 goes the opposite direction
   deliberately — it *adds* launches ("M1" MLA prologue/epilogue) on top of K1/K2 rather than
   growing one kernel, because more, smaller launches were easier to build and schedule than one
   giant one.

So the rule of thumb: **put a launch boundary wherever a tensor‑parallel all‑reduce happens, and
wherever you want the option to fall back to stock for one block** (attention) while keeping the
fused version of the other (FFN). For a dense model that lands naturally on two: K1 = attention,
K2 = FFN.

### 1.6 Where it plugs into vLLM

Two integration surfaces exist today; both are thin by design (RFC #60904 goal: *"the author
should not have to write another way to glue the kernel in"*).

**(A) `SpecializedDecode` (PR #61311, #61867, used by GLM‑5.2 #60889).** The model declares a
`DecodeSpec` describing *when* it may run and *how* to build it; the library does validation,
JIT warm‑up, capture‑size amendment, per‑step dispatch, and TP‑rank agreement.

```python
# vllm/models/common/specialized_decode/spec.py
@dataclass(frozen=True)
class DecodeSpec(Generic[KernelT]):
    name: str
    supports: frozenset[Feature]                 # default-deny: a feature in use & absent -> refuse
    widths: Callable[[VllmConfig], tuple[int,...]]# the row counts you build -> CUDA graph capture sizes
    refuse_config: Callable[[VllmConfig], list[str]]
    refuse_model:  Callable[[Any], list[str]]
    refuse_caches: Callable[[VllmConfig, KernelT, Mapping], list[str]]
    skip_step:     Callable[[VllmConfig, KernelT, int], str | None]  # per-step fallback reason
    build:         Callable[[VllmConfig, Any, tuple[int,...]], KernelT]
```

Enable: `--additional-config '{"decode_backend": "latency_specialized"}'`. Dispatch
(`dispatch.py::SpecializedDecode.plan`) falls back to stock when: width not built, no forward
context, **PIECEWISE** graph mode, caches unbound, or the spec's `skip_step` vetoes.

**(B) `MonoKernel` (PR #60683/#61435, `vllm/models/common/mono/`).** The model declares
`SupportsMonoKernel` with `mono_spec="pkg.module:MONO_SPEC"` and calls only `attach`,
`forward_layer`, `health`. Enable: `--monokernel` (`KernelConfig.monokernel`, per RFC #60904)
or the legacy env `VLLM_ROCM_MONO_DECODE=1`. Adds fail‑stop safety and Prometheus metrics
(`vllm:monokernel_steps`, `vllm:monokernel_incidents`).

**(C) The lowest‑friction path — custom‑op + a separate compiled graph (Qwen3.8‑27B dense, #61573).** No framework at all: the launches are `direct_register_custom_op` ops, and a second
`@support_torch_compile` module (`DenseMonoGraph`) is the branch‑free decode entry point that
the uncompiled outer forward calls only for an eligible width. This is the simplest to stand up
and is what the rest of this guide builds on.

#### The non‑negotiable gates (true for all three)

A step runs on the megakernel **only if ALL hold** (`guards.py`, `dense_mono.py::_unsupported`):

- Pure **decode** (`num_prefills == 0`, ≥1 decode row, no spec‑mask unless you support it).
- **Width** (row count) ∈ the built widths (e.g. 1..8 at TP1, 1..16 at TP2).
- **FULL** CUDA graph (or eager for debugging) — never PIECEWISE.
- **Model Runner V2**.
- Right **GPU arch** (gfx950), **TP** (1 or 2), **CU count == BLOCKS**.
- No unsupported feature: PP, DP, EP, EPLB, DCP, ubatching, LoRA, KV connector, sleep mode,
  CPU offload, weight updates, sequence parallel, speculative (unless supported).
- Exact **shape + quant** match (hidden/intermediate sizes, block‑FP8 `FP8_BLOCK` scheme, norm variant).

If the flag is set but the deployment can't run it, you **raise at startup and list every
reason** (never silently fall back at init). Per‑*step* ineligibility *does* silently fall back.

---

## PART 2 — WHAT TO FUSE FOR GEMMA‑4 (THE COMPUTE CONTRACT)

### 2.1 The Gemma decoder layer (decode shapes, B rows, seqlen 1)

Gemma‑4‑31B uses **sandwich norms** (pre‑ and post‑norm around *both* attention and MLP),
**QK‑norm** (per‑head RMSNorm on q and k), **interleaved local(sliding)/global attention**, and a
**final logit soft‑cap of 30.0** at the LM head — which sits *outside* the decoder layer, so it
never enters K1/K2. There is no *attention* soft‑cap. Verified against the target checkpoint
[`RedHatAI/gemma-4-31B-it-FP8-block`](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-block)
`config.json` (`model_type: gemma4_text`, `architectures: ["Gemma4ForConditionalGeneration"]` — a
vision+audio multimodal model; this guide targets the **text backbone's** decode only).

```text
x [B, H]  (H = hidden_size)
 ├─ residual = x
 ├─ x = input_layernorm(x)                         # GemmaRMSNorm, (1+w), eps=1e-6
 ├─ qkv = Wqkv(x)                                  # FP8 GEMV  -> q[B,nH,hd] k[B,nKV,hd] v[B,nKV,hd]
 ├─ q = q_norm(q)  ; k = k_norm(k)                 # per-head RMSNorm over hd  (QK-norm)
 ├─ q,k = RoPE(positions, q, k)                    # local: θ=10k, full rotary; global: θ=1M, partial_rotary 0.25
 ├─ a = attention(q,k,v, scale=head_dim**-0.5,      # local: hd=256,nKV=16 ; global: hd=512,nKV=4 (k_eq_v)
 │                 window = 1024 if local else ∞)   # no ATTENTION soft-cap (final logit cap 30.0 is at the LM head)
 ├─ a = o_proj(a)                                  # FP8 GEMV  (+ TP all-reduce)
 ├─ a = post_attention_layernorm(a)                # sandwich: norm AFTER attn, BEFORE residual add
 ├─ x = residual + a ; residual = x
 ├─ x = pre_feedforward_layernorm(x)
 ├─ g,u = gate_up_proj(x)                          # FP8 GEMV, MergedColumnParallel
 ├─ h = gelu_tanh(g) * u                           # GeGLU (approximate gelu)
 ├─ h = down_proj(h)                               # FP8 GEMV  (+ TP all-reduce)
 ├─ h = post_feedforward_layernorm(h)              # sandwich: norm AFTER mlp, BEFORE residual add
 └─ x = residual + h                               # -> next layer
```

**Config — `RedHatAI/gemma-4-31B-it-FP8-block`, text backbone (read live from `model.config.text_config`):**

| field | value | field | value |
| --- | --- | --- | --- |
| hidden_size | 5376 | num_hidden_layers | 60 |
| intermediate_size | 21504 | num_attention_heads | 32 |
| head_dim (local / global) | 256 / 512 | num_key_value_heads (local) | 16 |
| num_global_key_value_heads | 4 | attention_k_eq_v | true |
| rms_norm_eps | 1e‑6 | hidden_activation | gelu_pytorch_tanh |
| sliding_window | 1024 | layer_types | 5 sliding : 1 full (10 full of 60) |
| rope θ local / global | 10 000 / 1 000 000 | partial_rotary local / global | 1.0 / 0.25 |
| max_position_embeddings | 262 144 | vocab_size | 262 144 |
| final_logit_softcapping | 30.0 (LM head) | tie_word_embeddings | true |

> **Two Gemma‑4 features that bite the kernel:** (1) local and global layers have **different
> `head_dim` (256 vs 512) and KV‑head counts (16 vs 4)**, so K1's geometry is layer‑type‑dependent —
> size the attention tiles per layer type, not once; (2) global layers apply RoPE to only
> **25 % of head_dim** (`partial_rotary_factor=0.25`), local layers to all of it. Still **read every
> number from `model.config.text_config`** in code (that's what `_unsupported` does) and
> **hard‑gate** on it; plug the sizes into the geometry classes (§4.3).

### 2.2 Differences from Qwen3.8‑27B dense (what you must change)

| Aspect | Qwen3.8‑27B dense (#61573) | **Gemma‑4 (yours)** |
| --- | --- | --- |
| Mixer (K1) | GDN linear‑attention (`dense_gdn.py`) | **Full MHA + QK‑norm + sliding/global** (new K1) |
| MLP act | SiLU‑mul | **GeGLU (gelu_tanh)·mul** |
| Norms | pre + post (K2 already uses *Gemma* norm) | **4 sandwich norms / layer** (same RMSNorm kernel) |
| Weights | Quark MXFP4 | **block FP8** — e4m3 weights in 128×128 blocks + dynamic per‑group‑128 FP8 activations (compressed‑tensors `FP8_BLOCK`) |
| q/k norm | none in GDN | **q_norm, k_norm per head** (new in K1) |
| attention | n/a (linear) | **paged attention over bf16 KV, windowed** (`kv_cache_scheme: null` — KV stays bf16; only Linear weights/acts are FP8) |

**The FFN (K2) is ~90 % reusable.** You swap SiLU‑mul for GeGLU and keep the norm/GEMV/split‑K
machinery. **K1 is the real new kernel**: an FP8 GEMV for QKV, two per‑head RMSNorms, RoPE, a
*decode* attention over the paged KV cache (windowed for local layers), and o_proj.

### 2.3 The decode attention sub‑problem (K1's hard part)

At decode, attention is one query row per request against the KV cache:

- **Local (sliding‑window) layers** (hd=256, 16 KV heads) attend to the last `sliding_window=1024`
  keys. The KV cache block table already encodes the window; you load the windowed keys/values
  (**bf16** for this checkpoint — `kv_cache_scheme: null`, no FP8 KV), do `scale·(q·kᵀ)`,
  online‑softmax, `·v`. This is small and LDS‑resident.
- **Global layers** (hd=512, 4 KV heads, 25 % partial RoPE) attend to the full context (up to
  256 K). This is bandwidth‑bound; you will
  likely **leave global‑layer attention on the stock AITER paged‑attention kernel** and only
  fuse the surrounding GEMVs/norms — exactly how DSv4.1's mono runs an FFN‑only launch on its
  "heavy" layers (#60397 scope: whole‑layer on standard layers, FFN‑only on compressor/index
  layers). **Recommended first cut: fuse local layers fully (K1+K2); on global layers run stock
  attention + your K2.**

This checkpoint keeps a **bf16 KV cache** (`kv_cache_scheme: null`), so attention reads bf16 K/V
directly (no KV dequant). If you later turn on an FP8 KV cache, `BaseKVCacheMethod`
k_scale/v_scale/prob_scale with per‑token‑head scales is the path. The numerics contract is
"reproduce the stock ROCm attention
chain" (DSv4.1 README §Numerics is the template: `mxfp8(x)` block‑32 E8M0, softmax in base‑2
with the sink folded in, etc.).

---

## PART 3 — THE TOOLCHAIN & HARDWARE CONTRACT

### 3.1 FlyDSL (what the kernels are written in)

Mono kernels are **not raw HIP**. They are **FlyDSL** — AMD's Python‑embedded DSL (shipped via
`aiter.ops.flydsl`) that lowers Python kernel bodies to MLIR → ROCDL → amdgcn. One file both
*emits the device kernel* and *plans the host launch*. Key facts (from `common/ops.py`,
`common/plan.py`):

- You import `flydsl.expr as fx`, `rocdl`, `gpu`, and `aiter.ops.flydsl.kernels.buffer_ops as bo`.
- A kernel is a `@flyc.kernel` function; helpers are `@traced` (`ASTRewriter.transform`) so their
  Python `while`/`if` over traced values lower to structured control flow (`scf`).
- **Raw addresses, not tensors.** Kernels take `Int64` addresses; you build a buffer resource
  with `rsrc(addr)` = `bo.create_buffer_resource_from_addr(...)` and load/store with
  `bo.buffer_load/buffer_store`. Loads past 4 GiB must use `global_load` (64‑bit addressing).
- **The build is keyed by scalars only.** FlyDSL's JIT caches a compiled kernel by the *scalar
  closure values* of its launcher. So every build parameter is one **frozen dataclass of scalars**
  (`FfnBuild`, `Geo`…), turned into a cache key by `key_tuple(key, sources)`. Two builds differing
  only in a non‑scalar would *share a binary* — that once made every TP rank run rank 3's kernel.
  The key also carries `source_digest(...)` (sha256 of your kernel `.py` files) so an edit
  rebuilds.
- **ABI by name.** A kernel and its launcher spell out positional params; `KernelAbi` is the one
  list of names and `check()` verifies both match (an out‑of‑order arg is a silent pointer bug).

> You *could* write the kernel in raw HIP under `csrc/` (the Kimi‑K3 `fused_kda_decode_kernel_
> rocm.cu` does), but the mono family standardized on FlyDSL because the host *plan* and the
> device *kernel* must agree on dozens of tiling constants, and FlyDSL lets one file own both.
> **Follow the family: use FlyDSL.**

### 3.2 The device primitives you get for free (`common/ops.py`)

Reuse these verbatim (copy the `common/` package from the dense PR). Highlights:

- **Reductions/shuffles:** `wave_sum`, `wave_max`, `butterfly(v, offsets)` (the *offset order is
  the numerical contract* — match aiter's per‑head norm order to stay bit‑exact), `block_max`,
  `wave_incl_scan`, `block_excl_scan`, `xshfl`/`xred` (DPP within a 16‑lane row, `permlane*_swap`
  across halves), `readlane`, `ballot`/`popcount`.
- **Approximate HW math:** `hw_rsq`, `hw_rcp`, `hw_exp2` (`v_rsq/rcp/exp2_f32`), `div_rn`
  (bit‑exact IEEE divide via Markstein), `sigmoid`, `pow2` (e8m0→f32).
- **MFMA:** `mfma_bf16` (`mfma_f32_16x16x32_bf16`), `mfma_fp8` (`mfma_f32_16x16x32_fp8_fp8`,
  unscaled, operands = i64 of 8×e4m3), `mfma_scaled` (block‑scaled, in `common/mx.py`).
- **FP8/BF16 packing:** `fp8_pack4`, `fp8x8_bf16_pk`, `bf16_pair`, `bf_lo/bf_hi/bf2_f32`.
- **Loads:** `ld_dev` (device scope), `global_load` (64‑bit, nontemporal option), `batched_rounds`
  (issue a batch of loads before any use — software pipelining), `rows_to_lds`.
- **Codegen hygiene:** `fresh(v)` (opaque move so LLVM recomputes lane math at use, avoiding
  cross‑stage register spills — **a kernel with scratch is refused**), `unlikely(cond)` (lay rare
  paths out of line), `uniform`/`readfirstlane`.

### 3.3 Hardware contract (CDNA3 vs CDNA4)

| | MI300X (gfx942, CDNA3) | MI355X (gfx950, CDNA4) |
| --- | --- | --- |
| CUs | 304 | **256** (⇒ BLOCKS=256) |
| LDS / CU | 64 KiB | **160 KiB** |
| Wavefront | 64 | 64 |
| FP8 format | **e4m3 FNUZ** (bias 8, max ±240) | **e4m3 OCP‑FN** (bias 7, max ±448) |
| FP8 MFMA | 16×16×32, 32×32×16 fp8 | + 16×16×128 / 32×32×64 **f8f6f4**, **block‑scaled** `mfma_scale_*` |
| Async copy | buffer_load→LDS | 128‑bit `global_load_lds` |

**Portability landmine:** the *same FP8 byte decodes to different real values* on FNUZ vs OCP.
FP8 weights/KV quantized on MI300X are **not bit‑portable** to MI355X. Start on **gfx950**
(like every lead mono PR); port to gfx942 as a follow‑up (#60731/#60997 show it's a separate,
nontrivial effort — smaller LDS forces different tiling).

### 3.4 AITER (the library under everything)

- `rocm_aiter_ops` (`vllm/_aiter_ops.py`) is the singleton that exposes AITER kernels (rms_norm,
  mhc pre/post, fused GEMMs, MoE, paged attention) and manages the `UncachedIpcHeap` peer buffer.
- Reference GEMM you must match bit‑for‑bit for this checkpoint: the **block‑FP8** path
  `gemm_a8w8_blockscale` (128×128 weight blocks, dynamic per‑group‑128 FP8 activations), plus the
  `gelu_and_mul` custom op. (`gemm_afp4wfp4`/`dynamic_mxfp4_quant` are the MXFP4 path other mono
  PRs use — not this one.) The dense PR's `_ref_linear`/`_ref_silu_and_mul` (`dense_mono.py`) are
  the pattern for your golden references; write `_ref_gelu_and_mul` for Gemma.
- **Pin an AITER version.** The mono PRs pin exact AITER commits/images (#61677 "re‑bump AITER
  to v0.1.25 and pin the DSv4.1 mono reference MoE"); APIs move (e.g. `communication_ops_utils.
   traced` removed in v0.1.25). Pick one and record it in your README.

---

## PART 3.5 — FLYDSL FROM ZERO (do this *before* Part 4 if you've never written a kernel)

If you have never written GPU‑kernel or FlyDSL code, do **not** start on K2. Spend a day on these
**eight tiny exercises** first. Every one follows the same loop:

> **WRITE** a ~20–60 line piece → **RUN** it with one exact command → **LOOK/MEASURE** a specific
> number → **UNDERSTAND** one concept → the piece is **a real building block of the megakernel.**

They go from "launch a kernel at all" to "a persistent multi‑CTA stage with a mailbox hand‑off" —
which *is* the megakernel's skeleton. Keep them in a scratch package you can delete later:

```text
scratch/flydsl_bootcamp/
    ex0_copy.py  ex1_pipe.py  ex2_wavesum.py  ex3_rmsnorm.py
    ex4_qdq.py   ex5_gemv.py  ex6_geglu.py    ex7_mailbox.py
    run.py       # a tiny launcher harness you grow across exercises
test_bootcamp.py # pytest: each exercise asserts == torch reference
```

First, confirm the toolchain exists and learn what to import (this is the only "read" step):

```bash
# WRITE: nothing yet.  RUN:
.venv/bin/python -c "import flydsl, aiter; from aiter.ops.flydsl.kernels import buffer_ops; \
print('flydsl ok', flydsl.__file__)"
.venv/bin/python -c "from vllm.platforms.rocm import on_gfx950; print('gfx950:', on_gfx950())"
# LOOK/MEASURE: both print without ImportError, and gfx950 is True on your box.
# UNDERSTAND: FlyDSL ships inside AITER; mono kernels are gfx950-first. If this fails, fix the
#             env (AGENTS.md: uv + .venv, pinned AITER) before writing a single kernel line.
```

> Treat the real sources as your API reference throughout: `vllm/models/qwen3_5/amd/mono/common/
> {ops,sync,plan}.py` (copy them into your scratch package and import from there). The code
> sketches below show the *shape*; the exact FlyDSL signatures live in those files.

---

### EX0 — Launch a kernel at all: a buffer copy

- **WRITE** (`ex0_copy.py`): a `@flyc.kernel` that copies `n` i32 from `src` to `dst`, one
  element per thread of a `THREADS=512` block, and a launcher that packs `(src_addr, dst_addr, n)`
  and launches `BLOCKS=1`. Use `rsrc(addr)` + `bo.buffer_load/buffer_store` (raw addresses, not
  tensors — pass `t.data_ptr()` from the host).

```python
@flyc.kernel
def copy_k(src, dst, n):
    tid = fx.block_idx.x * THREADS + fx.thread_idx.x
    if tid < n:
        v = bo.buffer_load(rsrc(src), tid, vec_width=1, dtype=T.i32)
        bo.buffer_store(v, rsrc(dst), tid)
```

- **RUN:** `.venv/bin/python -m scratch.flydsl_bootcamp.ex0_copy`  (host: make `a=torch.arange(...)`
  on GPU, `b=torch.empty_like(a)`, launch, then `assert torch.equal(a, b)`).
- **LOOK/MEASURE:** the assert passes; print the compiled kernel symbol name and the **first‑call
  vs second‑call** wall time (`time.perf_counter()` around two launches).
- **UNDERSTAND:** the *first call pays JIT compilation* (seconds), the second is microseconds —
  this is exactly why the megakernel needs **warm‑up** (§4.10). You now own the launch harness.
- **Feeds the megakernel:** the host launcher + `data_ptr()` packing pattern used by every stage.

### EX1 — Software pipelining: grid‑strided copy with batched loads

- **WRITE** (`ex1_pipe.py`): copy `n ≫ THREADS` using a strided loop, but issue a **batch of loads
  before any store**. Do it twice — once naive (load; store; load; store), once with the family's
  `batched_rounds(tid, n, load=…, use=…, batch=4)` helper from `common/ops.py`.
- **RUN:** `.venv/bin/python -m scratch.flydsl_bootcamp.ex1_pipe --n 8388608 --iters 200`
  (loop the launch, take the median).
- **LOOK/MEASURE:** effective bandwidth = `2*n*4*iters / seconds` GB/s for both variants; compare
  to the GPU's HBM peak. The batched version should be meaningfully closer to peak.
- **UNDERSTAND:** separating *issue* from *use* hides HBM latency — the single most important
  megakernel idea ("no bubbles"). A store between loads serializes them.
- **Feeds the megakernel:** `batched_rounds` is how every GEMV stage prefetches weights before
  polling activations.

### EX2 — Wave‑level reduction (and why order matters)

- **WRITE** (`ex2_wavesum.py`): sum 64 f32 (one per lane of a wavefront) to a single value using
  `butterfly(v, (32,16,8,4,2,1))` / `wave_sum` from `common/ops.py`. Then compute a second version
  with offsets in a **different order**.
- **RUN:** `.venv/bin/python -m scratch.flydsl_bootcamp.ex2_wavesum` (reference: `x.sum()` in torch).
- **LOOK/MEASURE:** print `(kernel_sum − torch_fp64_sum)` for both orders. The *aiter order* should
  match the reference as closely as fp32 allows; the other order differs in the low bits.
- **UNDERSTAND:** **reduction order is a numerical contract.** Matching aiter's butterfly order is
  what keeps RMSNorm/attention *bit‑exact* vs stock (Pitfall #3). Also learn `gpu.barrier()` and
  `block_max`/`block_excl_scan` for cross‑wave reductions.
- **Feeds the megakernel:** the reduction inside every RMSNorm, QK‑norm, and softmax stage.

### EX3 — Gemma RMSNorm `(1+w)` as a one‑block kernel

- **WRITE** (`ex3_rmsnorm.py`): for a `[rows, H]` tensor, each block normalizes one row:
  `rstd = hw_rsq(mean(x²) + eps)`, `y = x * rstd * (1 + w)`. Reductions via EX2; `hw_rsq` from
  `common/ops.py`. Match `vllm.model_executor.layers.layernorm.GemmaRMSNorm` (note the `+1` on the
  weight and the fp32 accumulation).
- **RUN:** `.venv/bin/python -m pytest scratch/flydsl_bootcamp/test_bootcamp.py -k rmsnorm -q`
  (reference = `GemmaRMSNorm(H)(x)`).
- **LOOK/MEASURE:** max abs error and per‑row cosine vs the stock module; aim for ≤ a bf16 ulp.
- **UNDERSTAND:** the Gemma sandwich‑norm primitive — you'll call this **four times per layer**.
  Also learn fp32‑accumulate‑then‑round‑to‑bf16 discipline.
- **Feeds the megakernel:** the `norm` stage of K1 and K2 verbatim (this is `_gemma(1+w)`).

### EX4 — FP8 quant / dequant pack (the qdq every stage uses)

- **WRITE** (`ex4_qdq.py`): quantize a bf16 row to the checkpoint's **block FP8** — e4m3 weights in
  128×128 blocks, activations dynamic per‑group‑128 (use `fp8_pack4`/`fp8x8_bf16_pk` from
  `common/ops.py`), then dequantize back.
- **RUN:** `.venv/bin/python -m pytest scratch/flydsl_bootcamp/test_bootcamp.py -k qdq -q`
  (reference = the stock block‑FP8 quant used by `gemm_a8w8_blockscale`).
- **LOOK/MEASURE:** assert your packed bytes **equal** aiter's byte‑for‑byte; print the
  dequant round‑trip error.
- **UNDERSTAND:** the megakernel feeds FP8 into MFMA, so you must pack **exactly** like the
  reference GEMM expects (Pitfall #11). This is finicky; isolating it here saves hours later.
- **Feeds the megakernel:** the activation qdq before every GEMV stage.

### EX5 — A one‑tile FP8 GEMV with MFMA (the core of every projection)

- **WRITE** (`ex5_gemv.py`): compute one output tile `y[0:16] = W[0:16, :] · x` with FP8 `W` and
  `x`, using `mfma_fp8` (`mfma_f32_16x16x32_fp8_fp8`) accumulated over the K dimension **in the
  stock GEMM's K‑split order** (`split_total` from `dense_ffn.py`).
- **RUN:** `.venv/bin/python -m pytest scratch/flydsl_bootcamp/test_bootcamp.py -k gemv -q`
  (reference = `_ref_linear` — aiter `gemm_afp4wfp4`/scaled‑mm — on the same 16‑row slice).
- **LOOK/MEASURE:** assert **bit‑equality** with the reference slice; if it fails, print where the
  K‑split sums diverge (this is the 24‑bit lesson). Also time it and compute TFLOP/s.
- **UNDERSTAND:** MFMA operand layout (who holds which lanes), K‑split order, and why the MFMA
  *shape* can change but the K *grouping* cannot.
- **Feeds the megakernel:** `Wqkv`, `o_proj`, `gate_up`, `down` are all this, tiled over more CTAs.

### EX6 — GeGLU epilogue (`gelu_tanh(g) * u`) bit‑exact

- **WRITE** (`ex6_geglu.py`): given two tiles `g,u`, compute `bf16(gelu_tanh(g)) * u` using
  `hw_exp2`‑based `gelu_tanh` (your `helpers.gelu_tanh`). Match the stock `GeluAndMul` custom op's
  bf16 rounding exactly.
- **RUN:** `.venv/bin/python -m pytest scratch/flydsl_bootcamp/test_bootcamp.py -k geglu -q`
  (reference = a `_ref_gelu_and_mul` wrapping `torch.ops._C.gelu_and_mul`).
- **LOOK/MEASURE:** assert **bit‑equality** on every element.
- **UNDERSTAND:** this is the *only* compute difference between Gemma's K2 and Qwen's (SiLU‑mul);
  nailing its rounding is what keeps K2 bit‑exact.
- **Feeds the megakernel:** the gate_up epilogue of K2 (§4.4).

### EX7 — A mailbox hand‑off across two CTAs (the megakernel's nervous system)

- **WRITE** (`ex7_mailbox.py`): launch `BLOCKS=2`, `THREADS=512`, **persistent**. CTA 0 computes a
  partial and `publish()`es a `(value, tag)` pair into a scratch region (`sreg`, device scope);
  CTA 1 `poll()`s that region until the tag matches, then adds its own partial. Use `Mailbox`,
  `publish`, `sreg` from `common/sync.py`; zero the region before launch.
- **RUN:** `.venv/bin/python -m scratch.flydsl_bootcamp.ex7_mailbox` (reference: the two partials
  summed on host).
- **LOOK/MEASURE:** the assert passes; now **break it on purpose** — remove the `publish()`
  waitcnt+barrier and poll the payload directly — and watch it produce stale/garbage values
  intermittently. Restore it.
- **UNDERSTAND:** CTA↔CTA hand‑off = tagged pair + poll, and *the flag must not overtake the data*
  (`publish` = waitcnt+barrier, then the pair; Pitfall #4). Device scope (`CM_DEV`) vs system scope
  (`CM_SYS`) for the TP all‑reduce. This is the exact mechanism K1→K2 and rank→rank use.
- **Feeds the megakernel:** every stage boundary and the in‑kernel all‑reduce (§1.4, §4.8).

### Graduation check (you are ready for Part 4 when…)

```bash
.venv/bin/python -m pytest scratch/flydsl_bootcamp/test_bootcamp.py -q
```

- All seven exercises green.
- You can explain, without looking: why the 2nd launch is faster (JIT warm‑up), why batched loads
  beat interleaved (bubbles), why butterfly order matters (bit‑exactness), and why `publish`
  precedes a poll (flag‑vs‑data ordering).
- EX3 (RMSNorm), EX4 (qdq), EX5 (GEMV), EX6 (GeGLU), EX7 (mailbox) are literally the stages of K2.
  **Assembling them in order — norm → qdq → gate_up GEMV → GeGLU → down GEMV → norm, with mailbox
  hand‑offs — *is* Milestone 1.** Delete the scratch package once Part 4's `common/` is in place.

---

## PART 4 — STEP‑BY‑STEP IMPLEMENTATION

> Build order: **K2 first** (bit‑exact, self‑contained) → **K1 local attention** → **TP** →
> **global‑layer handling** → **integration/graph** → **warmup/safety**. Keep the stock path
> untouched; everything is behind the flag.

### 4.0 Directory layout (clone the dense PR's shape)

```text
vllm/models/gemma4/
    __init__.py
    amd/
        __init__.py
        dense_mono.py            # eligibility, compiled mono graph, custom ops, vLLM tensors
        mono/
            __init__.py
            attn_k1.py           # K1: input_norm -> Wqkv -> q/k norm -> RoPE -> attn -> o_proj
            dense_ffn.py         # K2: norms -> gate_up -> GeGLU -> down -> norm   (fork of #61573)
            helpers.py           # step_tag, gelu_tanh, e8m0
            sources.py           # source_digest() over this package (JIT cache correctness)
            common/              # device ops, mailbox, MX, plan/build keys  (COPY from #61573)
                ops.py  sync.py  plan.py  mx.py
tests/models/gemma4/
    test_amd_dense_mono.py       # FfnTest, AttnTest, RowCapsTest, StockChainTest
vllm/model_executor/models/gemma4.py   # Gemma4 text backbone: bind DenseMono at init, wrap forward
```

Add to `.github/CODEOWNERS` and `.buildkite/test_areas/models_basic.yaml` (every mono PR does).

### 4.1 Step A — Scaffolding + enable flag + eligibility guards

1. Add the env flag(s) in `vllm/envs.py` (mirror `VLLM_ROCM_MONO_DECODE`,
   `VLLM_ROCM_QWEN38_DENSE_MONO_K1_MAX_ROWS` / `_K2_MAX_ROWS`, `_OPROJ`, `_K2`). Or wire
   `KernelConfig.monokernel` / `specialized_decode` in `vllm/config/kernel.py` + `arg_utils.py`.
2. Write `dense_mono.py::_unsupported(model, vllm_config)` returning the first reason or `None`:
   gfx950, CU==BLOCKS, TP∈{1,2}, no PP/LoRA/KV‑connector/SP/spec, `hidden_size`/`intermediate_
   size`/`hidden_act`=='gelu_pytorch_tanh' match, every layer's MLP is the dense Gemma MLP,
   sandwich norms present. **Raise with the reason at init if the flag is set and `_unsupported`
   is not None.** (Copy the structure from #61573 lines ~106–232.)
3. Add `row_caps(tp)` and the width tuple → these become your **CUDA‑graph capture sizes**
   (`_set_capture_sizes` / `widths`).

**Verify:** start the server with the flag on an unsupported config → it refuses and lists the
reason; on a supported config → it logs `widths [...]`.

### 4.2 Step B — Build keys, geometry, ABI (the scalar contract)

Define the per‑TP geometry and the frozen build‑key dataclasses. Fork `FGeo`/`FfnBuild` from
`dense_ffn.py` and add a `Geo`/`AttnBuild` for K1.

```python
HIDDEN = <config.hidden_size>          # 5376-ish; READ FROM CONFIG, gate on it
INTER  = <config.intermediate_size>
NH, NKV, HD = <heads>, <kv_heads>, <head_dim>
ROWS   = 16                            # MFMA tile token dim; MAX_TOKENS=8 (TP1), 16 (TP2)

@dataclass(frozen=True)
class FfnBuild:                        # every field a scalar -> reaches the JIT key
    tokens: int
    eps: float = 1e-6
    shuffled: bool = False             # aiter shuffle_weight((16,16)) layout
    oproj: bool = False                # fold o_proj into this launch
    tp: int = 1
    normed: bool = False               # input is post_attention_layernorm output (TP2)
    geglu: bool = True                 # GEMMA: gelu_tanh instead of silu   <-- your change

ABI = KernelAbi(("hidden","residual","w_o","w_os","ln_w","w13","w13s",
                 "w2","w2s","out","res_out","scratch","epoch","layer"))
```

Match **stock split‑K order** exactly: `gu_splits`, `d_splits`, `O_SPLITS`, `split_total`
(`dense_ffn.py`). The MFMA *shape* (16×16×128 vs stock 32×32×64) may differ but the **K grouping
must equal the stock GEMM's**, or sums beyond 24 bits diverge.

### 4.3 Step C — Device primitives (reuse)

Copy `common/ops.py`, `common/sync.py`, `common/plan.py`, `common/mx.py` from #61573 **unchanged**
(they're model‑agnostic, adapted from ROCm/ATOM and ROCm/FlyDSL, MIT, with origin notices — keep
the copyright headers). Add only Gemma‑specific helpers to `helpers.py`:

```python
# helpers.py
LOG2E = 1.0 / math.log(2.0)
def gelu_tanh(x):                      # Gemma GeGLU gate activation (approximate gelu)
    # 0.5*x*(1+tanh(√(2/π)*(x+0.044715 x³)))  — implement via hw_exp2, matching the stock
    # GeluAndMul custom op's bf16 rounding exactly (see _ref_gelu_and_mul in tests)
    ...
def step_tag(epoch, slot): ...         # from #61573
def pow2(code): ...
```

### 4.4 Step D — K2: the FFN megakernel (do this first)

`dense_ffn.py` is your template; the stage structure is already correct for a dense layer:

```text
norm     : every CTA redundantly computes  res = h + residual (fp32) -> res_out (bf16);
           post_attention_layernorm/pre_ffn_norm (Gemma (1+w)) -> bf16 x -> block-FP8 qdq -> LDS
           (first gate_up weights already in flight — software pipelining)
gate_up  : CTA b owns a slice of the 2*INTER columns; wave picks gate/up by w%2, K-quarter by w/2;
           block-FP8 weights · x in the STOCK GEMM's K chains -> bf16 g,u
           h = bf16( gelu_tanh(g) ) * u          # <-- YOUR CHANGE: GeGLU, not silu-mul
           qdq h back to block-FP8 (the stock down GEMM's input precision) -> LDS
down     : CTA b owns a hidden row group; weights · HQ in the stock K chains -> bf16 out
           (TP2: write the rank's partial BEFORE the all-reduce)
```

Concrete changes vs #61573:

- Replace `silu` with `gelu_tanh` in the gate_up epilogue. Match the stock `GeluAndMul` op's
  rounding (write `_ref_gelu_and_mul` in the test and assert bit‑equality like #61573 does for
  silu).
- This checkpoint is **block FP8**, so use the block‑FP8 qdq (e4m3 weights in 128×128 blocks,
  dynamic per‑group‑128 activations, as `gemm_a8w8_blockscale` expects) with `mfma_fp8` /
  block‑scaled `mfma_scaled`. Keep the **K‑split order** identical to the stock block‑FP8 GEMM.
  (If you ever retarget an MXFP4 checkpoint, that's the `dynamic_mxfp4_quant`/`gemm_afp4wfp4` path
  instead.)
- Keep the sandwich‑norm order: K2 applies `post_attention_layernorm` on the attention output
  *and* `pre_feedforward_layernorm` before gate_up, each as Gemma `(1+w)` RMSNorm.

**Verify (RED→GREEN):** `tests/models/gemma4/test_amd_dense_mono.py::FfnTest` builds one layer's
real weights, runs K2 vs the stock chain (`_ref_linear` FP8 GEMM, Gemma norm, GeGLU), asserts
**bit‑equality** (K2 is bit‑exact), incl. the TP2 down‑partial‑before‑all‑reduce. This is your
first green milestone and needs **no GPU model, no eval**.

### 4.5 Step E — K1: the attention megakernel (the real work)

Stages (new file `attn_k1.py`, modeled on `attention/front.py` + `attention/back.py` from #60397, but for **dense MHA + QK‑norm + windowed decode** instead of MLA):

```text
norm     : fused_add_rms_norm(input)  (Gemma (1+w)) -> x -> block-FP8 qdq -> LDS
qkv      : Wqkv FP8 GEMV -> q[nH,hd], k[nKV,hd], v[nKV,hd]   (stock K-split order)
qk_norm  : per-head RMSNorm over hd on q and k  (butterfly order == aiter's, stay bit-exact)
rope     : load cos/sin by position; rotate first rotary_dim of q,k; θ picked by local/global
kv_write : insert k,v into the paged KV cache (bf16 for this checkpoint) at this step's slots
attend   : LOCAL layer (hd=256,16 KV): load windowed bf16 keys/values (last 1024) -> scale·qkᵀ ->
           online softmax (base-2, no attn soft-cap) -> ·v -> bf16 core
           GLOBAL layer: see §4.7 (leave on stock attention first)
o_proj   : FP8 GEMV; push the (unreduced) partial to K2 / to peers (TP)
```

Numerics contract: reproduce vLLM's ROCm attention exactly — `head_dim**-0.5` scaling (no
`query_pre_attn_scalar` in Gemma‑4; local hd=256, global hd=512), QK‑norm eps 1e‑6, neox‑style
RoPE (full for local, 25 % partial for global), bf16 KV, softmax in base‑2.
Validate K1 against stock with a **cosine/abs bound**, not bit‑equality (K1 sums in a different
order and may differ by a bf16 ulp — the dense PR uses `_K1_COS_SLACK=1.25`, `_K1_ABS_SLACK=2.0`
vs an fp32 reference, requiring K1 to be *about as close to fp32 as stock is*).

**Verify:** `AttnTest` checks K1 vs the stock qkv→qk‑norm→RoPE→attention→o_proj chain and vs an
fp32 reference, for local layers, with pad rows zeroed (CUDA‑graph pad rows: `row_live`).

### 4.6 Step F — The runner (scratch, epoch, launches)

`mono/runner.py` (fork `runner.py` from #60397) owns per‑worker state: the scratch buffer
(mailbox regions via `pair_layout`), the `epoch` device word (incremented once per step by the
first layer's tail op — see `_tail_op` in #61573), the peer buffers, and the actual `flyc`
launches with the ABI‑packed arguments. It exposes `_layer(index, hidden, residual, positions)`
(TP1 whole layer) and the per‑stage ops `_k1_from_normed`, `_ffn_partial`, `_tail`.

Key correctness rules it enforces:

- **Zero mailboxes at step start** (`begin_step`) so stale tags never read live.
- **Bump `epoch` exactly once per step** (the tag's high bits), from layer 0.
- **Fixed device pointers only** (graph safety): preallocate scratch; never allocate in forward.

### 4.7 Step G — Global (full‑context) layers

Gemma's 1‑in‑6 global layers attend to up to 128 K tokens — bandwidth‑bound, not launch‑bound.
**First cut:** on global layers, run **stock attention** and only use your **K2** (FFN launch),
exactly like DSv4.1's "FFN‑only launch" on its heavy layers (#60397). Decide per layer in
`_layer`/`skip_step` using `config.layer_types[index]`. Full fusion of global attention is a
later optimization (it mainly saves the surrounding GEMV launches, not the attention itself).

### 4.8 Step H — Tensor parallelism

Two strategies, both in the dense PR:

- **TP1:** fold everything (incl. o_proj) into K1/K2; no all‑reduce.
- **TP2 (split graph, recommended first):** keep the stock norms and attention visible to
  `torch.compile`, run **each rank's K1/K2 partial**, and **all‑reduce with the same
  `RowParallelLinear` collective** (`_reduce()` in `dense_mono.py`) so FlyDSL QuickReduce still
  sees the op. This avoids the in‑kernel peer all‑reduce entirely and is far simpler.
- **In‑kernel peer all‑reduce (advanced):** the DSv4.1/Kimi‑K3 way — push `(value,tag)` system‑
  scope into every peer's symmetric buffer, reduce exact‑fp32‑in‑rank‑order→bf16. Do this only
  after TP2‑split works, for the extra win.

### 4.9 Step I — Integration: custom ops + the compiled mono graph

Follow `dense_mono.py` precisely:

```python
# register one custom op per fused chunk  (direct_register_custom_op)
torch.ops.vllm.gemma4_dense_mono_layer(hidden, residual, positions, index, first)  # TP1 whole layer
torch.ops.vllm.gemma4_dense_mono_attn(x, index)   # TP2 K1 partial (-> all-reduced by the graph)
torch.ops.vllm.gemma4_dense_mono_ffn(x, index)    # TP2 K2 partial
torch.ops.vllm.gemma4_dense_mono_tail(core, residual, index)  # bumps epoch on index==0

@support_torch_compile(dynamic_arg_dims={"input_ids":0,"positions":-1,"inputs_embeds":0})
class DenseMonoGraph(nn.Module):           # branch-free decode entry point
    def forward(self, input_ids, positions, inputs_embeds=None):
        return self._forward_tp1(...) if self._mono.whole_step else self._forward_tp2(...)
```

- A module can't be a custom‑op arg → stash the bound `DenseMono` in a module‑global `_BOUND`
  and read it back inside the op (`dense_mono.py::_BOUND`).
- The outer *uncompiled* forward calls `DenseMonoGraph` **only for an eligible width**; wider/
  non‑decode steps call the original compiled model (which is *not* wrapped). The mono graph is
  a *separate* compiled graph, so `torch.compile` traces norms + all‑reduces but keeps K1/K2 as
  opaque calls.
- **The model must not carry `@support_torch_compile` on the path the mono graph replaces**
  (the dispatcher refuses a `TorchCompileWithNoGuardsWrapper`, see `guards._refuse_model`).

**Verify:** at TP1, width 4 → the mono graph logs "layer graph at width 4" and produces logits
equal (within tolerance) to stock; width 32 → stock path, no mono launch; 0 fallback cost.

### 4.10 Step J — Warm‑up & capture sizes

- Make your kernel `DecodeWarmable` (if using #61311) or register a `VllmJitKernel`: expose
  `warmup_keys()` (one key per `(width, layer_type)` you build) and `compile(key)` so the first
  decode step doesn't pay JIT. The dense PR warms via the "layer graph at width W" capture.
- Ensure your `widths` are **added to `cudagraph_capture_sizes`** (`_set_capture_sizes`) so decode
  steps at those widths get a FULL graph and don't pad.

### 4.11 Step K — Fail‑stop safety (if you target `MonoKernel`)

Adopt `common/mono/{failstop,abi,peer}.py`: every in‑kernel wait is **bounded** (`_spin_bounded`
gives up after `DIAG_TICKS` = 10 s on the 100 MHz clock), records which region/tag it stalled
on, raises its rank's flag on the output rank, and sets a **sticky FAILED word** so the runner
raises *before* a bad step's tokens are returned (`model_runner` copies `step_health()` next to
the sampled ids; `get_output()` raises). This turns a hung peer into a clean engine error, not a
silent garbage token. Emit `vllm:monokernel_steps{spec,path}` / `vllm:monokernel_incidents{rank}`.

### 4.12 Step L — Porting to MI325X / MI300X (gfx942, CDNA3)

Everything above targets **MI355X (gfx950, CDNA4)** first, because that is where every lead mono
PR landed. **MI325X is CDNA3 (gfx942)** — the *same compute die as MI300X* (304 CUs, 64 KiB LDS,
FNUZ FP8, no block‑scaled MFMA), just with more/faster HBM. So "port to MI325X" means "port the
gfx950 kernel to gfx942", which the AMD team already did for other models — read those diffs first:
**#60731** ("DSv4.1 Mono decode on MI300X and MI325X (gfx942)") and **#60997** ("Kimi‑K3 Mono
decode kernel: port to gfx942"); #51195 ("Dispatch gfx942 verify to Gluon") is also useful. Treat
the port as a **second build target of the same FlyDSL sources**, gated by `on_gfx942()`, not a
rewrite — the stage structure, mailbox, and integration layer are unchanged. Seven things actually
differ; each is a concrete edit with its own before/after check.

1. **Grid size: `BLOCKS` must equal the GPU's CU count (304 on MI300X/MI325X, not 256).** The
   runner refuses a GPU whose CU count ≠ `BLOCKS` (§1.3), so `BLOCKS` becomes an arch constant
   (`current_platform.num_compute_units()`), and everything keyed on it — task tables
   (`first_task`, `b, b+BLOCKS, …`), mailbox region counts (`pair_layout`), peer layout — recomputes
   from it. *Check:* `rocminfo | grep "Compute Unit"` shows 304; assert `BLOCKS == num_CUs` at build,
   and re‑run a K2 `FfnTest` so the new task partition is exercised.
2. **LDS: 64 KiB, not 160 KiB — the biggest change.** A stage's LDS union (`LDS_BYTES`) must fit in
   64 KiB on gfx942. If your gfx950 tiles assume 160 KiB you must **shrink the per‑stage working
   set**: lower `MAX_TOKENS` (e.g. 8→4), tile the intermediate in more passes, or split a stage.
   *Check:* the FlyDSL build refuses if LDS over‑subscribes — print `lds_bytes(struct)` per
   stage and keep each ≤ 65536; re‑tune `row_caps` down for gfx942.
3. **FP8 is FNUZ, not OCP.** gfx942 e4m3 is **FNUZ** (bias 8, max ±240) vs gfx950 OCP (bias 7, max
   ±448): the *same byte is a different real value*. vLLM converts the checkpoint's OCP FP8 to FNUZ
   in `process_weights_after_loading` on gfx942, so your kernel reads **FNUZ** weights and must
   dequant/scale with FNUZ semantics and match the **gfx942 FNUZ block‑FP8 reference GEMM** (not the
   OCP one). *Check:* EX4 (qdq) and EX5 (GEMV) re‑run on gfx942 must stay bit‑exact against the
   *gfx942* `gemm_a8w8_blockscale` reference — a wrong bias shows up as a ~2× / off‑by‑exponent error.
4. **No block‑scaled MFMA.** CDNA4's `mfma_scale_f32_*_f8f6f4` / block‑scaled `mfma_scaled`
   (`common/mx.py`) **do not exist on gfx942**. Replace them with an **unscaled `mfma_fp8`
   (`mfma_f32_16x16x32_fp8_fp8`) plus the e8m0 block scale applied in software** around the MFMA
   (dequant the 128‑block, or fold the scale into the accumulator). *Check:* a GEMV microbench vs the
   fp32 reference; the scaled‑vs‑manual path must match to a bf16 ulp.
5. **Cross‑lane swaps differ.** `permlane16_swap`/`permlane32_swap` (used by `permlane_swap` → the
   `xred`/`butterfly` 16/32 offsets in `common/ops.py`) are CDNA4 instructions. On gfx942 implement
   the 16/32‑lane butterfly with `ds_bpermute`/`ds_permute` (or DPP where it reaches), **preserving
   the exact reduction order** (the numerical contract, Pitfall #3). *Check:* EX2 (wave reduction) on
   gfx942 must still match the fp64 reference and the aiter order bit‑for‑bit.
6. **Device‑ vs system‑scope cache bits, and 8 XCDs.** MI300X/MI325X package **8 XCDs**; "device
   scope" (past the per‑XCD L2) and the CPol bits `CM_DEV`/`CM_SYS` (`common/ops.py`, commented as
   *gfx95x*) must be re‑derived for gfx942 so the **CTA↔CTA mailbox** and the **peer all‑reduce**
   (system scope over XGMI) still see each other's stores. This is the subtlest correctness risk —
   a wrong scope silently reads stale mailbox data. *Check:* EX7 (mailbox) and a TP2 all‑reduce test
   on gfx942; a scope bug shows up as intermittent wrong tokens, so run it at several widths.
7. **Async copy / clock width.** CDNA4's 128‑bit `global_load_lds` narrows on CDNA3; the
   `rows_to_lds`/buffer‑load→LDS paths may need a smaller transfer width. The `s_memrealtime` clock
   feeds `DIAG_TICKS` (the 10 s fail‑stop) — confirm its frequency on gfx942 and adjust the constant
   if it differs. *Check:* a timed `rows_to_lds` microbench; verify the fail‑stop still trips at ~10 s.

**Order of work:** gate a gfx942 build behind `on_gfx942()` and run the **bootcamp (Part 3.5) on the
MI325X first** — EX2/EX4/EX5/EX7 will immediately surface the swap, FNUZ, scaled‑MFMA, and scope
differences in isolation, before you touch K1/K2. Then bring up **K2 (bit‑exact, FNUZ) → K1**, re‑tune
`row_caps`/`MAX_TOKENS` for 64 KiB LDS, and finally re‑run the §5 numerics + a perf sweep on the
MI325X. Keep gfx950 and gfx942 as two builds of one source; never fork the kernels.

---

## PART 5 — TEST, VALIDATE, BENCHMARK

### 5.1 Numerics (the gate that lets you ship without a full eval)

`pytest tests/models/gemma4/test_amd_dense_mono.py` on one MI355:

- **`FfnTest`** — K2 **bit‑exact** vs stock FP8 GEMM + Gemma norm + GeGLU, incl. TP2 down‑partial.
- **`AttnTest`** — K1 vs stock qkv→qk‑norm→RoPE→attn→o_proj, **and** vs fp32, within
  `(1 − min‑row‑cosine)` and max‑abs slacks; pad rows zero; local & (if fused) global.
- **`RowCapsTest`** — steps inside the cap take the mono path; outside take stock; the clamp/warn
  on out‑of‑range `*_MAX_ROWS` works.
- **`StockSplitTest` / `StockChainTest`** — K2 follows the stock GEMM's split‑K reduce order.

Each test builds **real checkpoint weights** in a TP process group and compares against the
model's **own vLLM modules** — this is the DSv4.1/Qwen3.8 methodology (README §Validation).

### 5.2 End‑to‑end accuracy (do before claiming a win)

Run the standard evals mono‑on vs mono‑off and show they're within noise (per AGENTS.md:
"run model evals for model‑affecting changes"): GPQA‑Diamond, AIME25, MMLU/MMMU, and a RULER
NIAH long‑context point at 32K/128K. The dense PR shows GPQA 0.8232→0.8434 (TP1), NIAH flat —
your target is "within noise", since the kernel is numerically equivalent by construction.

### 5.3 Performance (manual QA — actually run it)

`sglang.bench_serving` / `vllm bench serve`, 8k in / 1k out, random dataset, sweep concurrency
1/4/8/16/32. Expect gains **inside the row cap** (conc 4/8) and **parity outside** (the kernel
doesn't launch; prove "no fallback cost", within ±1 %). Record TPOT/ITL/output‑tok‑s medians,
mono‑on vs mono‑off, on the **same image/commit**, both arms **simultaneously** on split GPUs
(the DSv4.1 methodology). Profile one decode step with `rocprof`/torch profiler to confirm the
launch count dropped (DSv4.1: ~40 kernels/layer → 2; Kimi‑K3 #61435: 607 → 415 kernels/step).

---

## PART 6 — TASK BREAKDOWN (execute top‑to‑bottom)

> Every task uses the same loop as the bootcamp: **WRITE** a small change → **RUN** one exact
> command → **LOOK/MEASURE** a specific signal → **UNDERSTAND** what it proved. Each task is 1–3
> edits; keep exactly one `in_progress`. `S*` = scenario it advances (S1 K2 bit‑exact, S2 K1
> cosine‑bound, S3 e2e decode correct). Do **Part 3.5 (bootcamp)** first if you've never written a
> kernel. All commands assume the repo's `.venv` (AGENTS.md: `uv` + `.venv/bin/python`).

### Milestone 0 — Groundwork

- [ ] **0.1 — Read the config.** WRITE: the Gemma‑4‑31B shape into `mono/__init__.py` constants
      (hidden/intermediate/heads/kv/head_dim/eps/sliding_window/layer_types/rope θ/quant) + a
      config‑gate list. RUN: `.venv/bin/python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('RedHatAI/gemma-4-31B-it-FP8-block'); print(c)"`.
      LOOK/MEASURE: your constants equal the printed config field‑for‑field. UNDERSTAND: these are
      the values `_unsupported` will hard‑gate on — a mismatch here is a wrong‑kernel bug later.
- [ ] **0.2 — Confirm the box.** WRITE: nothing. RUN: `rocminfo | grep -m1 gfx` and
      `.venv/bin/python -c "import aiter; print(aiter.__version__)"`. LOOK/MEASURE: arch is `gfx950`
      and the AITER version matches the one you pin in the README. UNDERSTAND: mono is arch‑ and
      AITER‑version‑pinned (Pitfall #6, §3.4); wrong arch/version = silent numeric drift.
- [ ] **0.3 — Scaffold.** WRITE: the §4.0 directory skeleton; copy `common/` from #61573 with its
      headers. RUN: `.venv/bin/python -c "import vllm.models.gemma4.amd.mono.common.ops as o; print(o.BLOCKS, o.THREADS)"`.
      LOOK/MEASURE: prints `256 512` with no ImportError. UNDERSTAND: the execution‑model constants
      (§1.3) are now importable — your kernels and launchers will share them.
- [ ] **0.4 — Register with CI.** WRITE: CODEOWNERS + `.buildkite/test_areas/models_basic.yaml`
      entries. RUN: `git diff --stat`. LOOK/MEASURE: both files changed. UNDERSTAND: the numerics
      test (Milestone 1) will run on a real MI355 CI node.

### Milestone 1 — K2 (FFN) bit‑exact  *(first shippable green)*

- [ ] **1.1 — Geometry + ABI.** WRITE: `FfnBuild`, `FGeo`, `ABI`, split‑K helpers
      (`gu_splits`/`d_splits`/`split_total`) for Gemma shapes. RUN:
      `.venv/bin/python -m pytest tests/models/gemma4/test_amd_dense_mono.py -k abi -q`.
      LOOK/MEASURE: `KernelAbi.check()` passes on a stub launcher (asserts the arg names line up).
      UNDERSTAND: a scalar‑only build key + name‑checked ABI prevents the shared‑binary / arg‑order
      bugs (Pitfalls #1, and `plan.py`).
- [ ] **1.2 — Golden references.** WRITE: `_ref_linear` (aiter FP8 GEMM), `_ref_gelu_and_mul`,
      `_gemma(1+w)` norm in the test. RUN:
      `.venv/bin/python -m pytest tests/models/gemma4/test_amd_dense_mono.py -k ref -q`.
      LOOK/MEASURE: the refs run on one layer's real weights and return finite tensors of the right
      shape. UNDERSTAND: these *are* the stock chain K2 must equal; EX4/EX6 feed straight in.
- [ ] **1.3 — RED.** WRITE: `FfnTest` asserting `K2(x) == stock_chain(x)` against a *not‑yet‑built*
      kernel. RUN: `.venv/bin/python -m pytest tests/models/gemma4/test_amd_dense_mono.py::FfnTest -q`.
      LOOK/MEASURE: it fails with a **shape/precision assertion**, not an ImportError. UNDERSTAND:
      the test exercises the real surface before any kernel code exists (TDD floor).  **(S1)**
- [ ] **1.4 — GREEN (K2 TP1).** WRITE: `dense_ffn.py` stages norm→gate_up→**GeGLU**→down→norm
      (reuse EX3/EX4/EX5/EX6), stock K‑split order. RUN: same `FfnTest` command.
      LOOK/MEASURE: GREEN, and print `max_abs_err` — it must be **0** (bit‑exact), TP1.
      UNDERSTAND: K2 is a numeric drop‑in; if `max_abs_err≠0` your K‑split order diverged (EX5's
      24‑bit lesson).  **(S1)**
- [ ] **1.5 — GREEN (K2 TP2).** WRITE: the TP2 split (down‑partial before the all‑reduce). RUN:
      `.venv/bin/python -m pytest tests/models/gemma4/test_amd_dense_mono.py::FfnTest -k tp2 -q`
      (torchrun‑backed 2‑rank fixture). LOOK/MEASURE: GREEN; the per‑rank partial summed on host
      equals the single‑rank output. UNDERSTAND: the rank partition (`FGeo(tp)`) and "partial before
      reduce" contract (§4.8).

### Milestone 2 — Integration of K2 only (prove the wiring)

- [ ] **2.1 — Eligibility.** WRITE: `dense_mono.py::_unsupported`, `row_caps`, bind `DenseMono` at
      model init (raise‑and‑list when unsupported). RUN, twice:
      `VLLM_ROCM_MONO_DECODE=1 vllm serve RedHatAI/gemma-4-31B-it-FP8-block -tp 1` on a good box, then with `-pp 2` (unsupported).
      LOOK/MEASURE: good box logs `widths [...]`; bad config **raises at startup listing the reason**.
      UNDERSTAND: fail‑closed at init, silent per‑step fallback later (§1.6 gates, `guards.py`).
- [ ] **2.2 — Custom‑op graph.** WRITE: register `gemma4_dense_mono_ffn` + `_tail`; build
      `DenseMonoGraph._forward_tp1` (stock attention + your K2). RUN:
      `VLLM_ROCM_MONO_DECODE=1 vllm serve RedHatAI/gemma-4-31B-it-FP8-block -tp 1 --async-scheduling` then send one request at
      concurrency 4, and one at concurrency 32. LOOK/MEASURE: server logs "layer graph at width 4"
      for the first; width‑32 shows **no** mono log (stock path). Compare the two responses' logits/
      text to a mono‑off run. UNDERSTAND: the width gate routes only eligible steps onto the graph,
      wider steps cost nothing (§4.9).  **(S3 partial)**
- [ ] **2.3 — Warm‑up + capture.** WRITE: `warmup_keys()`/`compile()` (or JIT registration) and
      confirm `widths` land in `cudagraph_capture_sizes`. RUN: start the server and watch the first
      decode step at a built width. LOOK/MEASURE: no multi‑second JIT stall on the first token
      (compare first‑vs‑second token latency in the log); grep the startup log for your capture
      sizes. UNDERSTAND: EX0's "2nd call is faster" at production scale — warm‑up hides JIT (§4.10).

### Milestone 3 — K1 (attention)

- [ ] **3.1 — qk‑norm + RoPE units.** WRITE: `helpers.rope`, `qk_norm` (butterfly order == aiter,
      from EX2/EX3). RUN: `.venv/bin/python -m pytest tests/models/gemma4/test_amd_dense_mono.py -k "qknorm or rope" -q`.
      LOOK/MEASURE: max‑abs error vs the stock `GemmaRMSNorm`/`rotary_emb` ≤ a bf16 ulp. UNDERSTAND:
      the per‑head norm + neox RoPE match stock exactly before you build the harder attention.
- [ ] **3.2 — RED.** WRITE: `AttnTest` asserting K1 == the stock qkv→qk‑norm→RoPE→attn→o_proj chain.
      RUN: `.venv/bin/python -m pytest tests/models/gemma4/test_amd_dense_mono.py::AttnTest -q`.
      LOOK/MEASURE: fails for a real (shape/precision) reason. UNDERSTAND: the attention contract is
      pinned before implementation.  **(S2)**
- [ ] **3.3 — GREEN (local K1).** WRITE: `attn_k1.py` for local layers: norm→Wqkv→qk‑norm→RoPE→
      kv‑write→windowed FP8 attention→o_proj (reuse EX3/EX4/EX5/EX7). RUN: same `AttnTest` command.
      LOOK/MEASURE: GREEN within the cosine/abs slack vs fp32 (print `1-cos`, `max_abs`; must be
      ≤ `1.25×` stock's and `2.0×` stock's, per `_k1_verdict`). UNDERSTAND: K1 isn't bit‑exact but
      must be *as close to fp32 as stock is* (§4.5).  **(S2)**
- [ ] **3.4 — Whole‑layer wiring.** WRITE: K1 as the whole‑layer op on local layers in
      `_forward_tp1`; global layers use stock attention + K2 (§4.7). RUN:
      `VLLM_ROCM_MONO_DECODE=1 vllm serve RedHatAI/gemma-4-31B-it-FP8-block -tp 1`, send a short prompt at concurrency 4, and a
      mono‑off run. LOOK/MEASURE: greedy completions **match** mono‑off token‑for‑token for a few
      hundred tokens. UNDERSTAND: full TP1 decode is now numerically equivalent end‑to‑end.  **(S3)**

### Milestone 4 — TP + safety + perf

- [ ] **4.1 — TP2 K1.** WRITE: the K1 rank partial + `_reduce`. RUN:
      `VLLM_ROCM_MONO_DECODE=1 vllm serve RedHatAI/gemma-4-31B-it-FP8-block -tp 2` + a greedy prompt vs mono‑off.
      LOOK/MEASURE: completions match mono‑off. UNDERSTAND: the TP2 split keeps the all‑reduce on the
      stock collective (§4.8).
- [ ] **4.2 — Safety/metrics (optional).** WRITE: fail‑stop (`_spin_bounded`) + Prometheus counters,
      or the in‑kernel peer all‑reduce. RUN: force a stall in a test (skip one `publish`) and start
      with metrics on. LOOK/MEASURE: the engine raises a clean error (not a hung process); `curl
      :<port>/metrics | grep monokernel` shows `steps`/`incidents`. UNDERSTAND: a hung peer becomes a
      surfaced error, never a silent bad token (§4.11).
- [ ] **4.3 — Accuracy evals.** WRITE: nothing. RUN: `lm_eval`/`vllm bench` for GPQA‑Diamond, AIME25,
      and a RULER NIAH 32K/128K point, mono‑on vs mono‑off. LOOK/MEASURE: deltas within noise (±1σ);
      put the table in the PR body. UNDERSTAND: numeric equivalence shows up as "within noise"
      (§5.2); a real gap means a stage diverged.
- [ ] **4.4 — Perf sweep.** WRITE: nothing. RUN: `vllm bench serve` (or `sglang.bench_serving`) 8k/1k,
      concurrency 1/4/8/16/32, mono‑on vs mono‑off on the same commit; and `rocprof`/torch‑profile one
      decode step. LOOK/MEASURE: TPOT/ITL/output‑tok‑s improve inside the row cap and stay within ±1 %
      above it; the profiled kernel count drops (e.g. ~40/layer → 2). UNDERSTAND: the win is at low
      concurrency and there is **no fallback cost** above the cap (§5.3).
- [ ] **4.5 — PR hygiene.** WRITE: the PR body (not‑a‑duplicate check, tests run, eval + perf tables,
      "AI assistance used", pinned AITER). RUN:
      `gh pr list --repo vllm-project/vllm --state open --search "Gemma mono decode"` and
      `pre-commit run --all-files`. LOOK/MEASURE: no duplicate PR; linters clean; a mono‑off run is
      byte‑identical to `main`. UNDERSTAND: AGENTS.md's accountability bar — the stock path must be
      provably unchanged with the flag off (Pitfall #10).

---

## PART 7 — PITFALLS & GOTCHAS (learned from the real PRs)

1. **Non‑scalar in a build key → shared binary.** A non‑scalar closure value isn't JIT‑keyed;
   two ranks/widths silently run one kernel. Keep build params in a frozen dataclass of scalars;
   carry `source_digest()`.  (`common/plan.py` docstring — this literally bit the team.)
2. **Register spills ⇒ kernel refused.** A mono kernel with scratch memory is rejected
   (`runtime.widths`). Use `fresh()` to stop LLVM hoisting lane math into whole‑kernel registers;
   bound your live mailbox polls (`POLL_MAX=12`).
3. **Reduction order is a numerical contract.** `butterfly` offset order must equal aiter's or
   RMSNorm rstd drifts a bf16 ulp and bit‑exact tests fail. Same for GEMM **K‑split order**
   (`split_total`) — any other grouping diverges on sums > 24 bits.
4. **Flag may not overtake data.** Always `publish()` (waitcnt+barrier, then the pair) before a
   consumer can poll; device vs system scope (`CM_DEV`/`CM_SYS`) must match the hand‑off.
5. **PIECEWISE graphs are out.** Decode mono needs FULL graphs; the dispatcher falls back under
   PIECEWISE (`dispatch.plan`). Document the serving flags.
6. **FP8 FNUZ↔OCP is not bit‑portable.** gfx942 weights ≠ gfx950 weights. Arch‑pin; port later.
7. **CU count must equal BLOCKS.** A partitioned GPU (CU != 256) deadlocks a persistent grid →
   the runner must refuse it.
8. **Pad rows.** CUDA‑graph pad rows have `batch_id == -1` (`row_live`); they must run harmlessly
   and their outputs must be zeroed (tests assert it).
9. **Don't wrap the replaced path in `@support_torch_compile`.** The mono graph *is* the compiled
   graph; the model's own layer loop must be plain Python there, or the dispatcher refuses it.
10. **Keep the stock path provably identical with the flag off.** Every PR demonstrates "nothing
    changes when off". Gate construction so `DenseMono` isn't even built unless the flag is set.
11. **Match the reference GEMM, not "an" FP8 GEMM.** Prefill may use the ASM GEMM; decode uses the
    Triton `gemm_afp4wfp4`/scaled‑mm. Pick the one your decode path uses as the golden ref.
12. **Global layers aren't worth full fusion first.** They're bandwidth‑bound; FFN‑only launch
    there (stock attention) is the high‑ROI first cut.

---

## APPENDIX A — Reference file map (what to read, in order)

Canonical **merged** DeepSeek‑V4.1 mono (PR #60397) — the design reference:

```text
vllm/models/deepseek_v41/amd/mono/README.md         # the execution-model + numerics spec (read first)
vllm/models/deepseek_v41/amd/mono_decode.py          # eligibility per layer/step, vLLM tensors -> kernel
vllm/models/deepseek_v41/amd/mono/runner.py          # scratch, epoch, peer memory, launches
vllm/models/deepseek_v41/amd/mono/layer.py           # K1/K2/FFN launch composition
vllm/models/deepseek_v41/amd/mono/common/{sync,ops,plan,peer_memory,mx}.py   # reuse verbatim
vllm/models/deepseek_v41/amd/mono/attention/{front,back,gemv,plan,device}.py # MLA attn (pattern for K1)
vllm/models/deepseek_v41/amd/mono/stages/{seam,moe,moe_plan,router,gemv,dims}.py
```

**Dense** Qwen3.8‑27B (PR #61573) — your actual template:

```text
vllm/models/qwen3_5/amd/dense_mono.py                # eligibility, compiled graph, custom ops, TP
vllm/models/qwen3_5/amd/mono/dense_ffn.py            # K2 (fork this for Gemma GeGLU)
vllm/models/qwen3_5/amd/mono/dense_gdn.py            # K1 for GDN (structural reference for your MHA K1)
vllm/models/qwen3_5/amd/mono/{helpers,sources}.py
vllm/models/qwen3_5/amd/mono/common/*                # same common/ as above
tests/models/qwen3_5/test_amd_dense_mono.py          # the test methodology to copy
```

Abstractions: `vllm/models/common/specialized_decode/*` (#61311) or `vllm/models/common/mono/*`
(#60683). Gemma‑4 source: `vllm/model_executor/models/gemma4*.py` (text backbone layer structure),
block FP8: `vllm/model_executor/layers/quantization/compressed_tensors/` + `fp8.py`, norms:
`.../layers/layernorm.py` (`GemmaRMSNorm`), RoPE: `.../rotary_embedding/base.py`.

## APPENDIX B — PR index

| PR | Title | Why read it |
| --- | --- | --- |
| **#60397** | DSv4.1 mono decode (MI355X) — **merged** | The canonical design + README spec |
| **#61082** | DSv4.1 mono on every backbone layer — **merged** | `--monokernel` switch, capture sizes, evals |
| **#61573** | **Qwen3.8‑27B dense mono** | Your template (dense K1+K2, TP1/TP2, custom‑op graph) |
| #60683 / #61435 | Kimi‑K3 monokernel + M1 | `common/mono/` abstraction, fail‑stop, 64‑row tiling |
| #60793 / #61568 | Kimi‑K3 mono MoE | one‑launch MoE; stacking launches |
| #60889 | GLM‑5.2 monokernel | first AMD user of `SpecializedDecode` (#61867 iface) |
| #61311 | Shared specialized‑decode abstractions | `DecodeSpec`/`SpecializedDecode` interface |
| #61867 | NVIDIA Kimi‑K3 CuTeDSL specialized decode | the pattern's origin (contrast, not ROCm) |
| #60731 / #60997 | DSv4.1 / Kimi‑K3 mono on **gfx942** | what a CDNA3 port takes |
| RFC #60904 | Monokernel RFC | the `--monokernel` / layout convention |

Find more: `gh pr list --repo vllm-project/vllm --state all --search "mono decode"`.

## APPENDIX C — External references

- Stanford Hazy Research, *"Look Ma, No Bubbles! Designing a Low‑Latency Megakernel for
  Llama‑1B"* — <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles> (grid‑persistent,
  on‑GPU instruction interpreter, global‑counter sync, software pipelining; 78 % vs 50 % HBM BW).
- Hazy Research, TP megakernel follow‑up — <https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main>
- Mirage Persistent Kernel (MPK) — <https://github.com/mirage-project/mirage> (automatic
  megakernel compiler; SM‑level graph, decentralized in‑kernel scheduler; 1.2–6.7× latency).
- ROCm HIP Cooperative Groups (grid.sync, hipLaunchCooperativeKernel) —
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html>
- AMD Matrix Cores / MFMA on CDNA (FP8 shapes, block scaling) —
  <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html>
- CDNA4 FP8 GEMM optimization (LDS swizzle, double buffering, 8‑wave ping‑pong) —
  <https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html>
- OCP 8‑bit FP (OFP8) spec — <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>
- AITER — <https://github.com/ROCm/aiter> ; vLLM AITER integration PR #14007.
- GPU arch specs (LDS/CU, wavefront) — <https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>
- Target checkpoint — [`RedHatAI/gemma-4-31B-it-FP8-block`](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-block)
  (`Gemma4ForConditionalGeneration`, text backbone `gemma4_text`; block‑FP8 weights + dynamic
  group‑128 activations, bf16 KV; 60 layers, 5:1 sliding:full, sandwich norms, QK‑norm, GeGLU,
  final logit soft‑cap 30.0).

---

*Scope note: the mono‑decode PRs referenced here are an active, mostly‑unmerged line of work;
file paths and the enable switch (`VLLM_ROCM_MONO_DECODE` → `--monokernel` →
`--additional-config decode_backend`) are converging under RFC #60904 — re‑check the head of the
tree before you start. Build order, the execution model, the mailbox/peer‑memory design, the
bit‑exact‑vs‑stock methodology, and the Gemma fusion plan are the stable parts.*
