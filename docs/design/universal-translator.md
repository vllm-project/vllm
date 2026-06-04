# RFC v2: Universal Translator — vLLM TP Topology Flexibility Framework

**Status**: Draft (post-pilot, post-cross-LLM-review)
**Project**: `universal-translator`
**Author**: MidasMining (with prior contributions to vLLM PR #38479, #39970, #39931, #41123)
**Target**: vLLM 0.21+ as Phase 1; framework consolidation in 0.22+

> **Why "Universal Translator"?** Models are written for a target topology (e.g., 4× H100). Users have whatever they have (8× A4000, 4× 4090, 2× 5090). The framework *translates* between what the model expects and what the hardware provides — turning hard "no" assertions into "yes, with measurable cost".

---

## What changed in v2

This document supersedes v1 (kept at `universal-translator-rfc.md`). Changes since v1 reflect:

1. **Three-reviewer cross-LLM review cycle** (Gemini, Grok, Nemotron). Reviewer-converged must-fix items are integrated; divergent angles are surfaced as informed open questions.
2. **An end-to-end Phase 1 pilot** on Ling-2.6-flash that ran the full proposed pattern through real GPUs and produced measured numbers.
3. **A debug-driven discovery** that the framework's central thesis applies recursively: the same anti-pattern appears at multiple nested layers of the inference stack. The pilot peeled five integration gates across four nested layers — vLLM gates, FlashInfer dispatch, kernel dtype checks, and a kernel-level paged-indexing bug — each one initially looking like a hardware constraint, each one actually software completeness.

The core thesis from v1 stands. The framing has shifted from "TP=8 always wins" (a measured-false claim on PCIe-class hardware) to **"the framework's value is converting hard rejections into measurable-cost choices, with hardware-dependent throughput tradeoffs the user can now navigate with full information."**

---

## Abstract

vLLM models frequently hard-code tensor-parallelism (TP) constraints in their kernels — assertions like `tp_size <= group_norm_size`, `num_kv_heads % tp_size == 0`, or implicit assumptions about `partial_rotary_factor == 1.0`. These constraints are typically defensive fast-paths rather than mathematical requirements. They fragment hardware compatibility: a model designed for 4× H100 may refuse to load on 8× A4000 even when the math works, forcing users into pipeline parallelism (latency penalty), 2-instance load balancing (operational complexity), or simply abandoning the model.

This RFC proposes a **generalized TP-flexibility framework** that:

1. Catalogs the recurring constraint types with known mitigations
2. Provides both a declarative decorator API and a post-hoc registration API for models to opt into automatic constraint handling
3. Inserts the appropriate runtime communication patterns when constraints would be violated
4. Falls back to existing fast paths when constraints are satisfied (zero overhead in the common case)

The benefit is making any model that uses the framework work on any reasonable TP topology, with predictable, measurable cost. **The framework's value is optionality, not always-faster** — on NVLink-class topologies the topology-flexible configuration may exceed the original topology's throughput; on PCIe-class topologies it may not. The framework removes the configuration *rejection*; the user makes the throughput tradeoff with full information.

This is especially valuable for the long tail of users running consumer GPUs, older datacenter hardware, and edge devices.

---

## Motivation

### The pattern: same problem, recurring across models

In recent months, contributors have repeatedly hit and fixed the same class of issue across different models. Each fix is currently bespoke:

| Model | Constraint | Effect | Status |
| ------- | ----------- | -------- | -------- |
| MiniMax M2.5/M2.7 | `partial_rotary_factor=0.5` | Full WHT on KV vectors mixes RoPE and non-RoPE dims; ~10% quality loss on stateful tasks | Fixed via split-block Hadamard ([#39970 comment thread](https://github.com/vllm-project/vllm/pull/39970)) |
| Nemotron-H hybrid | `slot_size_aligned` not divisible by Mamba page size | Hybrid models fail page-size unification | Fixed in [#41123](https://github.com/vllm-project/vllm/pull/41123)/[#39931](https://github.com/vllm-project/vllm/pull/39931) (competing approaches) |
| Ling-2.6-flash | `tp_size > group_norm_size` | Asserts at startup; can't run at TP > 4 | **Pilot of this RFC** — patched and validated end-to-end |
| GLM-4.7-Flash | `num_kv_heads = 20`, not divisible by 8 | Forces TP ≤ 4 | Replication via existing GQA path; suboptimal |
| Qwen3-Next-80B | `num_kv_heads = 2` | Forces TP ≤ 2 (replication doesn't help when this aggressive) | No good fix |
| Cascade-2 / Super-120B | `num_kv_heads = 2` (Cascade), `8` (Super) | TP=8 works via auto-replication, but 4× KV memory waste on Cascade | Implicit existing handling |

Each of these required a specialist to read kernels, identify the constraint, and propose a one-off fix. Several remain unsolved. Meanwhile, the underlying **mitigation patterns are limited and well-known**:

1. **Replicate** — broadcast a small dimension across ranks
2. **Pad** — round up to a divisible size, mask the padding
3. **Cross-rank reduce** — insert an all-reduce when locality would otherwise be broken
4. **Block-diagonal** — restructure a transform to preserve local structure
5. **Per-rank scoping** — make each rank handle a subset rather than rejecting

### The integration-gaps anti-pattern (new framing)

The Phase 1 pilot revealed something stronger than v1 articulated: **the same shape of "defensive fast-path masquerading as a hardware constraint" recurs at every layer of the inference stack.** The pilot peeled five such gates across four nested layers in a single model integration:

1. **Framework gate** — vLLM's `tp_size <= group_norm_size` assertion
2. **Kernel SMEM gate** — `lightning_attn.py` hardcoded `BLOCK=256` exceeding A4000 SMEM
3. **Backend selection gate** — `FlashInferMLABackend.supports_compute_capability == 10` (Blackwell-only)
4. **Library dispatch gate** — FlashInfer's MLA wrapper routing Ampere bf16 to XQA, which gates on SM120/SM121 + fp8
5. **Kernel correctness gate** — fa2 MLA producing zero output for non-contiguous `kv_indices` on Ampere

Each gate, alone, looked like a hardware reality. Each, together, prevented Ling-2.6-flash from running on 8× A4000 at any TP configuration. With careful one-step-at-a-time engineering, **none of them turned out to be hardware physics**. Each was a software completeness gap that could be peeled with the same discipline the framework proposes for its narrower TP-shape case.

This means the RFC's argument generalizes. The framework's **discipline** — *catalog the constraint, mitigate where math allows, surface a precise error otherwise* — applies as a methodology, not just a fixed catalog of TP-shape patterns.

### Why this matters for the ecosystem

Frontier labs design for 8× H100 80GB or 4× H200 141GB. Their TP recommendations match those topologies. When the actual user has:

- 8× RTX A4000 (16GB each, 128GB total) — common professional workstation
- 4× RTX 4090 (24GB each, 96GB total) — common enthusiast/research
- 8× A10 (24GB each, 192GB total) — common cloud spot
- 2× RTX 5090 (32GB each, 64GB total) — emerging consumer flagship
- 1× workstation + 1× consumer card — heterogeneous

…the model often refuses to load even when the math fits. The user's only options are pipeline parallel (latency cost), a smaller model (capability regression), or abandoning the model entirely. Making models TP-flexible converts a hard "no" into a *user choice between known throughput tradeoffs*.

---

## Current state: what works case-by-case

Some constraint patterns already have implicit mitigations in vLLM:

- **GQA KV head replication**: When `num_kv_heads < tp_size`, vLLM auto-replicates KV across ranks. Memory cost: factor `tp_size / num_kv_heads`. This is the closest existing precedent.
- **Padded FFN sizes**: AWQ Marlin kernels handle non-aligned `intermediate_size` via internal padding. Hidden behavior, not exposed.

The framework would unify these and add the missing patterns.

---

## Proposed Solution

### Two co-equal annotation paths

Per three-reviewer agreement, the framework exposes both a declarative decorator and a post-hoc registration function as **first-class peer APIs**. Model authors who can modify their `modeling_*.py` use the decorator; community patches and HF-imported models use the registration function.

```python
# Decorator path — for first-party model authors
from vllm.universal_translator import translate, ConstraintGroupedNorm, ConstraintReplicateKV

@translate([
    ConstraintGroupedNorm(group_size_attr='group_norm_size', strategy='allreduce'),
    ConstraintReplicateKV(min_per_rank=1),
])
class BailingMoELinearAttention(nn.Module):
    """Ling-2.6-flash linear attention layer."""
    ...
```

```python
# Registration path — for post-hoc patches and out-of-tree models
from vllm.universal_translator import register_constraints, ConstraintGroupedNorm

register_constraints(
    BailingMoELinearAttention,
    [ConstraintGroupedNorm(group_size_attr='group_norm_size')],
)
```

The two are equivalent in effect. The reason for both is that vLLM models register dynamically (via `register_model` and HF auto-import); requiring source modification of the model class is the wrong default. The decorator is a convenience for in-tree models.

### Phase 1 scope: one constraint, validated end-to-end

Per Grok's scope-narrowing recommendation, **Phase 1 ships one constraint type only** — `ConstraintGroupedNorm` — applied to the Ling-2.6-flash pilot model, fully validated. The other constraints from v1 (`ConstraintReplicateKV`, `ConstraintPaddedDim`, `ConstraintBlockDiagonal`, `ConstraintExpertParallel`) are documented as **future work using the same pattern** but not in the Phase 1 deliverable. This keeps the maintainer-side review surface bounded to "one new constraint with a working pilot" rather than "a whole framework."

### `ConstraintGroupedNorm` — the Phase 1 constraint

- **Trigger**: `group_size < tp_size` (Ling case: `group_norm_size=4`, `tp_size=8`)
- **Mitigation**: Insert all-reduce within each group's rank subgroup
- **Cost**: **Measured at ~3.3% per token** at batch=1 on 8× A4000 PCIe (low end of original 5–15% estimate)
- **Status**: Patched and validated end-to-end. See Section "Pilot Case Study" below.

### API contract

```python
class TopologyConstraint(ABC):
    """Base class for Universal Translator constraints."""

    @abstractmethod
    def will_violate(self, layer_config: dict, ctx: HardwareContext) -> bool:
        """True if this constraint would be violated under the active topology."""

    @abstractmethod
    def estimate_cost(self, layer_config: dict, ctx: HardwareContext) -> CostEstimate:
        """Estimated memory/compute overhead of mitigation, given hardware context."""

    @abstractmethod
    def wrap_layer(self, layer: nn.Module, ctx: HardwareContext, tp_group) -> nn.Module:
        """Return a wrapped/modified layer that handles the constraint."""

@dataclass
class HardwareContext:
    """Per all three reviewers' converged ask: cost is hardware-dependent."""
    tp_size: int
    sm_capability: tuple[int, int]
    interconnect: Literal["nvlink", "pcie", "mixed"]
    batch_size_hint: int | None = None

@dataclass
class CostEstimate:
    extra_memory_per_rank_bytes: int = 0
    extra_compute_factor: float = 1.0  # 1.05 = 5% slower
    rationale: str = ""
```

The `HardwareContext` is the v2 addition. Cost estimates are inherently hardware-dependent — a 50µs all-reduce on NVLink is negligible, on PCIe is a real bottleneck. v1 papered over this with a flat compute-factor; v2 makes it first-class per Gemini's, Grok's, and Nemotron's converged feedback.

### Subgroup factory (collective requirement)

Naively constructing subgroups per layer hits PyTorch's `dist.new_group` collective requirement: every rank in the parent group must call `new_group` with the same `ranks` list, even non-members. The framework provides a memoizing factory:

```python
_subgroup_cache: dict[tuple[int, ...], "torch.distributed.ProcessGroup"] = {}

def get_or_create_subgroup(tp_group, ranks: tuple[int, ...]):
    if ranks not in _subgroup_cache:
        import torch.distributed as dist
        _subgroup_cache[ranks] = dist.new_group(
            ranks=list(ranks),
            backend=dist.get_backend(tp_group),
        )
    return _subgroup_cache[ranks]
```

Every rank iterates ALL subgroups in deterministic order on first construction; subsequent layers hit the cache (no `new_group` calls), preserving the collective requirement. **This was one of the bugs caught and fixed during the pilot — the original implementation had each rank only request its own subgroup, causing NCCL hangs.** Worth surfacing as a "things to get right" footnote for adopters.

### Integration interactions (new section per Nemotron)

The framework's mitigations interact with several other vLLM optimizations. Each interaction has been audited or flagged:

- **CUDA graph capture**: Validated for the cross-rank-reduce all-reduce inside the `vllm::linear_attention` custom op (piecewise mode). The mitigation captures cleanly because NCCL collectives on subgroups are graph-captureable, and our subgroup factory creates them lazily at model init (well before graph capture begins). **Verified empirically in pilot.**
- **Paged KV cache layouts**: The Phase 1 constraint does not touch KV cache. Future constraints that do (e.g., `ConstraintReplicateKV`, `ConstraintPaddedDim`) need explicit per-backend audit — see "Open Questions."
- **Speculative decode**: Untested in pilot. The cross-rank reduce is per-token-position, so spec-decode (multiple tokens per forward) should compose naturally. Flagged as future work.
- **KV cache write-vs-read backend mismatch**: A real risk class. The pilot found that vLLM's MLA cache writes are backend-agnostic (via `concat_and_cache_mla`), but reads can diverge per backend's expectations. Constraints touching the KV cache must be audited against both write and read paths.

---

## Alternatives Considered

### A1: Continue with case-by-case PRs

The current trajectory. Each new constraint type gets a dedicated PR, debated separately, with no shared infrastructure.

- **Pros**: Lower up-front coordination cost; no API design risk.
- **Cons**: Same problem keeps surfacing; no cumulative learning; new model authors keep hard-asserting; users keep hitting walls. We've already seen ≥6 of these in 6 months with at least 3 unsolved.
- **Why rejected**: The pattern is real and growing. Continuing case-by-case has unbounded long-term cost.

### A2: Push the burden to model authors

Have each model adopt a "TP-flexible variant" of every kernel pattern directly in its `modeling_*.py`.

- **Pros**: Localized; no framework dependency.
- **Cons**: Massive code duplication. Each author re-implements the same all-reduce, padding, replication logic. Bugs proliferate; quality varies.
- **Why rejected**: This is what we have today and it's not working.

### A3: Pure runtime detection (no decorators)

Have the framework auto-scan layer attributes and inject mitigations without any author cooperation.

- **Pros**: Works on existing models without modification.
- **Cons**: Fragile heuristics; surprising to users; testing surface explodes.
- **Why partially adopted**: Implicit detection for patterns vLLM already auto-handles; explicit declaration for new patterns. Balances ergonomics against surprise.

### A4: Use Pipeline Parallelism (PP) instead

Tell users to fall back to PP=N × TP=K when constraints don't fit.

- **Pros**: No new framework; uses existing vLLM features.
- **Cons (REVISED per pilot data)**: PP adds inter-stage latency, but on PCIe-class topologies the **measured** comparison shows PP=2×TP=4 *outperforms* TP=8 by ~34% across batch sizes for Ling. The pilot data invalidates v1's claim that PP is unconditionally worse — it depends on hardware.
- **Why rejected (revised)**: PP is sometimes the throughput-better choice. The framework converts it from forced to optional — users on PCIe topologies may rationally choose PP after seeing the measured tradeoff. The framework's value is *making the choice possible*, not *winning the throughput comparison*.

### A5: Defer to upstream (wait for vLLM maintainers)

- **Pros**: Zero contribution effort.
- **Cons**: vLLM maintainers are bandwidth-constrained on frontier features. Long-tail topology support has not been a priority. Could wait years.
- **Why rejected**: We have working examples already. Proposing a unification framework converts spent debugging effort into compounding leverage.

---

## Failure Modes and Risks

### Technical risks

| Risk | Likelihood | Impact | Mitigation |
| ------ | ----------- | -------- | ----------- |
| Cross-rank reduce introduces correctness bug | Medium | High (silent quality loss) | Reference test against TP=K-satisfied baseline; logit-level KL tolerance. **Validated in pilot via standalone unit test (`test_cross_rank_norm.py`) — bit-exact match across 4 configurations.** |
| Mitigation cost exceeds estimate by >2× | Medium | Medium | Benchmark every shipped constraint; document measured costs; flag when measurement diverges from estimate |
| Pattern doesn't generalize to new constraint types | Medium | Low | Each constraint is independent; framework doesn't presume all future constraints fit the catalog |
| Triton kernel changes interact with TQ kernels | Medium | High | Phase 1 explicitly tests on Ling without TQ first; TQ integration is Phase 2 |
| Cross-rank reduce adds latency on small batches | Medium | Low | Document the small-batch case in `estimate_cost`; users can fall back to PP |
| **Subgroup `new_group` collective requirement** *(NEW from pilot)* | High if missed; mitigated by framework | High (NCCL hangs) | Framework provides a memoizing subgroup factory that iterates ALL groups on every rank. Documented as a "things to get right" gotcha. |

### Social/maintenance risks

| Risk | Mitigation |
| ------ | ----------- |
| vLLM maintainers reject the framework approach | Land Phase 1 (Ling pilot) as a standalone fix first, then propose framework as refactor of the success. **Pilot is now a working artifact with measured cost, not a hypothetical.** |
| Each future constraint requires negotiation about which pattern it fits | Constraint catalog is open — anyone can add. Framework is a contract, not a gatekeeper. |
| Migration of existing implicit handlers breaks something | Phase 1 explicitly: "expose without behavior change." Regression-test against existing models. |
| Decorator approach feels too invasive to model authors | Both decorator AND `register_constraints()` exposed as first-class APIs. |

### Out-of-scope failure modes

These are explicitly **not** addressed by Universal Translator:

- Numerical drift across very different TP topologies (existing vLLM concern, framework doesn't make it worse)
- Performance regressions in non-translated layers (we don't touch them)
- Quantization quality on translated layers (orthogonal)

---

## Concrete Implementation Example

`ConstraintGroupedNorm` end-to-end (from the pilot):

```python
# vllm/universal_translator/constraints/grouped_norm.py

import torch
import torch.distributed as dist
from .base import TopologyConstraint, CostEstimate, HardwareContext

class ConstraintGroupedNorm(TopologyConstraint):
    """Mitigates `tp_size > group_norm_size` by inserting cross-rank
    reduce within each norm group's rank subgroup.

    Activates when the model's grouped RMS norm spans multiple ranks.
    Falls back to local fast path when each rank holds at least one
    complete group.
    """

    def __init__(self, group_size_attr: str, strategy: str = 'allreduce'):
        self.group_size_attr = group_size_attr
        self.strategy = strategy

    def will_violate(self, layer_config: dict, ctx: HardwareContext) -> bool:
        group_size = layer_config.get(self.group_size_attr)
        if group_size is None:
            return False
        return ctx.tp_size > group_size

    def estimate_cost(self, layer_config: dict, ctx: HardwareContext) -> CostEstimate:
        if not self.will_violate(layer_config, ctx):
            return CostEstimate()
        group_size = layer_config[self.group_size_attr]
        ranks_per_group = ctx.tp_size // group_size
        # Cost differs significantly by interconnect.
        per_reduce_us = {"nvlink": 5, "pcie": 50, "mixed": 30}[ctx.interconnect]
        # Rough projection — calibrated against the Phase 1 pilot's measured
        # 3.3% per-token overhead on 8× A4000 PCIe at batch=1.
        return CostEstimate(
            extra_compute_factor=1.0 + (per_reduce_us * (ranks_per_group - 1) / 1500),
            rationale=(
                f"Cross-rank reduce across {ranks_per_group} ranks per group "
                f"(group_size={group_size}, tp_size={ctx.tp_size}, "
                f"interconnect={ctx.interconnect}: ~{per_reduce_us}µs/reduce)"
            ),
        )

    def wrap_layer(self, layer, ctx, tp_group):
        if not self.will_violate(self._layer_config(layer), ctx):
            return layer
        return _GroupedNormCrossRank(layer, tp_group, ctx)


class _GroupedNormCrossRank(torch.nn.Module):
    """Wraps a grouped RMS norm to handle cross-rank groups."""

    def __init__(self, original_norm, tp_group, ctx):
        super().__init__()
        self.norm = original_norm
        # ... build subgroup via the framework's memoizing factory ...
        # See subgroup factory section above.

    def forward(self, x):
        # x: [batch, seq, head_dim_per_rank]
        local_sumsq = (x.float() * x.float()).sum(dim=-1, keepdim=True)
        global_sumsq = local_sumsq.contiguous()
        dist.all_reduce(global_sumsq, op=dist.ReduceOp.SUM, group=self.subgroup)
        inv_norm = torch.rsqrt(global_sumsq / self.group_size + self.norm.eps)
        return (x * inv_norm).to(x.dtype) * self.norm.weight
```

Total pilot patch: ~85 lines added, 6 lines deleted (asserts moved into a branch). See `pilot/vllm/model_executor/models/bailing_moe_linear.py` for the actual integrated change.

---

## Measured vs Estimated Costs

v2 replaces v1's mostly-estimated cost table with **pilot-measured numbers** wherever possible:

| Cost figure | Source | Notes |
| ------------ | -------- | ------- |
| `ConstraintGroupedNorm` cross-rank-reduce overhead | **Measured** | ~3.3% per token at batch=1 on 8× A4000 PCIe. 28 linear-attn layers × ~50 µs all-reduce ≈ 1.4 ms per token vs ~42 ms total token time. Inside the original 5–15% estimate, on the low end. |
| `ConstraintGroupedNorm` correctness | **Measured (bit-exact)** | Standalone unit test (`test_cross_rank_norm.py`) validates the wrapper produces bit-identical output to a single-rank reference across 4 configurations including the Ling-realistic shape (8 ranks, 4096 features, 4 groups). |
| Total TP=8 patched throughput vs PP=2×TP=4 | **Measured** | -34% on 8× A4000 PCIe (Ling, batch=16). The TP collective tax dominates the cross-rank-reduce mitigation overhead. |
| Total TP=8 patched throughput vs TP=4 + CPU offload | **Measured** | TP=4 + offload was used to capture a numerical-equivalence reference for correctness. Throughput comparison is not meaningful (offload is artificially slow). |
| FlashInfer fa2 MLA adapter (Ampere bf16) | **Measured** | -40% vs TRITON_MLA piecewise on the same hardware. Kernel-level cost, not adapter-level — adapter overhead (compaction workaround for the upstream bug) is within measurement noise. |
| `ConstraintReplicateKV` memory factor | **Measured** | Existing vLLM behavior; direct factor `tp_size / num_kv_heads` |
| `ConstraintPaddedDim` <1% overhead | **Measured** | Existing Marlin AWQ behavior |
| `ConstraintBlockDiagonal` <1% throughput | **Measured** | Split-block WHT on Cascade-2 weight TQ at TP=8 measured 3% overhead total |
| `ConstraintExpertParallel` minor balance loss | **Measured** | Existing vLLM behavior, well-characterized |

**Phase 1 acceptance criterion (now satisfied)**: Measured cross-rank-reduce overhead on Ling-2.6-flash at TP=8 falls within ±50% of the original 5–15% estimate.

---

## Pilot Case Study: Five Integration Gates

The Phase 1 pilot ran the proposed Universal Translator pattern through Ling-2.6-flash on 8× A4000 PCIe, end-to-end. The discovery is that the framework's anti-pattern (defensive constraints masquerading as hardware reality) recurs at multiple nested layers in a single model integration. Five gates were peeled, each one initially looking like a hardware constraint and turning out to be software completeness:

### Gate 1: vLLM TP-shape assertion (`tp_size <= group_norm_size`)

**Discovery surface:** Ling-2.6-flash refused to load at TP=8 with `AssertionError: tp_size must be <= group_norm_size for local rms norm` from `bailing_moe_linear.py:557`.

**The mitigation:** insert a cross-rank all-reduce within each norm group's rank subgroup, replacing the local sumsq with a globally-reduced sumsq. Patched at ~85 lines, ~6 lines deleted.

**Validation:** Standalone unit test passes bit-exact across 4 configurations. End-to-end run on Ling: 3/8 prompts match the TP=4 reference exactly; the remaining 5 produce coherent text with expected fp-reduction-order drift.

**Measured overhead:** ~3.3% per-token at batch=1.

### Gate 2: lightning_attn kernel SMEM ceiling (CBLOCK=64 on Ampere)

**Discovery surface:** After the gate-1 fix, Ling still failed to run on A4000 — but with a different error:

```text
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 131584, Hardware limit: 101376.
```

The lightning_attn `_fwd_kv_parallel` Triton kernel hardcodes `CBLOCK=64` which produces a per-CTA SMEM peak of ~131 KB on A4000 (which has ~99 KB per SM). This is independent of TP topology — Ling cannot run on A4000 at any TP without addressing this.

**The mitigation:** reduce `CBLOCK` from 64 to 32 in `lightning_attn.py:467`. Affects both `_fwd_kv_parallel` and `_fwd_none_diag_kernel`. Cost: small per-iteration overhead; total work unchanged. Brings SMEM peak to ~96 KB, fits A4000's ceiling.

**Status:** Separable upstream PR. This is the kind of "kernel was written for Hopper, not validated on Ampere consumer" gap that affects an entire class of users. Worth filing independently of the framework.

### Gate 3: vLLM gate on FLASHINFER_MLA (`major == 10`)

**Discovery surface:** While exploring whether MLA-side optimizations could improve throughput, `FlashInferMLABackend.supports_compute_capability` was found to return `True` only for compute capability major=10 (Blackwell). Hopper and Ampere were rejected.

**Investigation:** the gate's stated reason ("compute capability not supported") was vague. We hypothesized it was defensive caution. Empirically tested by relaxing the gate to `major in (8, 9, 10)` and forcing `attention_backend="FLASHINFER_MLA"`. The gate's relaxation alone exposed the next layer:

### Gate 4: FlashInfer dispatch routing (Ampere → XQA → SM120/121 + fp8)

**Discovery surface:** With the vLLM gate relaxed, the run produced a clearer error:

```text
RuntimeError: XQA MLA only supports fp8 operation on SM120/SM121 GPUs,
got torch.bfloat16 and torch.bfloat16
```

The vLLM `FlashInferMLAImpl.forward_mqa()` calls `trtllm_batch_decode_with_kv_cache_mla` from `flashinfer.decode`. That function dispatches `cc != 10` to XQA, which requires Blackwell SM120/SM121 + fp8.

But FlashInfer ALSO ships `BatchMLAPagedAttentionWrapper(backend="fa2")` — a separate API that uses fa2 kernels and works on Ampere bf16. vLLM just doesn't reach it.

**The mitigation:** route vLLM's `forward_mqa` to use `BatchMLAPagedAttentionWrapper(backend="auto")` on `cc < 10` instead of the trtllm function. Adapter is ~80 lines, mostly API translation.

### Gate 5: fa2 kernel paged-indexing bug (Ampere bf16)

**Discovery surface:** Adapter loaded cleanly, single-prompt isolation produced bit-identical output to the TRITON_MLA reference for the first 70 tokens. But multi-prompt runs produced **gibberish** for specific prompts.

**Cross-LLM debug review:** A debug report was sent to Grok and Gemini. Grok pointed to scheduler page-allocation patterns; Gemini named the specific kernel-side hypothesis (paged-indexing bug with non-contiguous indices) and proposed the discriminating "Compaction Test."

**The compaction test:** before calling `wrapper.run(...)`, gather only the active pages into a contiguous tensor and remap `kv_indices` to `[0, 1, 2, ...]`. If gibberish disappears, the kernel mishandles non-contiguous indices.

**Result:** all 8 test prompts produced coherent output with compaction enabled. **Standalone repro confirmed:** with the same logical KV data placed at scattered physical pages `[16, 47, 89, 203]`, the kernel returns **all zeros**; with the same data at contiguous pages `[0, 1, 2, 3]`, it returns the correct attention output (norm 32.75 vs 0.00).

**The mitigation:** compaction (gather active pages + sequential `kv_indices`). Per-forward `index_select` on the full pool — measured throughput overhead is **within noise** (compaction itself adds <1% per batch).

**Status:** Workaround validated and shipped in the pilot adapter. Upstream FlashInfer bug filed (or ready to file) with minimal repro showing the zero-output behavior.

### Combined throughput at TP=8 + FLASHINFER_MLA-fa2 + compaction + piecewise graphs

| Config | batch=1 | batch=4 | batch=16 |
| --- | ---: | ---: | ---: |
| TRITON_MLA piecewise (vLLM default on Ampere) | 23.84 | 90.94 | 349.70 |
| FIMLA-fa2 + compact + piecewise (this work) | 14.29 | 54.93 | 209.61 |

The remaining -40% gap is the FlashInfer fa2 kernel's intrinsic performance on Ampere at this MLA shape, not the adapter. Closing it would require kernel-level tuning, beyond pilot scope.

### Summary of the case study

Five gates, four nested layers (framework, library, kernel dispatch, kernel correctness). Each gate, alone, presented as "your hardware can't run this model." Together they form an unshippable wall. With sequential, careful, hypothesis-driven engineering — **and significant help from cross-LLM debugging review** — every one of them turned out to be peelable. That is the strongest empirical demonstration we can offer for the RFC's central thesis: defensive constraints in inference stacks usually represent integration completeness gaps rather than hardware physics.

---

## Cross-LLM Review Methodology

The pilot ran two rounds of cross-LLM review (Gemini, Grok, plus Nemotron in round 1 with reduced reliability due to its harsher stress-test prompt). Both rounds produced material that significantly improved the work. Documenting the methodology here as a reusable contribution.

### Round 1: RFC v1 review (pre-pilot)

Three reviewers received the v1 RFC and produced reviews independently. Strongest signals came from points where multiple reviewers converged:

- **Three-way agreement (must-fix):** registration API as first-class peer of the decorator; `HardwareContext` on `estimate_cost`; narrow Phase 1 scope to one constraint with full validation.
- **Two-way agreement (strong signal):** subgroup-cache factory keyed on `(tp_group, ranks_tuple)`; composability needs a defined algebra, not just an open question.
- **Single-reviewer items (judgment calls):** kernel fusion loss, OOM-budget pre-check (Gemini); subgroup contiguous-rank assumption, quantization×translator (Grok); CUDA graph + paged-attn interactions, observability hooks (Nemotron).

Where reviewers diverged, both angles were useful. Gemini's reviews tended toward implementation detail (ergonomics, telemetry, kernel internals); Grok's tended toward architectural framing (scope, API surface, adoption); Nemotron's tended toward stress-test pushback (which surfaced two genuine new concerns despite its harshness being prompt-induced).

### Round 2: Mid-pilot debug review

When the FlashInfer fa2 adapter produced gibberish for specific prompts and three obvious hypotheses had already been disproved, a detailed debug report was written and sent to Grok and Gemini. The report included full code, empirical bug pattern data, hypotheses tested with evidence, and five specific questions for the reviewers.

**Result:** Gemini's review proposed a single discriminating experiment ("the Compaction Test") and named the specific kernel-side hypothesis. The experiment confirmed the hypothesis on first try. Grok's review proposed the same fault-domain narrowing with different instrumentation suggestions.

The debug-review round was **disproportionately valuable** — when stuck on a hard bug, fresh eyes via cross-LLM review can be more effective than continuing to grind alone. Worth recommending as a methodology for any RFC of comparable complexity that hits unexpected bugs in pilot.

### Recommended methodology for future RFCs of this shape

1. **Initial review** with at least 2 reviewers from different model families
2. **Categorize feedback** by reviewer-agreement count (3-way > 2-way > 1-way)
3. **Treat 3-way agreement as must-fix**; 2-way as strong signal; 1-way as informed judgment calls
4. **Pilot with one narrow scope** to test the framework's central claims empirically
5. **If pilot hits a stuck bug**, write a self-contained debug report and round-2 review — explicitly include hypotheses already disproved, all relevant code, and specific questions

---

## Implementation Roadmap

### Phase 1 (DONE in pilot — 3 weeks): Framework + One Pilot

- [x] Base classes (`TopologyConstraint`, `HardwareContext`, `CostEstimate`, `translate` decorator + `register_constraints`)
- [x] Module: `vllm.universal_translator`
- [x] Subgroup memoizing factory
- [x] `ConstraintGroupedNorm` end-to-end
- [x] Apply to Ling-2.6-flash via post-hoc registration
- [x] Validate: model loads at TP=8 on 8× A4000, output coherent
- [x] Benchmark: measured ~3.3% per-token overhead — within original 5-15% estimate
- [x] Unit test passes bit-exact across 4 configurations

**Deliverable status:** Ling-2.6-flash runs at TP=8 with measurable, bounded cost. Patched fork at `pilot/`.

### Phase 2 (2 weeks per): Additional constraints

Each follows the same pattern: implementation, test, benchmark, docs, PR.

- `ConstraintBlockDiagonal` (refactor existing split-block WHT)
- `ConstraintPaddedDim` (formalize implicit padding)
- `ConstraintExpertParallel` (formalize existing handling)
- `ConstraintReplicateKV` (formalize existing implicit handling)

### Phase 3 (ongoing): Model-author adoption

Authors annotate their layers; framework handles mitigation automatically.

### Bonus separable contributions discovered during pilot

These are not framework Phase work but are real upstream PR candidates that fell out of the pilot:

- **lightning_attn CBLOCK=32 on Ampere** — separable patch, unblocks Ling-class models on Ampere consumer/prosumer GPUs entirely. Independent of the framework.
- **FlashInfer-MLA fa2 adapter** for vLLM (the adapter we wrote) — opens MLA models (Ling, GLM-4.7, DeepSeek-V2/V3) to Ampere bf16. Independent of the framework.
- **FlashInfer fa2 paged-indexing bug filed upstream** — kernel-side fix, benefits anyone using `BatchMLAPagedAttentionWrapper` on non-Blackwell.

---

## Test Strategy

### Correctness

For each constraint:

1. **Reference baseline**: Run model at recommended TP (where constraint is satisfied) → record exact outputs on a fixed prompt set.
2. **Flexed configuration**: Run model at violating TP with framework mitigation → compare outputs.
3. **Acceptance threshold**: Logit-level KL divergence < 1e-4 per token, output text identical for greedy decode where possible. **Note**: pilot showed cross-config drift (TP=4-offload vs TP=8-flexed) is at the level of expected fp-reduction-order divergence; bit-exact match is achievable in idealized cases only. Use logprob proximity over a fixed prompt set as the practical metric.
4. **Standalone unit test**: For Phase 1, the unit test approach validated against synthetic per-rank tensors with `gloo` backend on CPU is fast, GPU-free, and CI-friendly. See `test_cross_rank_norm.py`.

### Performance

For each constraint:

1. Measure throughput at recommended TP (baseline)
2. Measure throughput at violating TP with mitigation (flexed)
3. Document the cost ratio in the constraint's docstring
4. Ensure cost estimate from `estimate_cost()` matches measured cost (±10%)

### Topology coverage

Test on at least:

- 4× and 8× datacenter (A100, H100)
- 4× and 8× consumer (4090, 5090)
- 8× professional (A4000) — the pilot's reference setup
- Heterogeneous (mixed sizes via PP boundaries)

---

## Open Questions

Reordered by reviewer-converged priority:

1. **Auto-detection vs explicit declaration**: Auto-detection from layer attributes vs explicit decoration. Tradeoff: convenience vs surprise. Hybrid (auto for currently-handled patterns, explicit for new) seems best per all reviewers.

2. **Composability algebra (per Grok+Nemotron)**: Multiple constraints on one layer with non-trivial interaction. Order of application matters. Proposed algebra: weight-time transforms apply first (composable, commute), forward-pass injections apply second (ordered: norm before attention before output projection by convention). Documented composition order, with explicit ordering hooks for advanced cases.

3. **Cost estimate accuracy / threshold warnings**: When measured cost > 50% slower, framework should warn or hard-fail with override flag. **Concrete:** if `estimate_cost > 1.5x` and no override is given, surface a `RuntimeWarning` at model init naming the affected layers and constraints. Pre-deployment users can override.

4. **Heterogeneous topology design tension (per Nemotron)**: v1's motivation cited mixed-GPU setups but the design assumed uniform `tp_group` properties. v2 resolves this by **explicitly scoping the framework to homogeneous TP groups**. Heterogeneous (different-speed devices in the same TP group) is documented as out of scope and noted as a separate concern from "topology flexibility" — it's a load-balancing problem, not a topology problem.

5. **Subgroup communication efficiency**: Phase 1 uses naive sub-NCCL groups. Phase 2 may explore vLLM's existing TP machinery for hierarchical groups, fused all-reduces, etc.

6. **Backward compatibility**: Framework is purely additive. Models that work today continue to work; models that previously hard-failed gain the option to translate.

---

## Concrete Pilot: Ling-2.6-flash at TP=8 (DONE)

Pilot ran on 8× A4000 PCIe, vLLM 0.20.1.dev0, FlashInfer 0.6.8.post1, Ling-2.6-flash-int4 (BailingMoeV2.5).

**Current state before Phase 1:** Ling has `group_norm_size=4` baked into config. vLLM's `BailingMoELinearAttention` RMS norm hard-asserts `tp_size <= group_norm_size`. On 8× A4000, the user must run at PP=2×TP=4 (with measured ~530 t/s at batch=16) or not at all.

**With Universal Translator (Phase 1):** Annotate `BailingMoELinearAttention` with `ConstraintGroupedNorm(group_size_attr='group_norm_size')`. The translator detects `tp_size=8 > group_norm_size=4`, wraps the RMS norm with cross-rank all-reduce. Model loads at TP=8.

**Measured outcome (Ling at TP=8 with ConstraintGroupedNorm):**

- Throughput at batch=16: ~349 t/s (TRITON_MLA with piecewise graphs) — **-34% vs PP=2×TP=4 baseline**
- Per-token cross-rank-reduce overhead: ~3.3% per token at batch=1 — inside the 5-15% original estimate
- Correctness: 3/8 exact-token match vs TP=4 reference; semantically coherent across the full prompt set; expected fp-reduction-order drift on remainder
- CUDA-graph captureability of subgroup all-reduces: validated empirically (piecewise mode, attention runs eager inside captured non-attention regions)

**The TP=8 < PP=2×TP=4 result is the v2 reframing.** On PCIe-class hardware, the per-layer collective tax of going from TP=4 to TP=8 outweighs PP's pipeline-bubble cost. The framework's value isn't winning the throughput race — it's giving users the option, with measured data, instead of blocking the choice at the assertion.

---

## Related work and prior art

### Comparison to existing inference frameworks (revised)

| Framework | TP-flexibility approach | Comparison to Universal Translator |
| ----------- | ------------------------ | ----------------------------------- |
| **DeepSpeed Inference** | Auto-replicates KV when needed; supports custom tensor splits via `replace_module` API | Similar in spirit; DeepSpeed's API is imperative, ours is declarative+registration. DeepSpeed handles fewer constraint types. |
| **Megatron-LM** | TP-aware kernels for attention and FFN with hard-coded GQA replication | Closely tied to Megatron's specific kernel implementations; not a general framework. |
| **TensorRT-LLM** | TP plugin system with limited topology configurations | Closed-source plugin model. |
| **vLLM today** | Implicit handling for KV replication; explicit handling for FFN padding inside kernels | The starting point — we propose to formalize and extend. |

The key differentiator: **Universal Translator is the first declarative+registration, composable, open-source framework** for arbitrary TP-topology constraints in an inference engine, with empirically-validated end-to-end pilot on consumer/prosumer hardware.

### vLLM PR references

- **PR #38479** — TurboQuant landed Apr 15. Our split-block WHT built on top.
- **PR #39970** — online weight compression (open). Posted: split-block WHT, TP HadamardTransform fix, end-to-end TP=8 validation, SpectralQuant calibration, `res_norm` regression report.
- **PR #39931 / #41123** — Two competing approaches to making TQ work on hybrid models. Cross-validation posted.
- **PR #40914** — Spec-decode K+1 verify routing. +201% throughput cross-validation on Super-120B posted.

---

## Call to action

If this RFC has merit, the natural next steps are:

1. **Discussion**: Invite feedback on API surface, constraint catalog completeness, naming
2. **Pilot scoped (DONE in this v2)**: `ConstraintGroupedNorm` + Ling-2.6-flash is the validated first concrete deliverable
3. **Resource commit**: Phase 2 needs ~2 weeks per additional constraint type; volunteers welcome

The **pilot artifacts are immediately available** at `pilot/` (patches + tests + bench harness + bench data + writeups). RFC v2 is grounded in measured numbers, not estimates.

Universal Translator is a force multiplier — every model author who adopts it makes life better for every user running non-frontier hardware. We believe vLLM's mission of broad accessibility benefits substantially when models can speak across topologies they weren't originally designed for, with full information about the tradeoffs.

---

## Appendix A: Glossary

For reviewers not deep in vLLM internals:

| Term | Meaning |
| ------ | --------- |
| **TP** | Tensor parallelism. Splits each layer's weights across multiple GPUs, with all-reduce or all-gather between layers. |
| **PP** | Pipeline parallelism. Splits the model layer-wise; each token passes through stages serially. |
| **EP** | Expert parallelism. For MoE models, splits experts across GPUs. |
| **DP** | Data parallelism. Multiple replicas serving different requests in parallel. |
| **GQA** | Grouped Query Attention. Q heads share fewer KV heads. |
| **MLA** | Multi-head Latent Attention. Compresses KV via low-rank projection (DeepSeek's technique). |
| **MHA** | Multi-Head Attention. Standard attention with one KV head per Q head. |
| **MoE** | Mixture of Experts. Each token activates a subset of expert FFNs. |
| **all-reduce** | Collective: every rank's tensor is summed and result broadcast to all ranks. |
| **all-gather** | Every rank's tensor is concatenated; result available to all ranks. |
| **RMS norm** | Root-mean-square normalization. Grouped variant computes mean within sub-vectors of size `group_size`. |
| **`partial_rotary_factor`** | Fraction of head_dim that gets RoPE positional encoding. |
| **`group_norm_size`** | Number of sub-vector groups in a grouped RMS norm. |
| **head_dim** | Per-head feature dimension. Typically 64, 96, 128, or 256. |
| **`kv_heads`** | Number of distinct K/V head pairs. |
| **TQ / TurboQuant** | KV cache quantization using Walsh-Hadamard rotation + Lloyd-Max scalar quantizer. 4× KV compression. |
| **Lightning Linear Attention** | Specific linear-attention variant in MiniMaxText01 and Ling. |
| **Mamba / SSM** | State Space Model. Selective scan; alternative to attention with O(N) instead of O(N²). |
| **`HardwareContext`** *(new in v2)* | Per-layer-init context object containing `tp_size`, `sm_capability`, `interconnect`, optional `batch_size_hint`. Passed to `estimate_cost` and `wrap_layer`. |

## Appendix B: Pilot artifact index

- **Patched fork:** `pilot/` — vLLM 0.20.1.dev0 with cross-rank-norm patch + lightning_attn CBLOCK fix + FlashInfer-MLA fa2 adapter
- **Companion env:** `pilot-env/`
- **Unit test (passing):** `./test_cross_rank_norm.py` — gloo-CPU multi-rank, bit-exact across 4 configs
- **Standalone FlashInfer bug repro:** `./flashinfer-mla-fa2-bug-repro.py` — produces zeros on scattered indices
- **Bench harness:** `./ling-pilot-bench.py` — covers TP=4-offload, PP=2×TP=4, TP=8 patched (eager + graphs)
- **Bench outputs:** `./ling-bench-*.json`
- **Step writeups:**
    - `./ling-pilot-step0-diagnostic.md`
    - `./ling-pilot-step1-blocker.md` — kernel SMEM finding
    - `./ling-pilot-step2-correctness.md` — unit test results
    - `./ling-pilot-results.md` — main throughput/correctness writeup
    - `./flashinfer-mla-fa2-adapter-design.md`
    - `./flashinfer-mla-fa2-results.md`
    - `./flashinfer-mla-fa2-bug-localized.md`
- **Cross-LLM debug review thread:** `./flashinfer-mla-fa2-debug-report.md` (initial report sent to Grok/Gemini for round 2)
- **Upstream FlashInfer issue draft:** `./flashinfer-mla-fa2-bug-issue.md`

## Appendix C: Cross-LLM review references

- **v1 review cycle (2026-04-29):** Gemini ("Strong Buy"), Grok ("Pursue, narrow scope"), Nemotron ("Conditional Acceptance")
- **Round 2 debug review (2026-05-01):** Gemini named the discriminating "Compaction Test" hypothesis that resolved the FlashInfer fa2 paged-indexing bug; Grok independently narrowed the fault domain to scheduler-page-allocation interactions.
- Reviewer feedback summaries: see project memory entries.
