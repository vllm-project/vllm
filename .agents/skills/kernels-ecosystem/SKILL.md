---
name: kernels-ecosystem
description: Find existing implementations by LLM primitive: GQA, MLA, sparse attention, GDN, Mamba2, KDA, GEMM, MoE, AllReduce, AllToAll and supporting operations. Compare kernel families, implementation languages and origins without having to crawl other frameworks. Use when microbenchmarking an operator or tuning model performance, to surface alternative kernel implementations that might improve performance.
---

# Kernels Ecosystem

> **Approximate index captured 2026-09-14.** This is a quick reference, not
> an exhaustive or always-up-to-date mapping of available kernels. Implementations and
> compatibility may have changed since then. Verify current source before
> integrating a candidate.

Start with **the operation you need**, then compare its implementation
families. Each entry identifies the kernel, its language, where it comes
from, and its known contract. The catalog answers availability questions;
source links are evidence and integration entry points, not a research task.

## Find implementations by primitive

| Kernel type | Available families at a glance | Implementation catalog |
| --- | --- | --- |
| GQA / MQA / MHA | FlashAttention; native FlashInfer; TRTLLM FMHA/XQA/generated; CuTe modular/prims-ts; cuTile | [GQA: prefill and decode](references/gqa.md) |
| Dense MLA | FlashMLA; FlashInfer CUDA/CUTLASS/CuTe/cuTile; TRTLLM CuTe/generated; TokenSpeed CuTe | [MLA: prefill and decode](references/mla.md) |
| Sparse attention | Sparse MLA consumers; block-sparse/HCA; QSA/MSA; separate logits/indexer/top-k implementations | [Sparse attention and indexers](references/sparse-attention.md) |
| Gated DeltaNet / GDN | FLA-derived Triton; FlashInfer CuTe chunk/recurrent; distinct in-tree CuTe pipelines; cuDNN/Cake | [GDN: prefill, decode and MTP](references/gdn.md) |
| Mamba2 | Mamba-derived Triton SSD/SSU; FlashInfer CuTe SSD; native CUDA SSU/MTP and replay | [Mamba2: SSD and state update](references/mamba2.md) |
| Kimi Delta Attention / KDA | FLA-derived Triton; CuTe chunk/recurrent; FlashKDA; native CUDA/PTX and Helion alternatives | [KDA: prefill, decode and MTP](references/kda.md) |
| Dense / grouped / skinny GEMM | CUTLASS, CuTe DSL, cuTile, TRTLLM-GEN, DeepGEMM, Marlin and specialized low-latency/low-bit families | [GEMM by format and algorithm](references/gemm.md) |
| MoE compute | CUTLASS, CuTe, TRTLLM-GEN, DeepGEMM, Marlin, cuTile, MonoMoE/BGMV and fused distributed MoE | [Expert implementations](references/moe.md) |
| AllReduce / fused collectives | vLLM custom AR; TRTLLM CUDA/MNNVL; FlashInfer CuTe LL/HT/BT; PCIe IPC; fused GEMM/collective families | [Collectives](references/collectives.md#allreduce) |
| EP AllToAll | One-/two-sided NVLink; DeepEP throughput/low-latency/V2; NCCL-EP, NIXL-EP, MoRI and other provider transports | [Expert dispatch/combine](references/collectives.md#expert-alltoall) |
| Attention-state exchange | Helix DCP and Ulysses IPC; not expert-token AllToAll | [Attention-state collectives](references/collectives.md#attention-state-exchange) |
| Norm / quant / RoPE / sampling | Native CUDA, CuTe DSL, cuTile and selected substantial fused-operation families | [Supporting primitives](references/supporting.md) |

The linked tables contain concrete callable/class names and pinned sources.
They are organized by phase, format or algorithm, **not by inference
framework**. For example, GDN prefill lists the FlashInfer CuTe implementation
once; vLLM, SGLang and TRTLLM importing it do not add three more candidates.

## What counts as an implementation

- Keep distinct kernel algorithms/backends and substantial reusable families.
  A framework can contain real kernel source: TokenSpeed MLA or TRTLLM
  CuTe MLA belongs here even though it lives in an inference repository.
- Consolidate known vendored copies and re-exports under their implementation
  origin. An integration location is not necessarily the author or owner.
  Preserve a material adaptation as a variant rather than claiming all
  same-named kernels are identical.
- Omit framework import wrappers, backend/linear adapter classes, selector
  aliases, fallback plumbing and one-off framework-local Triton kernels.
  Retain substantive upstream Triton families such as FLA GDN and Mamba SSD;
  this is not a blanket exclusion of Triton.
- Group tile/template specializations and constituent pipeline stages.
  Do not count launch configurations, cache helpers or each exported alias as
  independent implementation choices.

**CuTe DSL is Python DSL, not C++ CUTLASS/CuTe.** Triton, Gluon, TileLang,
Helion and cuTile are also distinct technologies. For example, FlashInfer's
`grouped_mm/cute_sm120_*` names denote C++ CuTe/CUDA, not Python CuTe DSL.

## Selecting a candidate

Filter by GPU, phase, shape, dtype, cache/state layout and distributed
topology. Prefill is not decode; sparse scoring is not an attention consumer;
MoE compute is not EP transport. An unrecorded restriction is unknown, not
implied support. A backend set for one dtype does not apply to every dtype.

This is a curated implementation inventory,
not an export census or a performance ranking. No kernels were built or
timed. Use [kernel-microbenchmark](../kernel-microbenchmark/SKILL.md) for
correctness/performance comparisons and
[triton-kernel-writing](../triton-kernel-writing/SKILL.md) when implementing
a new Triton kernel.

## Re-index

### How this report was made

The initial review used shallow clones (`--depth 1 --no-tags`) of
`flashinfer-ai/flashinfer`, `NVIDIA/TensorRT-LLM`, `sgl-project/sglang` and
`lightseekorg/tokenspeed`, plus the current vLLM worktree.

Source-directory inventories, public exports, backend registries, build/JIT
manifests and dependency declarations were cross-checked to find candidate
families. Review followed dispatch entries into device implementations to
establish language, origin, phase and restrictions; directory names alone
were not treated as evidence of a backend. Independent repository reviews
were then consolidated into primitive tables, removing import wrappers,
shared-provider duplicates and isolated framework-local Triton helpers.

Validation checked pinned paths against local Git objects, Markdown links
and anchors, table structure and lint. Catalog-only checks asked a reader to
name concrete alternatives and their origins without reopening source.
No GPU compilation, correctness testing or performance measurement was
performed; availability and measured suitability remain separate claims.

### Citation order

For each implementation, prefer:

1. **The kernel's own upstream project and device source**, when known.
2. **vLLM's implementation or vendored copy**, when upstream source cannot
   be identified or verified.
3. **Another framework's source**, only when neither above covers the
   implementation.

For example, cite FlashAttention source for FlashAttention, not SGLang's
import or vendored copy; vLLM's FlashAttention fork is the next choice.
Follow the current checkouts' dependency pins, source headers and build
manifests to locate upstream revisions. Prefer canonical project URLs and
the actual kernel file over an adapter, registry, README or repository root.
If only integration evidence is available, prefer vLLM's integration and
label it as such rather than presenting it as device source.

Keep a framework-specific source citation when it contains a material
adaptation not covered upstream or in vLLM. An upstream project link does
not establish that it implements that adaptation. Avoid redundant copies
of the same family; retain an additional API/contract link only when needed
to explain the indexed variant. Verify upstream paths before replacing
citations; explicitly label moving-branch links when no revision is known.
GitHub can resolve fork commits through upstream URLs. Verify the recorded
upstream sync point or release, not just URL resolution; cite fork-only
changes under the fork that actually maintains them.

### Refresh procedure

1. **Choose scope and revisions.** Refresh a primitive or the full inventory.
   Use the existing source links to identify old SHAs and choose target
   revisions before reviewing. Reuse the retained clones;
   if absent, shallow-clone the repositories above under the Git-ignored
   `.kernel-ecosystem-sources/` directory. Verify ignore rules before cloning
   and preserve any local changes. Do not reset the user's vLLM worktree.
2. **Review the source delta without moving the checkout.** For each external
   clone, fetch an explicit target ref and compare it with the recorded SHA:

   ```bash
   git -C <clone> status --short
   git -C <clone> fetch --depth 1 origin <target-ref>
   git -C <clone> rev-parse FETCH_HEAD
   git -C <clone> diff --name-status <old-sha> <new-sha>
   git -C <clone> ls-tree -r --name-only <new-sha>
   git -C <clone> show <new-sha>:<source-path>
   ```

   Fetch missing old objects if needed. Direct snapshot comparison works
   without a merge base in a shallow clone. Use the recorded target SHA,
   not a moving `FETCH_HEAD`, in subsequent review and citations.
3. **Repeat the inventory checks, not just the diff.** Enumerate current
   operation directories, exports, registries and JIT/build manifests to
   catch new backends, renamed modules and removed implementations. Inspect
   changed kernel bodies and eligibility checks; follow provider imports
   only far enough to identify their origin. If dependency device source is
   not reviewed, retain the explicit integration-evidence label.
4. **Reconcile by primitive and origin.** Apply the inclusion rules above.
   Update concrete callables, language, phase, formats and constraints in the
   relevant topic file. Add material algorithms/adaptations, consolidate
   copies, and remove retired candidates from the refreshed snapshot.
   For example, a new SGLang import of FlashInfer GDN does not warrant a row;
   a new FlashInfer CuTe GDN algorithm does. Keep private coverage notes out
   of the skill rather than restoring per-repository export inventories.
5. **Update evidence together with claims.** Use
   `https://github.com/<owner>/<repo>/blob/<sha>/<path>` for files and `tree`
   for directories, following the citation order above: kernel upstream,
   then vLLM, then another framework. Resolve sources from the checkouts'
   dependency pins and attribution, not a newer release's assumed layout.
   Update the overview only after review;
   never bulk-replace SHAs without rechecking the corresponding claims.
   For a partial refresh, update only the relevant entries and links; do not
   advance the whole index's capture date for untouched rows.
6. **Validate the refreshed report.** Check each citation with
   `git -C <clone> cat-file -t <sha>:<path>` (`blob` or `tree` as linked).
   For upstream sources not retained locally, verify the target path/ref
   through GitHub or the provider's source host; do not substitute an
   unverified upstream URL for a known vLLM implementation.
   Check relative links/anchors, reference definitions, table widths and the
   repository's existing Markdown lint; keep topic files below 300 lines.
   Confirm research clones stay ignored and no runtime files changed.
   Finally, answer from the catalog alone: "Which CuTe GQA/MLA alternatives
   exist?", "Which GDN prefill implementations have different origins?",
   and "Which EP transports are distinct from MoE compute?" For a targeted
   refresh, substitute equally concrete questions about that primitive.
   Report the reviewed scope, additions/removals and validation results;
   do not imply GPU validation unless it was actually performed.
