# 30-Minute Presentation: dLLM Integration in vLLM

**Audience**: vLLM maintainers (technical, deep vLLM/LLM inference; familiar with traditional bidirectional DLMs; less familiar with block/window semi-causal dLLMs.)  
**Supporting doc**: [pitch-dllm-integration.md](pitch-dllm-integration.md)  
**Total**: ~30 minutes talk + buffer for Q&A.

---

## Suggested Structure and Timing

| Section | Duration | Purpose |
|--------|----------|--------|
| 1. Title + one-liner | ~1 min | Set context. |
| 2. Trending dLLMs, competitors, and benchmarks | ~4 min | Why now: trending models, competitor support, papers comparing to vLLM. |
| 3. Motivation: why block-based dLLMs matter | ~3 min | Bridge from “diffusion LLMs” to “block-based = one step, one forward.” |
| 4. Background: traditional vs block-based dLLMs | ~5 min | Align on terminology; clarify what we’re integrating. |
| 5. Why vLLM / problem statement | ~2 min | Fit with continuous batching and existing engine. |
| 6. Proposed architecture (with diagrams) | ~10 min | Core of the pitch; use mermaid from pitch doc. |
| 6b. Prefix caching and per-model attention masks | ~2 min | How prefix cache is bounded; effect of variable attention masks. |
| 7. Design dilemmas and choices | ~3 min | Show tradeoffs and decisions. |
| 8. Scope, next steps, and wrap-up | ~2 min | Clear boundaries and follow-ups. |
| **Q&A** | As needed | Buffer after 30 min. |

---

## Section-by-Section Notes

### 1. Title + One-Liner (~1 min)

- **Slide**: Title + “Blocked masked diffusion LLMs (dLLMs): one diffusion step = one worker iteration.”
- **Say**: We’re proposing first-class support for block-based diffusion-style LLMs (SDAR, LLaDA2.0, Fast-dLLMv2, WeDLM, etc.) so they run inside vLLM with the same step-based execution and batching as today.

---

### 2. Trending dLLMs, Competitors, and Benchmarks (~4 min)

- **Goal**: Answer “why now?” with concrete names, competitor reality, and benchmark narrative.
- **Trending dLLMs** (slide or bullet list):
  - **Mercury 2** (Inception Labs): diffusion-based reasoning, **1,000+ TPS**, ~5× faster than leading speed-optimized AR models.
  - **LLaDA2.0 / LLaDA2.1** (Ant Group): **~800–900 TPS** on coding (HumanEval+, BigCodeBench); configurable speed vs quality.
  - **WeDLM** (Tencent): diffusion + standard causal attention; strong speedups in reported benchmarks.
  - **SDAR, Fast-dLLMv2**, and related block-based dLLMs; major labs (e.g. Google) exploring diffusion-style decoding.
- **Competitor support**: **Three major vLLM competitors already support or ship dLLM-style inference**: **SGlang** (e.g. LLaDA 2.0, block-diffusion text; chunked-prefill–style integration), **Ollama** (local/API for diffusion-style models), **LMDeploy** (dLLM deployment path). Users can today choose these stacks; vLLM support keeps the ecosystem aligned.
- **Benchmarks vs AR on vLLM**: Papers and reports already compare **dLLM inference to classic AR LLMs deployed with vLLM**. In several settings dLLMs **win by a large margin**—**several× to an order of magnitude** in throughput or latency. Examples: **WeDLM** ~3×–10× over vLLM-optimized AR; **LLaDA2.1** ~800+ TPS on coding; **dInfer** / **FlashDLM** report 2–3× over vLLM AR, 10–12× over prior dLLM systems. First-class dLLM in vLLM lets users get these gains inside the same engine.
- **Say**: “So the models are trending, competitors already support them, and the benchmarks comparing dLLM to vLLM-served AR are already in the literature—sometimes with order-of-magnitude wins. We want that to run inside vLLM.”

---

### 3. Motivation: Why Block-Based dLLMs Matter (~3 min)

- **Goal**: Explain why *block-based* diffusion decoding is relevant for an inference engine, not “diffusion LLMs” in the abstract.
- **Points to hit**:
  - Traditional fully bidirectional diffusion (MDLM, LLaDA1.0, Dream): full-sequence refinement, multiple steps over full length → harder to map to “one forward per step” and expensive at length.
  - Block-based dLLMs: fixed-size block per step (e.g. 32 tokens), mask-then-fill within the block, model returns “committed tokens” + “next block input” → **one step = one forward**, variable tokens per step (like spec decode).
  - Benefit: better quality/speed tradeoffs, parallel decode within the block, and a clean fit with vLLM’s execution model.
- **Avoid**: Deep dive into training or specific papers; keep it at “what the inference contract is.”

---

### 4. Background: Traditional vs Block-Based dLLMs (~5 min)

- **Goal**: Get everyone on the same page: “traditional bidirectional DLM” vs “block/window semi-causal dLLM.”
- **Suggested slide** (or two):

  - **Traditional (fully bidirectional)**: Full sequence visible; iterative refinement over all positions; attention can be full bidirectional; step = full sequence → doesn’t map to one forward per step.
  - **Block-based (semi-causal / window)**: Step = one block of fixed size; some positions MASK; model fills and decides commit + next block; attention is block/window; one step = one forward.

- **Concrete names**:
  - Traditional: MDLM, LLaDA1.0, Dream.
  - Block-based: SDAR, LLaDA2.0, LLaDA2.1, Fast-dLLMv2, WeDLM.
- **One diagram**: Optional simple sketch (or reuse from pitch) showing “one block in, committed + next block out.”
- **Emphasis**: We are only proposing to support the **block-based** family; the abstraction (LOOKAHEAD_SIZE, committed tokens, next-step input) is what we need from the engine’s perspective.

---

### 5. Why vLLM / Problem Statement (~2 min)

- **Goal**: Make the “why vLLM” case and state the integration problem clearly.
- **Points**:
  - Users will want to serve these models in production; a single stack (vLLM) is better than custom servers.
  - Block decoding is batch-friendly: one forward per step, variable tokens per step → continuous batching and existing scheduling apply.
  - **Problem**: Map “one diffusion step” to “one scheduler step” so that KV cache, prefix cache (with documented bounds), and metrics (TPF) work without special engine logic.
- **Keep it short**: Maintainers already know vLLM’s strengths; focus on “what we need to add and what we don’t change.”

---

### 6. Proposed Architecture (~10 min) — Use Mermaid from Pitch

This is the main technical section. Walk through the pitch doc’s diagrams in order.

- **Current vLLM step flow** (~2 min): Show the sequence diagram (scheduler → worker → scheduler; draft tokens on Request, sent via SchedulerOutput). Stress: next-step input lives on the Request; worker is stateless.
- **dLLM step flow** (~3 min): Same sequence diagram with dLLM: worker returns `DllmStepOutput` (committed + next-step input); scheduler appends committed, stores next-step input on Request, sends it next time as `scheduled_dllm_input_tokens`. Emphasize: **same pattern as spec decode** (scheduler as source of truth).
- **Data flow** (~2 min): Flowchart “where state lives”: Request holds `next_dllm_input_token_ids`; worker receives it via SchedulerOutput and returns DllmStepOutput; scheduler updates Request and KV (commit + free tail).
- **One step, one forward** (~2 min): Simple flowchart: one step → one forward → variable committed (0..K) + fixed next block (LOOKAHEAD_SIZE). No extra forwards; only the *meaning* of the output is dLLM-specific.
- **First decode step** (~1 min): When sequence length < LOOKAHEAD_SIZE: right-pad with MASK so “to decode” is on the right; one sentence on why (training convention).
- **Prefix caching and per-model attention masks** (~2 min): Dedicated slide or two. (1) **Prefix validity**: Only **committed** tokens form reusable prefix; the last LOOKAHEAD_SIZE positions are in flight until committed → prefix valid length ≤ num_computed_tokens. (2) **Per-model attention masks**: Each dLLM has its own mask (block causal, banded, full block). We only cache KV for committed positions; the mask defines how the *current block* attends to the prefix at forward time. It does **not** change what we cache or invalidate—same model, same committed prefix ⇒ same KV, safe to reuse. (3) **Caveat**: If a model's mask is step- or block-dependent, prefix reuse may need phase alignment; in practice shared prefix is usually prompt-only. (4) **Fully bidirectional prefill**: Some works use a **fully bidirectional prefill mask**; then KV at every position depends on the full prompt and **prefix caching is impossible**—must disable and run prefill per request. Point to pitch §11 for the full paragraph.

If time is short, trim “current vLLM” to one minute and give the extra time to “dLLM step flow” and “data flow.”

---

### 7. Design Dilemmas and Choices (~3 min)

- **Goal**: Show that key design questions were considered and decided in a way that fits vLLM.
- **Use the table from the pitch doc**; for each row, give a 30–60 second summary:
  - **Next-step input on Request**: Consistency with spec decode and preemption; worker stays stateless.
  - **Right-pad with MASK**: Matches model training (fill right-side masks).
  - **Worker validates lengths**: One place for the contract; scheduler trusts shape.
  - **Model-level dLLM only**: One model per instance; no mixing in one batch.
  - **Reuse metrics**: TPF = generation tokens that step; no new APIs.
  - **Prefix cache**: Valid up to committed length; document LOOKAHEAD boundary. See also dedicated §6b / pitch §11 on prefix caching and per-model attention masks.
- **Optional**: Mention alternatives briefly (e.g. “we considered storing next-step input in the worker but that would break preemption / multi-worker”).

---

### 8. Scope, Next Steps, and Wrap-Up (~2 min)

- **In scope**: Scheduler/worker contract, DllmStepOutput, Request and SchedulerOutput fields, first-step convention, KV tail free, tests. One model per instance; dLLM when a dLLM model is loaded.
- **Out of scope**: Per-architecture model code (SDAR/LLaDA2.0/etc. attention masks), training.
- **Next steps**: Reference spec and plan in `specs/001-dllm-integration/`; implementation phases (contracts, scheduler, worker, engine, tests); optional follow-up on prefix cache and observability.
- **Closing**: “We’re proposing this abstraction so block-based dLLMs run as first-class citizens with one step = one forward and existing batching; we’d like your feedback on the contract and placement of state.”

---

## Presentation Tips for This Audience

- **Assume vLLM expertise**: Don’t explain continuous batching or KV cache basics; do explain *where* dLLM state lives and *how* it flows (scheduler vs worker).
- **Assume some DLM familiarity**: Briefly contrast “traditional full-sequence diffusion” vs “block-based”; spend time on the **block abstraction** (LOOKAHEAD_SIZE, commit, next input) and how it maps to the engine.
- **Diagrams**: Use the mermaid from the pitch doc (export as images or render in slides). The sequence diagram (dLLM step flow) and the “where state lives” flowchart are the most important.
- **Be explicit on “one step, one forward”**: Maintainers care about scheduling and performance; stress that we’re not adding extra forwards or breaking the step invariant.
- **Q&A**: Likely topics: prefix cache boundary, interaction with chunked prefill, per-model attention and prefix reuse, multi-worker, and concrete model support (SDAR/LLaDA2.0). Have the spec and data-model.md open for detail.

---

## Optional Slides (If Time Allows)

- **Comparison with speculative decoding**: Both have “variable tokens per step” and “next-step input from scheduler”; table: spec decode = draft tokens + verify; dLLM = committed + next block. Same scheduler pattern.
- **KV cache**: One slide on “allocate LOOKAHEAD_SIZE per step; commit k; free (LOOKAHEAD_SIZE - k) tail slots.”
- **Prefix caching (detail)**: If Q&A goes deep, use pitch §11 (prefix valid up to committed length; per-model masks don't change cache rules; step-dependent mask caveat; **bidirectional prefill ⇒ prefix caching impossible**, disable for those models).
- **Contract summary**: One slide with DllmStepOutput fields and SchedulerOutput.scheduled_dllm_input_tokens; point to contracts/dllm-step-contract.md.

---

## Materials Checklist

- [ ] Slides (or deck) with mermaid diagrams from pitch-dllm-integration.md.
- [ ] Pitch doc (pitch-dllm-integration.md) for shared link or handout.
- [ ] Spec and plan (spec.md, plan.md) for “where to read more.”
- [ ] contracts/dllm-step-contract.md and data-model.md for deep-dive Q&A.
