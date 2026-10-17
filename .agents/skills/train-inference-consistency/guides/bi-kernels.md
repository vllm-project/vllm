# BI kernels: which kernel each operator uses and what it costs

Read [the numerical contracts](../workflow/README.md) first, then build candidates from this page. Each row gives the kernel used under BI=1, the cost of invariance, known pitfalls and directions already tested dead. Numbers come from Nemotron-H (GB200 single node, 4 GPUs) and DS4-Flash (GB200 DP8/EP8); only conclusions are given. When the user asks to optimize only the explicit Top-K/index overhead, do not expand into whole-model tuning; switching DSL does not relax the contract.

## By operator

| Operator | BI=1 kernel / approach | Cost and upper bound | Pitfalls |
| --- | --- | --- | --- |
| Attention (FA4) | `FlashAttnFixedSplit`: 32 splits × 640 keys, `MAX_SEQ_LEN` 16384; with FP8 Q/KV the KV dequantization path is fixed | The 16384 limit exists for invariance; exceeding it must raise, not silently change the split. MLA matrix absorption changes rounding; mathematical equivalence does not give train-inference bitwise equivalence | On SM100, do not declare support by bypassing the selector's "backend does not support BI" check. Implement and verify the numerical contract in the backend first, then let the selector declare it; BI0 keeps the default fast path |
| Dense GEMM (FP8 / NVFP4 linear) | fixed split-K; under Nemotron BI small-M does not use CuTe GEMM, FP8 bmm pinned to cutlass, W4A16 uses Humming | tcgen05 hand-written skinny GEMM: a 64-row UMMA tile under "TileM must not depend on M" caps tensor core utilization at 50%; large M with narrow N is about 2× cuBLAS; loading the skinny bi_gemm `.so` on DS4 BI=1 does not change throughput | The small-M gain comes from cluster split-K over DSMEM (S=8 at K=4096), removing the producer/MMA warp `try_wait` peek (~0.5 µs), and `st.async` sent directly to the owner slot (~0.5 µs at M=64); stage count, spin `test_wait` and S=16 gave nothing, do not retry |
| mHC (DS4 TileLang) | BI=1 pins `split_tokens=1` | no extra cost | Pitfall only under BI=0: `n_splits` varies with `num_tokens` and is a `@tilelang.jit` specialization parameter, so every unseen prefill size recompiles for 10–14 s, all DP ranks stall, and every BI=0 comparison is polluted. BI=0 must load the `fixed_splitk` patch and check `tl_after_bench=0`. `mhc_post` spreading h over 128 blocks is numerically right but 40% slower; gains can only come from fusion |
| MoE (NVFP4 W4A16, Nemotron) | CuTe-DSL W4A16 modular MoE (takes the router's top-k); one CTA runs the whole K (`_W4A16_K_TILE=256` fixed), no split-K, no atomic | Per-row K accumulation order is the same under both tile sets; whether a single FP32 element is bitwise identical across tcgen05 instruction shapes (M128/M256, N8/N128) is undocumented and must be measured | flashinfer_cutedsl and Humming are both modular; moving R3 routing capture into the kernel (monolithic, #44214 style) saves nothing |
| MoE (FP8 block, DS4) | backend locked to DEEPGEMM (open source, built-in BI mode, the deployment kernel; FlashInfer TRT-LLM MoE ships as precompiled cubins that cannot be modified or verified as BI); `flashinfer_cutlass` rejects the block-fp8 QuantKey, `triton` resolves to BATCHED_TRITON which does not declare BI → a2a locked to deepep low-latency | BI=1 vs BI=0 gap floor is 33%, 83% of it non-overlappable deepep LL communication (86 serial a2a round trips per decode step); BI=1 does 14% less GPU work than BI=0 but takes 28% more wall clock (overlap ratio 1.33 vs 1.97) | DBO measured −39% (split ubatches read expert weights twice per layer, and weight reads are already at 84% of HBM roofline); TP2/DP4 −17%, TP4/DP2 −28%, DP8 is optimal for BI=1 too; deterministic top-k + histogram radix-select together +6.7% (alone +2.3% / +0.1%, super-additive, measure together); the biggest lever is DeepEP LL over MNNVL instead of RoCE (`VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1`, `DEEPEP_MAX_NVL_PEERS=8`): gap 33% (start) → 17% (after kernel PRs) → 4.5% (paired, same fabric), finally BI=1 ahead by 3.2%; transport does not touch BI (output bitwise identical across transports) |
| Stable top-k / routing scores | BI grouped top-k, tie-break by request-local index | Deterministic top-k (radix above 4K) 5.9 µs vs a histogram radix-select 56.7 µs | Full output contract below; "select candidates unordered, then sort" is correct only if the candidate set itself meets the contract for cutoff ties; a final stable sort cannot repair an earlier step that randomly dropped equal-score candidates |
| Mamba2 (SSD scan + conv1d) | exact replay ported from vllm#55905 (three extra chunk buffers x/dt/B); a partial chunk is replayed from the FP32 boundary state through the same SSD path | row-gated emit (emit only outputs needed now, update state at chunk boundary) restores Graph, but measured 0.1874 ms/token vs BI0 0.1726, still 8.57% slower; not default | Under BI Mamba2 declares 5 states (conv + temporal×4) but align-mode PC copied only 2 → after a hit, every prompt longer than one block is wrong; now copies the full per-layer declared state list, with a unit test asserting equal counts. The conv window is written tail-aligned by `conv_state.size(-1)` while the kernel reads `[0, width-1)`; when spec decode widens the cache they misalign; write only `[0, conv_kernel_size-1)`. `test_mamba2_ssd_emit` is occasionally flaky because the reference's autotune picks different blocking; pin the reference to the BI config. Only chunk64/8 heads exposes that the cumsum tile changes FP32 prefix sums (#55895). Gating inside the FP32 scan vs rounding first then gated RMSNorm round differently (#56244). Merging #58851 × #57329 directly: 28-shot GSM8K 432 → 4/500, because later groups reused the first group's checkpoint target block; 438 after adding the per-group gather. Under BI Mamba2 rejects the KV connector; real PD disaggregation is not covered |
| RMSNorm / gated RMSNorm | `cuda_rms_norm` (`CudaRMSNorm`), `grouped_gated_rms_norm` custom op, norm/act+quant fusion off, standalone `+quant_fp8` | Training-side TE fused RMSNorm differs from vLLM native RMSNorm by 1 ulp → only 74.4% of in_proj GEMM inputs bitwise identical and 1.7% of FP8 codes flip; so the training side aligns with the vLLM kernel instead of te.RMSNorm | Direct and compiled paths of the same gated RMS kernel differ; the difference disappears behind an opaque custom-op boundary. Review comment: this is a second BI RMSNorm implementation; DS4 reuses `rms_norm_batch_invariant` |
| FP8 activation quantization | standalone `+quant_fp8` op, `use_cuda_fp8_quant` under BI | Audit scale grouping, amax, rounding, clamp and aggregation order separately; the communication mode does not guarantee quant BI | The DS4 vLLM fork's `is_batch_invariant_quant_kernel_enabled()` only checks that the `.so` exists, not the BI switch, so BI=0 runs the unoptimized BI kernel for nothing (nsys 9.9%, 35.7 µs); with the fixed `.so` + `VLLM_BI_QUANT_TOKEN_BLOCKS=128` it is 5.3 µs, DS4 16703 → 18313 tok/s. Before a BI=0 comparison, grep nsys for `vllm::batch_invariant::`, or set `VLLM_BATCH_INVARIANT_KERNEL_LIB=/nonexistent` to fall back to upstream |
| cudagraph | pinned to FULL under BI (on DS4 FULL_AND_PIECEWISE is fastest: 16409 vs FULL 16068 vs PIECEWISE 10423 tok/s, BI=1 bitwise PASS) | FULL has worse TTFT (1792 vs 764 ms): on DS4 FULL lands as FULL_DECODE_ONLY and mixed prefill is always eager | Piecewise breaks BI through the switch "does this step run the graph or fall back to eager"; the criterion is total token count ≤ `max_cudagraph_capture_size`. Graph and eager paths are each BI but not bitwise identical to each other. Unrelated to padding. It reappears under FULL when the capture limit is set below `max_num_seqs × decode_query_len`. A short-prompt probe passing in all three modes does not prove PIECEWISE safe |
| determinism | `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` (rewritten to `:16:8` by vLLM BI init); the mamba_ssm SSD backward uses atomics for dA/dD/ddt_bias, call `set_deterministic_mode(True)` explicitly | affects only backward step 0 | see the section below |
| KV block zeroing (DS4 multi-segment KV) | unrelated to BI, but both arms need it | The DS4 vLLM fork's `KVBlockZeroer` picks a 16 B program block by the gcd of per-segment page sizes, 1–12 ms per step (2.27 ms/step = 7.6% of wall clock); upstream #52058 uses a 3-D grid + masked tail; chunk mode 60 µs | Visible only as `_zero_kv_blocks*` in nsys; for a memset-like kernel taking ms, first check `blk_size` ≤16 |
| Prefix caching (Mamba align mode, BI=0) | precompute multi-group alignment indices once (#58851 fits best), skip align postprocess without spec decode, skip preprocess/precopy when the host determines nothing crosses a block; on prefill export block-boundary checkpoints inside the scan (#57329) | PC on as is: BI0 throughput −4 to −20%, TPOT +13 to 21%, c=1 TTFT roughly doubles (eager launches per step 9→37, 8K prefill split in two); after the fix BI1 is about on par with PC off | The BI path must export checkpoints separately or BI1 is silently wrong; a chunk truncated by the token budget must also export its last checkpoint (under DP+EP one extra prefill step on any rank stalls global decode) |

## Reference: DS4-Flash per-operator choices

The full per-operator table behind the blog's summary table (consistent arm = BI=1). "Same as left" means the training forward calls the same vLLM kernel as rollout.

### Dense, norm and mHC

| Operator | rollout (consistent) | Training | What is fixed / given up; cost |
| --- | --- | --- | --- |
| Embedding | Table lookup | Forward: Same as left<br>Backward: PyTorch | Naturally consistent |
| mHC residual mixing (pre / post / Sinkhorn) | Split TileLang mHC kernels, `split_tokens=1` | Forward: Same as left<br>Backward: Hand-written VJP, Sinkhorn iterations unrolled, FP32 | Gives up the baseline's fused post+pre kernel (split count varies with token count)<br>Cost: Both sides lose one fusion |
| RMSNorm | Batch-invariant RMSNorm (one program per row) | Forward: Same as left<br>Backward: FP32 VJP | Per-row reduction order fixed<br>Cost: Small |
| FP8 linear layers (q_a/kv, q_b, wo_b, shared expert) | DeepGEMM FP8 GEMM, batch-invariant mode; weights in 128×128 blocks, activations quantized dynamically per token in groups of 128 | Forward: Quantizes BF16 master weights on the fly to the same block FP8, same DeepGEMM<br>Backward: Transformer Engine (TE) BF16 GEMM w.r.t. BF16 master weights (STE) | Gives up the baseline's FlashInfer switch at M<32; scale is UE8M0 on SM100 (8-bit, exponent only, i.e. a power of 2), FP32 on SM90<br>Cost: rollout: no faster small-batch kernel; training: FP8 weights repacked every forward (including recomputation) |
| Small linear layers (compressor, indexer weight projection) | FP32-output cuBLAS GEMM / batch-invariant persistent Triton matmul | Forward: Same as left<br>Backward: BF16 | Summation order along K fixed<br>Cost: Small |

### Attention (DSA sparse attention + FP8 KV cache)

| Operator | rollout (consistent) | Training | What is fixed / given up; cost |
| --- | --- | --- | --- |
| q-norm + RoPE + KV write | Fused kernel quantizes K and writes the FP8 cache | Forward: Same kernel writes a private training-side FP8 cache, then reads it back<br>Backward: FP32 RoPE / norm in PyTorch (straight-through) | K takes the same FP8 round trip<br>Cost: Training: one extra cache and dequantization |
| FP8 KV cache | Stored as FP8; prefill and decode dequantize to BF16 before attention | Forward: Same as left<br>Backward: — | Gives up the default FP8-reading decode kernel (blog 4.2)<br>Cost: rollout decode |
| CSA compressor | vLLM compression kernel, state reset per request | Forward: Same kernel's values, functional graph's gradients<br>Backward: autograd | In-request state starts as in prefill<br>Cost: Training: loop over requests |
| DSA indexer scoring | DeepGEMM FP8 MQA logits; prefill reads contiguous KV, decode reads paged KV | Forward: Same as left, per request<br>Backward: None (indexer not trained) | Paged and contiguous reads bitwise identical |
| Top-K selection + index arrangement | Batch-invariant Top-K (64-bit key + radix select), sorted descending, invalid entries last | Forward: Same as left<br>Backward: None | Ties go to the larger in-request index (blog 4.3)<br>Cost: One extra sort per side, recovered by fusion |
| Sparse MLA | `flash_mla_sparse_fwd`; prefill one segment per request, decode same sparse kernel | Forward: Same as left, once per request<br>Backward: cuDNN frontend DSA backward | In-request geometry ignores neighbors; decode emits 1 token per request per call<br>Cost: rollout: multiple prefill launches; training: serial over requests |
| o_proj | SM100: DeepGEMM FP8 einsum; SM90: grouped Triton W8A8 | Forward: Same as left<br>Backward: TE BF16 grouped GEMM + inverse-RoPE VJP | Aligned per architecture; no bitwise identity across architectures<br>Cost: Loop over groups on SM90 |

### MoE

| Operator | rollout (consistent) | Training | What is fixed / given up; cost |
| --- | --- | --- | --- |
| Router scoring + expert selection | Same FP32-output gate GEMM for all batch sizes; sqrt-softplus + bias selects top-k | Forward: Same as left, routing recomputed bitwise<br>Backward: Expert IDs fixed, FP32 gradient w.r.t. routing weights | Gives up the small-batch gate kernel; **R3 is not needed** (blog 4.4)<br>Cost: Small |
| Dispatch | DeepEP low-latency (LL below), fixed per-rank capacity, all ranks segment in sync; FP8 quantization uses elementwise round-to-nearest | Forward: DeepEP normal, plus one dispatch of routing metadata (fingerprint, expert, weight)<br>Backward: DeepEP backward | Experts get bitwise-identical FP8 values and scales either way<br>Cost: rollout: LL communication cannot overlap compute; training: metadata dispatch and hash matching |
| SiLU · mul · quantization | Batch-invariant fused kernel: product rounded once to BF16, then quantized in groups of 128 | Forward: Same as left (mandatory)<br>Backward: FP32 VJP | Rounding position fixed (blog 4.4)<br>Cost: Small after optimization |
| Routed expert GEMM | DeepGEMM masked grouped GEMM, static capacity as scheduling hint | Forward: DeepGEMM contiguous grouped GEMM, per-expert rows aligned to 128<br>Backward: TE BF16 grouped GEMM | Gives up the baseline's FlashInfer TRT-LLM MoE (ships as precompiled cubins; cannot be modified or verified as batch-invariant)<br>Cost: rollout: a main throughput gap; training: alignment padding |
| Combine | DeepEP LL combine, fixed merge order | Forward: FP32 accumulation in top-k slot order on the token's owning rank, then BF16<br>Backward: Gradient scatter | Merge order by slot, not arrival order or rank<br>Cost: 43 layers × 2 serial all-to-all per step |
| Shared expert | FP8 linear layer; auxiliary stream under LL + MNNVL (multi-node NVLink) | Forward: Same as left<br>Backward: BF16 (STE) | Stream changes overlap, not values<br>Cost: A source of recovered speed |

### Output layer and infrastructure

| Item | rollout (consistent) | Training | What is fixed / given up; cost |
| --- | --- | --- | --- |
| lm_head + logprob | Batch-invariant matmul; FP32 batch-invariant log-softmax | Logits per 8192 tokens, vLLM logprob value; backward via PyTorch log-softmax on the same logits (straight-through) | temperature must be a scalar<br>Cost: Training: each chunk's FP32 log-softmax kept for backward |
| Cross-GPU reduction | TP=1, no all-reduce; NCCL channel count and algorithm pinned as a guard | TP=1; CP exchanges boundaries only, no reduction | TP>1 not supported (blog 4.5)<br>Cost: Cannot split with TP |
| CUDA Graph | Kept, with the right mode (blog 4.6) | — | Graph and eager paths bitwise identical |
| Weight sync | Reload copies new values to original addresses, refreshes derived weights | Exports block-FP8 weights | Graph-referenced addresses unchanged<br>Cost: One export and transfer per step |

Details that back the table:

- **Scales and precision.** DeepGEMM scales are powers of 2: UE8M0-packed on SM100 (a Triton kernel emits packed weights), FP32 on SM90 (vLLM weight post-processing). Router gate and compressor projection output FP32; indexer weight projection and lm_head are BF16. Before the first update the checkpoint scales are reused; afterwards they are recomputed from amax. FP8 weights are not cached by default, so every forward including recompute repacks them.
- **FP8 KV layout.** Per token per layer: 448 FP8 NoPE dims, 64 BF16 RoPE dims, one UE8M0 scale per 64 elements, 1 alignment byte = 584 bytes (BF16: 1024). The sparse decode kernel must be selected explicitly; the BI switch alone does not change the default FP8-reading FlashMLA decode. The CSA compressor zeroes state at each request start.
- **Top-K key.** 64-bit key: high 32 bits = order-preserving transform of the score, low 32 bits = request-local index; radix select at 8 bits per round. Deterministic Top-K and the Top-K + index-sort fusion gave +2.3% and +0.1% alone but +6.7% together; the interaction is unexplained, so measure them together. Under DCP>1 the cross-GPU candidate merge uses the opposite tie rule; unsupported. The indexer auxiliary loss does not support CP>1.
- **Expert GEMM.** Contiguous layout aligns each expert's rows to 128. LL quantization switched to elementwise round-to-nearest; training quantizes the contiguous layout under the LL dispatch convention. The SiLU·mul·quant kernel has fixed launch geometry, is mandatory on both sides (training refuses to start without it), and uses different SiLU formulations on SM100 and SM90 (no cross-architecture bitwise promise). Backward: TE BF16 grouped GEMM, second GEMM dgrad/wgrad first, SwiGLU recomputed from saved gate_up in FP32.
- **Layouts tried for performance only.** TP2/DP4 17% slower, TP4/DP2 28% slower; DP8 is best for BI=1.
- **Cross-GPU.** Training dispatches routing metadata once more (per route: 16-element token fingerprint, expert index, FP32 routing weight), matches outputs by bitwise hash, and scatters with a deterministic Triton kernel. NCCL channel count, thread count, protocol and algorithm are pinned and symmetric-memory / custom all-reduce are off; under TP=1 this is a guard, not the mechanism. CP pads each sequence, splits contiguously, exchanges boundary hidden states and all-gathers compressed KV rows.
- **CUDA Graph.** Workspaces are allocated in warmup before capture and locked afterwards; the Top-K row-start buffer is a resident zero buffer. Weight-derived mHC parameters are recomputed in place on reload.

## The full output contract of stable Top-K

What orders equal scores (e.g. the request-local index)? Take it from the training side or current baseline; do not invent a tie-break. Preserve:

- Comparison key for equal scores, positive and negative zero, NaN/Inf (if allowed); a global batch offset cannot replace the request-local index.
- Output indices and their order, not just the same set; swapping equal-score indices changes later floating-point reduction order.
- valid length, mask, invalid sentinel, K larger than the valid count, empty rows, padding, tail blocks.
- prefill/decode index coordinates, cache page mapping, the position of the same semantic token.

Check the candidate for repeated allocation, sort input range, radix/merge passes, workspace, launch/grid/block; prove order preservation before deleting a pass. For any CUB/custom sort or selection, verify the binding and the real lifetime of its temporary storage; a library's stability claim does not replace integration tests. A pre-generated index/workspace saves overhead but must not overwrite in-use content under multiple streams, concurrent requests or Graph replay.

## Index / elementwise fusion

Limit scope to confirmed hotspots such as gather lengths, row start/end, sparse index concatenation and invalid-row zeroing. For every row write down input coordinates, whether the interval is half-open (closed left, open right), output capacity, valid range, and who writes the padding.

Fusion can save a launch or temporary tensor, but can also raise register pressure or hurt memory access. Measure the gain on all real shapes; the number of changed lines says nothing. Shrink the grid and loop stride together so the full output is covered; fill a reused buffer with poison values to catch unwritten tail/empty rows.

Integer indices are not automatically correct: off-by-one, request offset, CP boundary, overflow and uninitialized rows can bias a whole sequence. If fusion reaches into dequant/norm, keep the original dtype conversions and rounding positions. Unit-test Top-K and index post-processing separately, then combined, then module forward, then engine gates. Record whether the optimization also speeds up BI0: dynamic block/grid selection on a shared path is a general optimization and cannot all be counted as BI-specific overhead reduction.

## Split-K and the M of grouped GEMM

A GEMM's M is usually the number of rows processed; masked grouped GEMM computes several experts in groups and uses only each group's valid token rows. `expert_num_tokens.max().item()` pulls the device max token count to the host, commonly as a capacity/scheduling hint; it does not "fix the shape" or "guarantee BI". The readback synchronizes and usually cannot be used in CUDA Graph capture. Before removing it, read the API: is the parameter a correctness bound, a capacity, or a tuning hint? Restore an estimate/safe upper bound the library supports, and cover heavy load and empty experts. Do not claim without source evidence that underestimating M makes the default kernel access memory illegally, and do not set an arbitrarily small value for speed.

One launch per token avoids some batch-dependent reduction/scheduling but means many launches and low utilization. Batched launch with fixed `n_split` may keep the per-row numerical order, but the K range of each split, the accumulation order within it, the accumulator dtype, partial reduce order, tail/padding and actual kernel specialization must still be fixed or verified. "Fixed K/n_split" is a candidate mechanism, not a proof of prefill=decode: different tiles, epilogues, precision or preprocessing can still differ at the same split count. Test identical operator inputs and the full semantic path.

## BI coverage of attention

Attention BI covers QK, online softmax, PV and the merge order of blocks/splits. Beyond the split count, the tile scheduler, Q-stage, cluster/scheduling choice, page layout, mask/sink, score scale and accumulator/rounding can depend on batch/length. Passing small batches does not cover long context, prefill chunk and decode, Graph buckets, mixed batches, page boundaries, prefix cache hit/miss. Graph+prefix enabled does not mean one test executed all these branches; record the coverage.

## Observation in Graph mode

Save the resolved Graph mode, capture sizes, prefill/decode/mixed step types and evidence of kernel calls. A successful capture guarantees neither eager-equal numerics nor that every step uses the same kind of graph. A Python hook usually runs at capture, not replay; a capture dump cannot stand for all replays.

To observe replay, use controlled device-side outputs or static diagnostic buffers, and state whether instrumentation changes the capture. Eager hooks can localize, but the final recheck must return to the original Graph mode. Even if the compiled path with Graph off matches production Graph on fixed samples, eager may still differ; keep artifacts for all three. A hook not firing on the compiled path does not mean the layer did not execute.

Turning the graph off changes speed and therefore online batching. First fix each step's tokens, requests/neighbors, padding and cache state, then compare the supported modes (eager/FULL/PIECEWISE). Use a 2×2 experiment to rule out the dispatch/split plan the switch changes at the same time. A short filler may put a mixed step into a capture bucket while a long filler goes eager. Prefill/decode overlap differs between the full and a small model; a small model passing does not cover untriggered production mixed steps. For intermittent failures keep multi-round samples and the trigger rate; do not pick only successful rounds.

<a id="determinism"></a>

## determinism takes effect only on the main thread

In PyTorch 2.13 (nv26.07) Inductor config overrides are thread-local (ContextVar). `torch.use_deterministic_algorithms(True)` sets `torch._inductor.config.deterministic` on the main thread; on autograd's CUDA device thread it stays False. A `torch.compile`d norm VJP first compiles on the autograd thread during backward, sees `deterministic=False`, and the generated looped-reduction kernel enables `dynamic_scale_rblock`: besides `R0_BLOCK=1024` it adds candidate 512, and the first launch picks by timing. The two reduce in different orders and differ at bit level. Only at the end of the dynamo compile is that thread set to deterministic; later compiles have one fixed config. Result: only backward step 0 is not bitwise reproducible across runs; forward logprob and the RL closed loop are unaffected.

Apply both fixes: `torch.compile(..., options={"deterministic": True})` at the compile site (environment-independent), and `TORCHINDUCTOR_DETERMINISTIC=1` in the recipe environment (covers any first compile on the autograd thread, including the vLLM forward function compiled inside recompute). Also: verl's full_determinism actually sets `warn_only=True`, and the mamba_ssm SSD backward must enable deterministic mode explicitly.

## How to break down the cost from BI0 to BI1

Draw the actual call chain first: what the default fast kernel, the BI branch and the extra preprocessing/sorting/postprocessing each are. Do not infer "split-K was changed" or "a temporary Triton was used" from function names; compare source, binding, compiled artifacts and real launches.

You may copy the BI0 implementation into a standalone candidate and add the deterministic/BI conditions one at a time for a bisectable A/B. It need not live in the same file, but the default path must stay unchanged. If the BI0 algorithm cannot meet the required tie/reduction contract, switching algorithms is allowed; state the cost.

Fairness between the arms comes before any kernel conclusion: same fabric (on DS4 two environment variables closed most of the remaining gap), same node, same image, BI=0 not running the BI kernel, no JIT stall, and the KV zeroing fix in both arms. Nemotron's BI1 vs BI0 E2E throughput loss is the median over 5 pairs, with single pairs reported separately.

## Minimal record for each candidate

```text
baseline / candidate source+binary hashes
one intended change; BI-specific or common
same-input bitwise + BI composition + prefill/decode gates
BI0 unchanged + real shape/tie/mask/tail + Graph/workspace reuse
paired latency, launch/allocation counts, repetitions
pass / fail / inconclusive; retained or rejected; remaining scope
```

On failure, save the minimal counterexample, then withdraw from the default path. When the user requires all micro gates to pass before E2E, do not skip them; performance numbers do not cover a correctness failure. Do not quietly remove an existing stable-sort patch to reproduce a long-length mismatch; that tests a regression, not a defect of the current best code.
