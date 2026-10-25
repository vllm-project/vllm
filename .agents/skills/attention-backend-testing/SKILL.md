---
name: attention-backend-testing
description: "Standard for testing a vLLM attention backend (vllm/v1/attention/backends/*): what a complete suite must cover (capability contract, metadata builder, every forward dispatch path, KV-cache write/read and dtypes, CUDA graph modes, workspace/memory, scheduler, model runner v1/v2, async scheduling, spec decode, PCP/DCP, logits), where the tests live, how accuracy and performance artifacts are produced and compared in Buildkite. Use when writing or restructuring tests for an attention backend, when reviewing a PR that touches an attention backend, its kernels, its selector rules or its CI steps, or when judging whether a backend's coverage is sufficient."
---

# Attention backend testing standard

An attention backend is a contract between four parties: the selector (which
reads its capability declarations), the model runner (which builds metadata,
pads batches and captures graphs), the KV-cache manager (which sizes and lays
out pages), and the kernels it dispatches to. Bugs live at those seams far more
often than inside a kernel. A suite meets this standard when every seam has a
test that fails if the seam breaks.

Read [the coverage matrix](references/coverage-matrix.md) for the full list of
required cases, [model shapes](references/model-shapes.md) for the shapes to
parametrize with, and [CI and artifacts](references/ci-artifacts.md) for
Buildkite wiring and perf/accuracy comparison. When reviewing a PR, use
[the review checklist](references/review-checklist.md).

## Rules

1. **Enumerate dispatch paths first.** Before writing tests, list every branch
   in `Impl.forward`, `Builder.build`, `build_for_cudagraph_capture`,
   `build_for_drafting`, `do_kv_cache_update` and any fused-update hooks, as a
   table of *condition -> kernel*. Each row gets at least one test that
   (a) checks the numerical output against an independent reference and
   (b) asserts the intended kernel ran, by spying on the op (monkeypatch a
   wrapper that records the call). Output-only tests cannot tell a fallback
   from the fast path. Use synthetic shapes where no model reaches a branch
   (a head-32 case for a head-below-64 fallback). Branch coverage of the
   backend module is necessary but not enough: one branch passes features
   (window, sinks, soft cap, ALiBi, non-causal, fp8) to its kernel as
   arguments, so also run every declared feature through every phase
   (prefill, extend, decode, spec decode, mixed) and check the output.
2. **Independent reference.** Compare against a float32 PyTorch reference
   computed from the logical (unpaged, unquantized-then-dequantized) K/V, not
   against another kernel of the same library. For fp8 KV, quantize the
   reference inputs with the same scales so the tolerance only absorbs
   accumulation error. Use non-unit K/V scales; unit scales hide missing
   scale plumbing.
3. **Declarations are tested both ways.** Every value a backend declares
   supported (dtype, KV dtype, block size, head size, attn type, sinks,
   sliding window, non-causal, PCP/DCP, KV layouts, CUDA graph support level)
   must run; every value it does not declare must be rejected by the selector
   with a clear reason, not crash later in a kernel.
4. **Phases and their mixtures.** Test prefill-only, decode-only, extend
   (context + new tokens, i.e. prefix cache hit or chunked prefill), and one
   batch containing all three in the order the reorder pass produces. Include
   multi-token decode (speculative decoding) with uniform and non-uniform
   query lengths.
5. **Shape matrix comes from real models.** Parametrize head size, H/KV ratio,
   window size and TP-sharded head counts from current flagship configs (see
   model shapes). Tokens per request: 1, 2048, 4096, 8192; batch sizes 1, 16,
   128; every supported block size. Keep the full Cartesian product out of
   pre-merge CI: pick a covering set (each value appears at least once with
   each phase) and run the product nightly.
6. **Boundaries are explicit cases.** Sequence length exactly at, one below
   and one above: a block boundary, the sliding window, a kernel partition
   size, and any context-chunk size the builder uses. Empty context, context
   longer than one chunk, padded tokens beyond `num_actual_tokens`.
7. **CUDA graphs are tested through the runner's modes.** Cover `NONE`,
   `PIECEWISE`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE` (and `FULL` if the
   builder claims `ALWAYS`). For each graph-captured shape, replay with
   different sequence lengths and block tables than at capture and compare to
   eager. Assert the builder's declared `AttentionCGSupport` matches what the
   runner actually captures.
8. **No hidden syncs or allocations in captured paths.** Decode-only builds must
   not sync the GPU (run under `gpu_sync_debug` / `gpu_sync_allowed`
   strictness). Persistent buffers must be allocated once, sized from
   `max_num_seqs` / `max_num_batched_tokens`, and reused. Per-forward
   allocations in the hot path are findings.
9. **Workspace is sized for the configuration, not the worst case.** For each
   config, record which persistent buffers the builder allocated and assert
   each one is read by at least one path that config can take (for example, no
   sliding-window tables without a sliding layer, no KV-sharing workspace
   without a sharing layer). Track peak memory of a fixed workload and fail on
   growth beyond a stated budget.
10. **Runtime integration through a dummy-weight model.** Build a small model
    from a real architecture (`load_format="dummy"`, `hf_overrides`) and run it
    through `LLM` with the scheduler, prefix caching, chunked prefill, async
    scheduling on/off, model runner v1 and v2, and each CUDA graph mode. For a
    hybrid model, keep exactly its smallest repeating layer block (Qwen3.8: 3
    GDN + 1 full; Gemma 4: 5 sliding + 1 global; GPT-OSS: sliding + full), so
    every layer kind and its neighbors run. Stock dummy weights give flat
    logits, which hide broken attention: rescale matrices to 1/sqrt(fan_in),
    give norms unit gain, make every MoE expert active (routing ties flip on
    any numeric change), and compare teacher-forced prompt logprobs against a
    reference backend with a tolerance calibrated on a second correct backend.
    Keep a negative control (a deliberately wrong window must fail).
11. **Context parallelism is either tested or rejected.** If the backend does
    not set `supports_pcp` / `supports_dcp`, add a selector test that a PCP/DCP
    config refuses it. If it does, test the LSE return path and the merge.
12. **Determinism and flakiness.** Seed everything. No timing assertions in
    pytest. A test that needs retries is a bug to fix, not to mark flaky.
13. **Speed lives in benchmarks, not tests.** Kernel and backend timing go in
    `benchmarks/attention_benchmarks/` with a per-backend config; CI stores the
    JSON per commit and compares PRs against their merge base (see CI and
    artifacts).
14. **Shard single-GPU tests.** Use `--shard-id/--num-shards` (Buildkite
    `parallelism`) so a 1-GPU suite spreads over the node's GPUs; keep
    multi-GPU cases (DCP/PCP) in their own step.
15. **Every test runs in CI.** A test file that no Buildkite step collects on
    the hardware it needs is not coverage. Add the backend file and its op
    modules to the step's `source_file_dependencies`.

16. **Land the suite first, fixes after.** The suite PR marks every test a
    known defect fails as `xfail(strict=True)` through one registry,
    `<backend>/known_defects.py` (one entry per defect: a reason and the test
    ids it breaks). Each defect is then fixed in its own PR stacked on the
    suite, whose diff is the fix plus the deleted registry entry; strict mode
    makes the fix prove itself, because a test that starts passing while
    still listed fails. Find each fix's tests by running the suite on the
    unfixed backend, then the failing set once per fix. A defect outside the
    backend (another library, a shared runner contract) stays registered
    with a link to its issue.
17. **Build the harness like the runner does.** Create layers under the target
    device (model loaders do), take the KV spec dtype from the layer's
    `kv_cache_torch_dtype`, pass the spec through `customize_spec`, and view
    the cache with the same layout helpers the runner uses. Spy on kernels by
    patching the module attribute the backend resolves at call time
    (`importlib.import_module`, not `import a.b.c as m`, which can return a
    re-exported function of the same name).

## Nuances that make a passing test meaningless

Each of these let a broken backend pass in this suite's history; check for
them before trusting a green run.

- **Flat dummy logits.** `load_format="dummy"` draws weights uniform in
  +-1e-3, so logits are almost flat and every token's logprob is near
  `-log(vocab)` whatever attention returns. Rescale weights (rule 10) and
  assert the logprob spread is wide before comparing.
- **MoE routing ties.** With dummy router weights most experts tie; any numeric
  change flips the routing and swamps the attention difference. Make every
  expert active.
- **Reference under different settings.** Compare against the reference
  backend with the same dtype, KV dtype, block size, graph mode and runner, so
  the only variable is the attention backend.
- **Uncalibrated tolerance.** Set the tolerance from the gap between two
  correct backends, not by loosening until green.
- **No negative control.** Run one deliberately wrong configuration (wrong
  window) and assert it fails the same comparison; otherwise the tolerance may
  accept anything.
- **Env read in another process.** Engine tests run model code in spawned
  workers that read `VLLM_*` at import; set env vars before the `LLM` is built
  and do not refresh cached env in the test process.
- **Output-only checks.** A correct output from a fallback kernel passes; spy
  on the kernel (rule 1).

## Tiers and where they live

| Tier | What | Needs GPU | Location |
| --- | --- | --- | --- |
| L0 contract | capability declarations, selector accept/reject, spec customization, layouts | no (mock platform) or 1 | `tests/v1/attention/backends/<backend>/test_capabilities.py` |
| L1 op | each kernel the backend calls, against a float32 reference | 1 | `tests/kernels/attention/test_<op>.py` |
| L2 builder | metadata split, chunking, SWA tables, drafting, capture build, sync and allocation checks | 1 | `.../<backend>/test_metadata_builder.py` |
| L2 dispatch | one test per forward branch, spy on kernel + reference compare | 1 | `.../<backend>/test_dispatch.py` |
| L2 KV cache | write path per layout and dtype, read-back round trip, sharing/bypass | 1 | `.../<backend>/test_kv_cache.py` |
| L2 correctness | model-shape matrix x phases x KV dtype x block size | 1 | `.../<backend>/test_correctness.py` |
| L2 graphs | capture/replay per mode, padded batches | 1 | `.../<backend>/test_cudagraph.py` |
| L3 runtime | dummy-weight model via `LLM`: scheduler, MRv1/v2, async, spec decode, CUDA graph modes, logprobs vs reference | 1-2 | `tests/v1/e2e/attention/test_<backend>_runtime.py` |
| L4 task | real-weight GSM8K for one model per head size | 1-8 | `tests/evals/gsm8k/configs/` |
| Perf | kernel and backend timing | 1 | `benchmarks/attention_benchmarks/configs/<backend>.yaml` |

Shared helpers live in `tests/v1/attention/backends/harness.py` (paged batch
builder with unique blocks, float32 reference with window/sinks/softcap/ALiBi,
kernel spy) and `tests/v1/attention/model_shapes.py` (shape table). A backend
adds only its own driver, as `rocm_aiter_fa/fa_harness.py` does.

## Writing a new backend suite

1. Build the dispatch table (rule 1) and paste it as the module docstring of
   `test_dispatch.py`; the table is what reviewers check coverage against.
2. Copy the coverage matrix and mark each row *covered by test X*, *N/A
   because Y*, or *gap*. A gap needs an issue link.
3. Write L0 and L2 first; they are cheap and find most bugs. L3 last.
4. Run the branch-coverage check from [CI and artifacts](references/ci-artifacts.md)
   on the backend module and explain every uncovered branch.
5. Wire CI and the benchmark config in the same PR.
