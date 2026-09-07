# PR #54706 — JartX Review Closure on W7900

> Work in progress — this file is filled in as items complete.
> Authoritative review source:
> https://github.com/vllm-project/vllm/pull/54706#issuecomment-5555703806

## Environment
- GPU: (pending build verification)
- gfx: gfx1100 (AMD Radeon Pro W7900D, 48 GB, 48 CUs)
- ROCm: wheel runtime 7.14.60850 (torch 2.14.0+rocm7.14); build toolchain /opt/rocm 7.2.1 (hipcc/hipify only)
- OS: Ubuntu 24.04.4 LTS, kernel 6.8.0-79-generic
- PyTorch: 2.14.0+rocm7.14
- vLLM commit: dddf164781 (fix/rdna3-w4a16-determinism) + closure commit
- upstream base: 6fbb00b188 (upstream/main); merge-base 40b2f62061

## JartX Review Checklist

- [ ] 1. Accuracy-first framing (PR description leads with accuracy, not greedy repro)
  - Review #1/#4: body text vs table contradiction; missing N/K and dtype dimensions
- [ ] 2. WMMA `at::zeros` memset investigated as the M=16/-17.2% hypothesis
- [ ] 3. Scratch coverage proven total; `zeros` -> `empty` where proven
  - Review #2/#3: out-of-range lanes storing 0.0f vs guarded writes
- [ ] 4. Correctness test added against FP32/dequantized reference (not just repeatability)
  - Review #5: stale `at::empty` scratch can be bitwise-repeatable AND wrong
- [ ] 5. Scalar regression strengthened K=1024/4-writers -> K=4096/~16-writers
- [ ] 6. Test context manager properly exited (`_cm.__enter__()` leak fixed)
- [ ] 7. Hardcoded MASTER_PORT=29741 replaced with repo-standard dynamic port
- [ ] 8. Benchmarks report dtype + M/N/K + k_split (+ path, scratch)
- [ ] 9. bf16/fp16 benchmark rerun on current branch (M = 1, 8, 16, 64, 128, 512)
- [ ] 10. Stale CAS wording fixed (TORCH_CHECK message, header comments)
  - Review #8 + #6 first half: `c` zeros->empty one-way door comment
- [ ] 11. split-count==1 optimization: scalar z_count>1 guard added (WMMA already had it)
- [ ] 12. FP32 atomic alternative (`global_atomic_add_f32`) documented in PR description

## Correctness
(pending)

## Determinism
(pending)

## Performance
(pending)

## Key Result
(pending)

## Remaining Risks
(pending)

## Files Changed
(pending)

## Commit
(pending)

## PR update
(pending)
