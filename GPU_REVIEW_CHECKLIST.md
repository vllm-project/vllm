# GPU Testing & Review Submission Checklist

This checklist ensures the ILP optimization is thoroughly validated before submitting for code review.

---

## Phase 1: GPU Environment Setup

- [ ] GPU hardware available (NVIDIA GPU with compute capability >= 5.0)
- [ ] NVIDIA drivers installed: `nvidia-smi` works
- [ ] CUDA toolkit installed: `nvcc --version` works
- [ ] PyTorch supports CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

**Recommended Setup:**
```bash
cd /Users/mohan/projects/vllm
source vllm_cpu_env/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cuda
```

**Verification:**
```bash
python3 -c "import torch.ops; print(hasattr(torch.ops._C, 'gelu_tanh_and_mul_ilp'))"
```

Expected output: `True`

---

## Phase 2: Quick Validation (5 minutes)

Run basic functionality test:

```bash
python3 scripts/quick_gpu_test.py
```

**Expected output:**
```
✓ GPU: [GPU model]
✓ CUDA Version: 12.x
✓ CUDA OPERATIONS CHECK
✓ gelu_tanh_and_mul - Original GELU+mul kernel
✓ gelu_tanh_and_mul_ilp - ILP GELU+mul kernel
Average Speedup: 1.2x - 1.8x
```

- [ ] All ops available
- [ ] Speedup between 1.0x and 3.0x (sanity check)
- [ ] No CUDA errors

---

## Phase 3: Comprehensive Benchmarking (30 minutes)

Run full benchmark suite:

```bash
python3 scripts/quick_gpu_test.py --full
```

This tests:
- Multiple batch sizes: 32, 64, 128, 256, 512, 1024
- Multiple hidden dimensions: 512, 1024, 2048, 4096
- Average 100 iterations per configuration

**Results to collect:**
- [ ] Min speedup: _____ x
- [ ] Max speedup: _____ x
- [ ] Average speedup: _____ x
- [ ] Any configurations slower than original? No / Yes ____

**Interpretation:**
| Avg Speedup | Status | Action |
|-------------|--------|--------|
| > 1.3x | Excellent | ✓ Ready for review |
| 1.1x - 1.3x | Good | ✓ Acceptable, but may want to profile |
| 0.9x - 1.1x | Marginal | ⚠ Investigate specific configurations |
| < 0.9x | Regression | ✗ Fix before review |

---

## Phase 4: Correctness Validation

Run comprehensive correctness tests:

```bash
python3 scripts/quick_gpu_test.py --correctness
```

Tests:
- Float32, Float16, BFloat16 dtypes
- Multiple shapes (32, 128, 512 tokens)
- Max difference between original and ILP kernels

**Validation criteria:**
- [ ] Max difference < 1e-5 for float32
- [ ] Max difference < 1e-2 for float16/bfloat16
- [ ] All dtypes supported
- [ ] No NaN or Inf values

---

## Phase 5: Advanced Analysis (Optional)

### GPU Utilization Analysis
```bash
# Monitor GPU during benchmark
nvidia-smi dmon -s puc
# In another terminal:
python3 scripts/quick_gpu_test.py --full
```

- [ ] GPU utilization > 80%
- [ ] GPU memory usage reasonable (< 80% of total)
- [ ] No thermal throttling warnings

### Performance Profiling (Optional, requires Nsight Systems)
```bash
nsys profile -o results.nsys-rep \
  python3 scripts/quick_gpu_test.py
# Open in Nsight Systems GUI to analyze kernel execution
```

### Energy Efficiency (Optional)
```bash
# Log power usage during test
nvidia-smi -l 100 -f gpu_power.log &
python3 scripts/quick_gpu_test.py --full
kill %1
# Calculate J/sample from logs
```

---

## Phase 6: Documentation

### Create Results Summary
Save GPU test results to `GPU_TEST_RESULTS_<DATE>.md`:

```markdown
# GPU Test Results - [Date]

## Environment
- GPU: [Model]
- CUDA: [Version]
- Driver: [Version]
- PyTorch: [Version]

## Results
- Min speedup: X.XXx
- Max speedup: X.XXx
- Avg speedup: X.XXx
- Correctness: ✓ All tests passed
- Dtypes: float32, float16, bfloat16

## Key Findings
- [Performance observations]
- [Any anomalies]
- [GPU-specific notes]
```

- [ ] Speedup results documented
- [ ] GPU environment details saved
- [ ] Any anomalies noted
- [ ] Recommendation for review included

---

## Phase 7: Code Review Readiness

### Commit Message Requirements
```
Task 3: Instruction-Level Parallelism (ILP) Optimization for Activation Kernels

[existing commit message]

GPU Test Results:
- Environment: GPU [Model], CUDA [Version]
- Speedup: [X.XXx] average across [N] configurations
- Correctness: ✓ All dtypes (float32, float16, bfloat16)
- Max difference from baseline: [value]
- Test command: python3 scripts/quick_gpu_test.py --full
```

### Files to Include in Review
- [ ] `csrc/activation_kernels.cu` - ILP kernel implementation
- [ ] `csrc/torch_bindings.cpp` - Op registration
- [ ] `benchmarks/benchmark_ilp_kernels.py` - Benchmark suite
- [ ] `ILP_OPTIMIZATION_GUIDE.md` - Technical documentation
- [ ] `GPU_TESTING_GUIDE.md` - How to reproduce tests
- [ ] `GPU_TEST_RESULTS_<DATE>.md` - Actual measurements

### PR Description Template
```markdown
## Description
Implement Instruction-Level Parallelism (ILP) optimization for activation 
kernels (GELU, SiLU) to hide transcendental function latency.

## Changes
- New ILP kernel: 4-element loop unrolling
- Expected speedup: 1.5-2.5x theoretical, [X.XXx measured]
- Fully backward compatible - coexists with existing kernels

## GPU Testing
✓ Tested on: [GPU model]
✓ Speedup: [X.XXx] average
✓ Correctness: Bitwise identical to original (within FP rounding)
✓ Supported dtypes: float32, float16, bfloat16

See GPU_TESTING_GUIDE.md for reproduction steps.

## Test Results
- Quick test: `python3 scripts/quick_gpu_test.py` (5 min)
- Full benchmark: `python3 scripts/quick_gpu_test.py --full` (30 min)
- Correctness: `python3 scripts/quick_gpu_test.py --correctness` (5 min)

Results: GPU_TEST_RESULTS_[date].md
```

---

## Phase 8: Common Issues & Troubleshooting

### ❌ "CUDA not available"
```bash
# Reinstall with CUDA support
uv pip uninstall torch vllm
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cuda
```

### ❌ "torch.ops._C.gelu_tanh_and_mul_ilp not found"
```bash
# Rebuild C++ extensions
python3 setup.py build_ext --inplace
```

### ⚠️ "Speedup is negative (ILP slower)"
**Possible causes:**
1. GPU doesn't benefit from ILP on this workload
2. Compiler or driver version differences
3. Register pressure causing spilling
4. Thermal throttling

**Next steps:**
- Check GPU utilization: `nvidia-smi`
- Profile kernel: `nvidia-smi -l 1` during test
- Try different tensor sizes
- Check thermal status: `nvidia-smi dmon`

### ⚠️ "Speedup varies widely (0.8x - 2.0x)"
**Normal due to:**
- Different tensor sizes having different characteristics
- GPU scheduling variations
- Memory bandwidth limitations at small sizes
- Computational overhead at large sizes

**Analyze:** Look at throughput (GB/s) rather than speedup for details.

---

## Final Checklist Before Submitting PR

- [ ] All GPU tests passed
- [ ] Average speedup >= 1.1x (or well-documented if not)
- [ ] Correctness validated across dtypes
- [ ] No regressions on any configuration
- [ ] Code compiles without warnings
- [ ] All existing tests still pass: `pytest tests/kernels/core/test_activation.py`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] GPU test results documented
- [ ] PR description includes speedup numbers
- [ ] Commit includes "Co-authored-by" trailers

---

## After PR Review

1. **If reviewer asks for more data:**
   - Additional GPU models: Re-run tests on different hardware
   - Different CUDA versions: Test with CUDA 11.8, 12.0, 12.2
   - Large-scale benchmarks: Test with production batch sizes

2. **If optimization doesn't improve enough:**
   - Try different unroll factors (currently 4)
   - Consider kernel fusion with other ops
   - Investigate memory bottlenecks with profiling

3. **If optimization causes issues:**
   - May need to disable on specific GPUs
   - Consider making it configurable via env variable
   - Can revert to original kernel with git bisect

---

## Success Criteria

✓ Speedup measured and reported: **[X.XXx]**
✓ All dtypes working: **float32, float16, bfloat16**
✓ Correctness validated: **Bitwise identical (within FP tolerance)**
✓ No regressions: **Yes**
✓ Compiles cleanly: **Yes**
✓ Documentation complete: **GPU_TESTING_GUIDE.md + GPU_TEST_RESULTS.md**
✓ Ready for review: **YES**

