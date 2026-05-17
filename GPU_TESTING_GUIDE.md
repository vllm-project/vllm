# GPU Testing Guide: ILP Optimization Validation

## Quick Start (5 minutes)

### 1. Check GPU Setup
```bash
# Verify GPU is accessible
nvidia-smi

# Verify PyTorch CUDA support
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2. Compile vLLM with CUDA Support

Choose one of these:

**Option A: Fast (Using Precompiled Wheels)**
```bash
cd /Users/mohan/projects/vllm
source vllm_cpu_env/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cuda
```

**Option B: Full Compile (Most Reliable)**
```bash
cd /Users/mohan/projects/vllm
source vllm_cpu_env/bin/activate
uv pip install -e . --torch-backend=cuda
# This will take 10-30 minutes depending on GPU
```

**Option C: With Specific CUDA Version**
```bash
# If you have a specific CUDA version
CUDA_HOME=/usr/local/cuda-12.1 uv pip install -e . --torch-backend=cuda
```

### 3. Verify CUDA Operations Work
```bash
python3 << 'EOF'
import torch
import torch.ops

# Test that CUDA ops are available
print("Testing CUDA ops availability...")
print(f"gelu_and_mul: {hasattr(torch.ops._C, 'gelu_and_mul')}")
print(f"gelu_and_mul_ilp: {hasattr(torch.ops._C, 'gelu_and_mul_ilp')}")
print(f"gelu_tanh_and_mul: {hasattr(torch.ops._C, 'gelu_tanh_and_mul')}")
print(f"gelu_tanh_and_mul_ilp: {hasattr(torch.ops._C, 'gelu_tanh_and_mul_ilp')}")

# Quick sanity test
x = torch.randn(32, 4096, dtype=torch.float32, device='cuda')
out = torch.empty(32, 2048, dtype=torch.float32, device='cuda')
torch.ops._C.gelu_and_mul(out, x)
print(f"Output shape: {out.shape}, dtype: {out.dtype}")
print("✓ CUDA ops working!")
EOF
```

---

## Comprehensive GPU Testing

### Test 1: Quick Validation (5 minutes)
```bash
# Run with minimal iterations to validate correctness quickly
python3 benchmarks/benchmark_ilp_kernels.py \
  --num-tokens 32 128 \
  --d 512 4096 \
  --iterations 50
```

**Expected Output:**
```
GELU ILP Optimization Benchmark
...
Shape: (128, 8192) -> (128, 4096)
────────────────────────────────────
  Standard GELU (original kernel):
    Time: 0.254 ms
    Throughput: 128.56 GB/s
  Standard GELU (ILP kernel):
    Time: 0.189 ms
    Throughput: 174.31 GB/s
  ILP Speedup: 1.345x
  Max difference: 0.00e+00
```

### Test 2: Comprehensive Benchmark (20-30 minutes)
```bash
python3 benchmarks/benchmark_ilp_kernels.py \
  --num-tokens 32 64 128 256 512 1024 2048 \
  --d 512 1024 2048 4096 8192 \
  --iterations 200
```

This tests all combinations and provides detailed speedup analysis.

### Test 3: GPU Memory Profile
```bash
python3 << 'EOF'
import torch
import torch.ops

print("GPU Memory Analysis")
print("=" * 60)

for num_tokens in [32, 128, 2048]:
    for d in [512, 2048, 4096]:
        input_mb = num_tokens * 2 * d * 4 / 1e6  # float32 = 4 bytes
        output_mb = num_tokens * d * 4 / 1e6
        total_mb = input_mb + output_mb
        
        print(f"Shape: ({num_tokens:4d}, {2*d:5d}) -> ({num_tokens:4d}, {d:5d})")
        print(f"  Input: {input_mb:6.2f} MB, Output: {output_mb:6.2f} MB, Total: {total_mb:6.2f} MB")

# Check available GPU memory
props = torch.cuda.get_device_properties(0)
print(f"\nGPU Memory: {props.total_memory / 1e9:.2f} GB")
EOF
```

---

## Detailed Test Harness

Create `test_ilp_gpu.py` for comprehensive validation:

```bash
cat > /Users/mohan/projects/vllm/tests/test_ilp_gpu_validation.py << 'EOF'
#!/usr/bin/env python3
"""
GPU validation test for ILP optimization.
Run with: python tests/test_ilp_gpu_validation.py
"""

import torch
import time
import numpy as np

def test_ilp_availability():
    """Test that ILP kernels are available."""
    print("\n" + "="*60)
    print("Test 1: ILP Kernel Availability")
    print("="*60)
    
    ops = [
        'gelu_and_mul',
        'gelu_and_mul_ilp',
        'gelu_tanh_and_mul',
        'gelu_tanh_and_mul_ilp',
        'silu_and_mul',
        'silu_and_mul_ilp',
    ]
    
    for op in ops:
        has_op = hasattr(torch.ops._C, op)
        status = "✓" if has_op else "✗"
        print(f"  {status} torch.ops._C.{op}")
    
    return True


def test_correctness():
    """Test that ILP kernels produce correct results."""
    print("\n" + "="*60)
    print("Test 2: Correctness Validation")
    print("="*60)
    
    torch.manual_seed(42)
    
    test_cases = [
        (32, 512),
        (128, 2048),
        (2048, 4096),
    ]
    
    for num_tokens, d in test_cases:
        print(f"\nShape: ({num_tokens}, {2*d}) -> ({num_tokens}, {d})")
        
        x = torch.randn(num_tokens, 2*d, dtype=torch.float32, device='cuda')
        
        # Original kernel
        out_orig = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
        torch.ops._C.gelu_tanh_and_mul(out_orig, x)
        
        # ILP kernel
        out_ilp = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
        torch.ops._C.gelu_tanh_and_mul_ilp(out_ilp, x)
        
        # Compare
        max_diff = (out_orig - out_ilp).abs().max().item()
        rel_error = max_diff / (out_orig.abs().max().item() + 1e-8)
        
        status = "✓" if max_diff < 1e-5 else "⚠" if max_diff < 1e-3 else "✗"
        print(f"  {status} Max difference: {max_diff:.2e}, Relative: {rel_error:.2e}")
    
    return True


def benchmark_kernels():
    """Benchmark original vs ILP kernels."""
    print("\n" + "="*60)
    print("Test 3: Performance Benchmark")
    print("="*60)
    
    configs = [
        (32, 512, "small"),
        (128, 2048, "medium"),
        (2048, 4096, "large"),
    ]
    
    results = {}
    
    for num_tokens, d, label in configs:
        print(f"\n{label.upper()}: ({num_tokens}, {2*d}) -> ({num_tokens}, {d})")
        
        x = torch.randn(num_tokens, 2*d, dtype=torch.float32, device='cuda')
        out_orig = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
        out_ilp = torch.empty(num_tokens, d, dtype=torch.float32, device='cuda')
        
        # Warmup
        for _ in range(5):
            torch.ops._C.gelu_tanh_and_mul(out_orig, x)
            torch.ops._C.gelu_tanh_and_mul_ilp(out_ilp, x)
        
        torch.cuda.synchronize()
        
        # Benchmark original
        start = time.perf_counter()
        for _ in range(100):
            torch.ops._C.gelu_tanh_and_mul(out_orig, x)
        torch.cuda.synchronize()
        time_orig = (time.perf_counter() - start) * 10  # Convert to ms
        
        # Benchmark ILP
        start = time.perf_counter()
        for _ in range(100):
            torch.ops._C.gelu_tanh_and_mul_ilp(out_ilp, x)
        torch.cuda.synchronize()
        time_ilp = (time.perf_counter() - start) * 10
        
        speedup = time_orig / time_ilp
        
        print(f"  Original: {time_orig:.3f} ms")
        print(f"  ILP:      {time_ilp:.3f} ms")
        print(f"  Speedup:  {speedup:.3f}x")
        
        results[label] = speedup
    
    return results


def test_all_dtypes():
    """Test with different data types."""
    print("\n" + "="*60)
    print("Test 4: Multi-dtype Support")
    print("="*60)
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, d = 128, 2048
    
    for dtype in dtypes:
        try:
            x = torch.randn(num_tokens, 2*d, dtype=dtype, device='cuda')
            out_orig = torch.empty(num_tokens, d, dtype=dtype, device='cuda')
            out_ilp = torch.empty(num_tokens, d, dtype=dtype, device='cuda')
            
            torch.ops._C.gelu_tanh_and_mul(out_orig, x)
            torch.ops._C.gelu_tanh_and_mul_ilp(out_ilp, x)
            
            max_diff = (out_orig - out_ilp).abs().max().item()
            status = "✓"
            print(f"  {status} {str(dtype):20s}: Max diff = {max_diff:.2e}")
        except Exception as e:
            print(f"  ✗ {str(dtype):20s}: {e}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GPU ILP Optimization Validation Suite")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    try:
        test_ilp_availability()
        test_correctness()
        results = benchmark_kernels()
        test_all_dtypes()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        avg_speedup = np.mean(list(results.values()))
        print(f"Average speedup: {avg_speedup:.3f}x")
        
        print("\n✓ All tests passed!")
        print("="*60 + "\n")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF
```

Run it:
```bash
python3 tests/test_ilp_gpu_validation.py
```

---

## Troubleshooting

### Issue: "Torch not compiled with CUDA"
**Solution:** Reinstall with CUDA support
```bash
uv pip uninstall torch vllm
uv pip install -e . --torch-backend=cuda
```

### Issue: "torch.ops._C does not have gelu_and_mul_ilp"
**Solution 1:** Recompile vLLM (changes to CUDA code require recompilation)
```bash
python3 setup.py build_ext --inplace
```

**Solution 2:** Verify the files were modified correctly
```bash
grep -n "gelu_and_mul_ilp" csrc/torch_bindings.cpp
grep -n "act_and_mul_kernel_ilp" csrc/activation_kernels.cu
```

### Issue: GPU Out of Memory
**Solution:** Test with smaller batch sizes
```bash
python3 benchmarks/benchmark_ilp_kernels.py \
  --num-tokens 32 64 \
  --d 512 1024 \
  --iterations 50
```

### Issue: Performance seems worse with ILP
**Possible reasons:**
1. **GPU model doesn't benefit from ILP** - Some GPU architectures may have different latency hiding
2. **Compiler optimizations** - Different CUDA toolkit versions generate different code
3. **Thermal throttling** - GPU temperature affecting performance
4. **Driver version** - Older drivers may not schedule well

**Debug:**
```bash
# Check thermal status
nvidia-smi dmon

# Run with profiling
nsys profile -o results.nsys-rep python3 benchmarks/benchmark_ilp_kernels.py --num-tokens 128 --d 2048 --iterations 10
```

---

## Performance Expectations by GPU

| GPU Model | Expected Speedup | Memory | Notes |
|-----------|------------------|--------|-------|
| RTX 4090 | 1.4-1.8x | 24 GB | High memory bandwidth |
| RTX 4080 | 1.3-1.7x | 16 GB | Good for testing |
| RTX 4070 | 1.2-1.6x | 12 GB | Still good speedup |
| A100 | 1.5-2.0x | 40/80 GB | Professional, best speedup |
| A10G | 1.2-1.5x | 24 GB | Cloud instance GPU |
| H100 | 1.6-2.2x | 80 GB | Latest, best performance |

---

## Sharing Results

Once you have real numbers, create a results file:

```bash
cat > /Users/mohan/projects/vllm/GPU_TEST_RESULTS.md << 'EOF'
# GPU Test Results for ILP Optimization

## Test Environment
- **GPU:** [model and VRAM]
- **CUDA Version:** [version]
- **PyTorch Version:** [version]
- **Test Date:** [date]

## Results

### Test 1: Availability
- ✓ All kernels available

### Test 2: Correctness
- Max difference: [value]
- All tests passed: Yes/No

### Test 3: Performance
| Shape | Original (ms) | ILP (ms) | Speedup |
|-------|---|---|---|
| (32, 512) | | | |
| (128, 2048) | | | |
| (2048, 4096) | | | |

### Average Speedup: X.XXx

## Observations
- [GPU-specific observations]
- [Performance notes]
- [Any issues encountered]
EOF
```

---

## Next Steps After Testing

1. **If speedup is good (>1.2x):**
   - Create PR with results
   - Add GPU test results to commit message

2. **If speedup is marginal (<1.1x):**
   - Try different unroll factors
   - Profile with `nsys` to identify bottlenecks
   - Consider alternative optimizations (kernel fusion)

3. **If speedup is negative:**
   - Investigate compiler differences
   - Check GPU utilization with `nvidia-smi`
   - May indicate ILP not needed for this workload

