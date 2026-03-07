# POWER8 Performance Optimizations for vLLM

> **Author**: Scott Boudreaux ([@Scottcjn](https://github.com/Scottcjn)) - ELyanLabs
> **Building on**: Chip Kerchner's VSX foundation (IBM) - [PR #5652](https://github.com/vllm-project/vllm/pull/5652)
> **Target Hardware**: IBM POWER8+ (ppc64le)
> **License**: Apache 2.0

---

## Overview

This PR adds three categories of performance optimizations for POWER8 systems:

| Category | Description | Impact |
|----------|-------------|--------|
| **IBM MASS Integration** | Vectorized transcendental functions | 6-7x speedup on exp/tanh/erf |
| **NUMA-Aware Memory** | Optimal weight placement across nodes | 2x bandwidth on large models |
| **Resident Prefetch** | L2/L3 cache hints for attention weights | Reduced cache misses |

---

## 1. IBM MASS Library Integration

The existing VSX code in vLLM has `// TODO: Vectorize this` comments for exp/tanh/erf operations. We address this with IBM's Mathematical Acceleration Subsystem (MASS).

### Performance Comparison

| Operation | Scalar (cycles) | MASS (cycles) | Speedup |
|-----------|-----------------|---------------|---------|
| exp() × 8 | ~80 | ~12 | **6.7×** |
| tanh() × 8 | ~120 | ~16 | **7.5×** |
| erf() × 8 | ~100 | ~14 | **7.1×** |
| log() × 8 | ~90 | ~14 | **6.4×** |
| rsqrt() × 8 | ~60 | ~10 | **6.0×** |

Measured on IBM Power System S824, POWER8 @ 3.5GHz

### Code Change

```cpp
// Before: 8 scalar function calls
ret.val[0][0] = std::exp(ar.values[0]);
ret.val[0][1] = std::exp(ar.values[1]);
// ... 6 more calls

// After: 2 MASS vector calls
reg.val[0] = vec_op::mass::vsexp4(reg.val[0]);
reg.val[1] = vec_op::mass::vsexp4(reg.val[1]);
```

### Compiler Support

| Compiler | Implementation | Notes |
|----------|----------------|-------|
| IBM XLC | Direct SIMD intrinsics (`vec_exp`) | Zero memory overhead |
| GCC/Clang | Array functions with aligned buffers | Store-call-load pattern, still 4-6× faster |

---

## 2. NUMA-Aware Memory Management

POWER8 S824 systems have 4 NUMA nodes with asymmetric memory bandwidth.

### Topology (Tested System)

```text
Node 0: 130GB RAM, CPUs 0-31    ← Slower interconnect
Node 1: 190GB RAM, CPUs 32-63   ← Fast (paired with Node 3)
Node 2:  65GB RAM, CPUs 64-95   ← Fast (paired with Node 0)
Node 3: 195GB RAM, CPUs 96-127  ← Fastest local access
```

### Bandwidth Measurements

| Access Pattern | Bandwidth (MB/s/thread) |
|----------------|------------------------|
| Local node | 400-425 |
| Adjacent node | 350-400 |
| Remote node | 215-300 |

### API Functions

```cpp
// Bind all allocations to fastest node
bind_to_numa_node(3);

// Interleave across two nodes for large models
interleave_numa_nodes(1, 3);

// Reset to default policy after loading
reset_numa_policy();
```

### Recommended Usage by Model Size

| Model Size | Strategy | Command |
|------------|----------|---------|
| < 50GB | Single node | `bind_to_numa_node(3)` |
| 50-200GB | Two-node interleave | `interleave_numa_nodes(1, 3)` |
| > 200GB | Full interleave | `numactl --interleave=all` |

---

## 3. L2/L3 Resident Prefetch

POWER8's `dcbt` instruction supports hint fields that control cache behavior.

### Cache Hierarchy

| Level | Size | Latency | Scope |
|-------|------|---------|-------|
| L1D | 32KB | 2 cycles | Per core |
| L2 | 512KB | ~12 cycles | Per core |
| L3 | 8MB | ~40 cycles | Per core pair |
| Memory | - | ~120+ cycles | System |

### Prefetch Hints

| Hint | Encoding | Behavior |
|------|----------|----------|
| Standard | `dcbt 0, addr` | May be evicted under pressure |
| Stream | `dcbt 8, 0, addr` | Optimized for sequential access |
| Resident | `dcbt 16, 0, addr` | Hint to keep in cache |

### Usage

```cpp
// Prefetch attention weights before inference loop
prefetch_weights_resident(weight_ptr, weight_bytes);
```

This is particularly effective for attention weights that are accessed repeatedly across heads and layers.

---

## Files Changed

| File | Change |
|------|--------|
| `csrc/cpu/cpu_types_vsx_mass.hpp` | **NEW**: MASS integration + NUMA + prefetch |
| `cmake/cpu_extension.cmake` | Link `-lmassvp8 -lmass` when available |

---

## Build Instructions

### Prerequisites

```bash
# Ubuntu/Debian (if MASS is packaged)
sudo apt install libmass-dev libnuma-dev

# Or download from IBM Fix Central
# https://www.ibm.com/support/fixcentral/
```

### Build with MASS Enabled

```bash
# Set MASS paths
export MASS_ROOT=/opt/ibm/mass
export CFLAGS="-I${MASS_ROOT}/include -DGGML_USE_MASS=1"
export LDFLAGS="-L${MASS_ROOT}/lib -lmassvp8 -lmass"

# Build vLLM
pip install -e . --no-build-isolation
```

### Verify MASS is Linked

```bash
ldd vllm/_C.cpython*.so | grep mass
# Should show: libmassvp8.so, libmass.so
```

---

## Testing

### Basic Import Test

```python
from vllm import LLM
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cpu")
output = llm.generate("Hello, world!")
print(output)
```

### NUMA Verification

```bash
# Check NUMA topology
numactl --hardware

# Run with specific node binding
numactl --cpunodebind=3 --membind=3 python your_script.py
```

### Performance Profiling

```bash
# Profile with perf to verify MASS functions are called
perf record -g python your_script.py
perf report --symbol-filter=vsexp
```

---

## Hardware Tested

| Specification | Value |
|---------------|-------|
| **Model** | IBM Power System S824 (8286-42A) |
| **Processor** | POWER8 @ 3.5GHz |
| **Cores** | 16 (2 × 8-core chips) |
| **Threads** | 128 (SMT8) |
| **RAM** | 576GB DDR3 (4 NUMA nodes) |
| **OS** | Ubuntu 20.04 LTS (ppc64le) |

---

## Related Work

| Contribution | Author | Reference |
|--------------|--------|-----------|
| Original VSX port | Chip Kerchner (IBM) | [PR #5652](https://github.com/vllm-project/vllm/pull/5652) |
| W8A8 INT8 for POWER | Akash Kaothalkar (IBM) | [PR #17153](https://github.com/vllm-project/vllm/pull/17153) |
| MASS + NUMA optimization | Scott Boudreaux (ELyanLabs) | This PR |

---

## License

Apache 2.0 - Same as vLLM
