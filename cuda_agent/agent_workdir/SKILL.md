# CUDA Kernel Optimization Skill

## Objective
Accelerate a PyTorch model via custom CUDA C++ extensions, achieving **minimum 5% speedup** over
`torch.compile` baseline while maintaining numerical correctness.

---

## Critical Constraints

### Strictly Forbidden
- Using PyTorch operators (e.g. `torch.*`, `F.*`, `nn.*`) inside C++ bindings or CUDA files
- Any third-party libraries except:
  - `cuBLAS` for GEMM operations
  - `cuDNN` for convolution operations
- Modifying `utils/`, `binding.cpp`, or `binding_registry.h`

### Allowed Operations
- Raw CUDA kernels in the `kernels/` directory
- `cuBLAS` / `cuDNN` for appropriate mathematical operations
- Tensor creation and custom extension ops in Python (`model_new.py`)

---

## Workflow

### 1. Understand the Task
Read `model.py` to understand:
- The neural network architecture (`Model` class)
- Input shapes via `get_inputs()`
- Initialization parameters via `get_init_inputs()`

### 2. Implement Custom CUDA Kernels
For each operation, create **paired** files:
- `kernels/<op_name>.cu` — CUDA kernel implementation
- `kernels/<op_name>_binding.cpp` — PyTorch tensor wrapper (uses `REGISTER_BINDING` macro)

Example kernel structure:
```cuda
// kernels/my_op.cu
#include <cuda_runtime.h>

template <typename T>
__global__ void my_op_kernel(const T* a, T* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = /* custom op */;
    }
}

void my_op_launcher(const float* a, float* out, int n,
                    int config, cudaStream_t stream) {
    int threads = (config == 1) ? 128 : (config == 2) ? 512 : 256;
    int blocks = (n + threads - 1) / threads;
    my_op_kernel<float><<<blocks, threads, 0, stream>>>(a, out, n);
}
```

Example binding structure:
```cpp
// kernels/my_op_binding.cpp
#include <torch/extension.h>
#include "../binding_registry.h"

extern void my_op_launcher(const float*, float*, int, int, cudaStream_t);

torch::Tensor my_op_forward(torch::Tensor a, int config = 0) {
    TORCH_CHECK(a.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(a.is_contiguous(), "Input must be contiguous");
    auto out = torch::empty_like(a);
    my_op_launcher(a.data_ptr<float>(), out.data_ptr<float>(),
                   a.numel(), config,
                   at::cuda::getCurrentCUDAStream());
    return out;
}

REGISTER_BINDING(my_op_forward,
    [](pybind11::module& m) { m.def("my_op_forward", &my_op_forward); })
```

### 3. Update model_new.py
Modify `ModelNew` in `model_new.py` to call your custom CUDA extension:
```python
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Store parameters

    def forward(self, *args):
        return cuda_extension.my_op_forward(args[0])
```

### 4. Compile
```bash
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
```

### 5. Verify Correctness
```bash
sudo python3 -m utils.verification
```
Must pass with `atol=1e-2, rtol=1e-2`. The verifier **blocks** `torch.nn.functional` calls
in `ModelNew` to ensure genuine CUDA kernel usage.

### 6. Profile Performance
```bash
sudo python3 -m utils.profiling
```
Output format:
```
Torch Baseline: Xus, Torch Compile: Yus, CUDA Extension: Zus
```
Target: CUDA Extension time ≤ 95% of Torch Compile time.

### 7. Cleanup
Remove any intermediate or failed kernel files. Only keep the **final working version** in `kernels/`.

---

## Optimization Priorities

| Priority | Technique | Expected Impact |
|----------|-----------|-----------------|
| **1 (High)** | Kernel fusion, memory tiling, access coalescing | >50% speedup |
| **2 (Medium)** | Vectorized loads (`float4`), warp primitives, occupancy tuning | 20–50% speedup |
| **3 (Low)** | Instruction-level parallelism, mixed precision | <20% speedup |

### Key Techniques
- **Memory Coalescing**: Ensure threads in a warp access consecutive memory
- **Shared Memory Tiling**: Use `__shared__` to cache frequently accessed data
- **Kernel Fusion**: Combine multiple operations into a single kernel launch
- **Warp Shuffle**: Use `__shfl_*` for intra-warp reduction
- **Vectorized Loads**: Use `float4` / `float2` for wider memory transactions

---

## Success Metrics

| Metric | Threshold |
|--------|-----------|
| Correctness | `atol=1e-2, rtol=1e-2` (5 verification passes) |
| Minimum speedup | CUDA Extension ≤ 95% of torch.compile baseline time |
| Target | Maximum achievable speedup |
| Code quality | Clean `kernels/` directory with only final implementation |
