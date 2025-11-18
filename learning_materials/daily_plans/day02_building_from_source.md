# Day 2: Building vLLM from Source & Debugging Setup

> **Goal**: Master building vLLM from source with debug symbols, set up debugging tools, run first inference with profiling
> **Time**: 6-8 hours
> **Prerequisites**: Day 1 completed, basic C++/CUDA knowledge, VS Code or preferred IDE
> **Deliverables**: Debug build of vLLM, working debugger setup, profiling results from first inference

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Build Setup

**9:00-9:30** - Clean Build Environment Setup
**9:30-11:00** - Build vLLM from Source with Debug Symbols
**11:00-11:30** - Break + Verify Build
**11:30-12:30** - Configure Debugging Tools (gdb, cuda-gdb, VS Code)

### Afternoon Session (3-4 hours): Hands-On Practice

**14:00-15:00** - Run First Inference with Debugging
**15:00-16:00** - Introduction to Profiling (Nsight Systems)
**16:00-16:30** - Break
**16:30-18:00** - Hands-On Exercises & Experiments

### Evening (Optional, 1 hour): Advanced Setup

**19:00-20:00** - Install profiling tools, explore build options, prepare for Day 3

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Build vLLM from source with debug symbols enabled
- [ ] Set up gdb and cuda-gdb for debugging
- [ ] Configure VS Code (or IDE) for C++/CUDA debugging
- [ ] Set breakpoints in both Python and C++ code
- [ ] Run basic profiling with Nsight Systems
- [ ] Understand the build system (CMake, setup.py, ninja)

---

## 🛠️ Morning: Building from Source (9:00-12:30)

### Task 1: Clean Build Environment (30 min)

**Prepare your environment**:

```bash
# Navigate to vLLM directory
cd ~/vllm

# Remove any previous builds
pip uninstall vllm -y
rm -rf build/ dist/ *.egg-info

# Verify CUDA toolkit
nvcc --version
# Expected: CUDA 11.8+ or 12.1+

# Check Python version
python --version
# Expected: Python 3.8+

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected: PyTorch 2.0+, CUDA: True
```

**📝 Environment Variables**:

```bash
# Set up for debug build
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Enable debug symbols
export CFLAGS="-g -O0"
export CXXFLAGS="-g -O0"
export CUDAFLAGS="-g -G"  # -G disables optimizations for debugging

# Increase verbosity
export VERBOSE=1
export MAX_JOBS=8  # Parallel build jobs

# Save to ~/.bashrc for persistence
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
```

### Task 2: Build vLLM with Debug Symbols (90 min)

**Understanding the build process**:

vLLM uses:
- `setup.py` for Python packaging
- `CMake` for C++ compilation
- `ninja` as build backend (faster than make)

**File**: `setup.py` (root directory)

```python
# Key build configuration in setup.py
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name='vllm._C',  # ← C++ module name
        sources=[
            'csrc/cache_kernels.cu',
            'csrc/attention/attention_kernels.cu',
            'csrc/pos_encoding_kernels.cu',
            # ... more sources
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],  # ← Change to -O0 -g for debug
            'nvcc': ['-O3', '--use_fast_math'],  # ← Change to -O0 -g -G
        }
    )
]
```

**Build Command** (Development Mode):

```bash
# Install in editable mode (changes reflect immediately)
pip install -e . -v

# This will:
# 1. Compile C++/CUDA extensions
# 2. Link with PyTorch
# 3. Install Python package

# Expected output:
# Building wheel vllm-0.x.x
# Running setup.py develop for vllm
# Compiling csrc/cache_kernels.cu
# Compiling csrc/attention/attention_kernels.cu
# ...
# Successfully installed vllm
```

**⏱️ Build Time**: 10-20 minutes (depending on hardware)

**Common Build Issues**:

```bash
# Issue 1: CUDA not found
# Solution: Set CUDA_HOME correctly

# Issue 2: PyTorch version mismatch
# Solution: pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Issue 3: Out of memory during compilation
# Solution: export MAX_JOBS=4  # Reduce parallel jobs

# Issue 4: ninja not found
# Solution: pip install ninja
```

**Verify Build**:

```bash
# Test import
python -c "import vllm; print(vllm.__version__)"

# Test CUDA extensions
python -c "from vllm import _C; print('C++ extensions loaded successfully')"

# Check symbols
nm -C build/lib.linux-x86_64-cpython-*/vllm/_C*.so | grep paged_attention
# Should show debug symbols if built correctly
```

### Task 3: Configure Debugging Tools (60 min)

#### **Setup 1: GDB for C++ Debugging**

```bash
# Install gdb if not present
sudo apt-get install gdb

# Create .gdbinit in home directory
cat > ~/.gdbinit << 'EOF'
# Pretty printing for STL containers
set print pretty on
set print array on
set print array-indexes on

# Python pretty printing
python
import sys
sys.path.insert(0, '/usr/share/gdb/auto-load')
end

# Skip system libraries
skip -gfi /usr/*
skip -gfi /lib/*
EOF

# Test gdb
gdb python
# (gdb) quit
```

#### **Setup 2: cuda-gdb for CUDA Debugging**

```bash
# Verify cuda-gdb installation
cuda-gdb --version

# Create ~/.cuda-gdbinit
cat > ~/.cuda-gdbinit << 'EOF'
# Break on CUDA errors
set cuda memcheck on
set cuda api_failures stop

# Show kernel info
set cuda kernel_events on
EOF
```

#### **Setup 3: VS Code Configuration**

Create `.vscode/launch.json` in vllm directory:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,  // ← Debug into vLLM code
            "env": {
                "CUDA_LAUNCH_BLOCKING": "1"  // ← Synchronous CUDA for debugging
            }
        },
        {
            "name": "C++: Attach to Python",
            "type": "cppdbg",
            "request": "attach",
            "processId": "${command:pickProcess}",
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        },
        {
            "name": "CUDA: Debug Kernels",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "/usr/bin/python",
            "args": ["${file}"],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [
                {"name": "CUDA_LAUNCH_BLOCKING", "value": "1"}
            ]
        }
    ]
}
```

**Install VS Code Extensions**:
- Python (Microsoft)
- C/C++ (Microsoft)
- Nsight Visual Studio Code Edition (NVIDIA)

---

## 🔬 Afternoon: Debugging & Profiling (14:00-18:00)

### Task 4: Run First Inference with Debugging (60 min)

**Create debug test script**: `debug_test.py`

```python
#!/usr/bin/env python3
"""
Day 2 Exercise: Debug vLLM execution
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous execution

from vllm import LLM, SamplingParams

def main():
    print("🔍 Starting debug session")

    # Small model for faster debugging
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=256,
        gpu_memory_utilization=0.3,
        dtype="float32",  # Easier to debug than float16
    )

    # Single prompt for simplicity
    prompt = "Hello, my name is"
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=10
    )

    print("🚀 Running inference...")
    outputs = llm.generate([prompt], sampling_params)

    print(f"✅ Output: {outputs[0].outputs[0].text}")
    print(f"📊 Tokens: {outputs[0].outputs[0].token_ids}")

if __name__ == "__main__":
    main()
```

**Debugging Exercise 1: Python Breakpoint**

```python
# Add breakpoint in vLLM code
# File: vllm/engine/llm_engine.py

def step(self):
    # Add this line
    import pdb; pdb.set_trace()

    scheduler_outputs = self.scheduler.schedule()
    # ...
```

Run and explore:
```bash
python debug_test.py

# When breakpoint hits:
# (Pdb) p scheduler_outputs
# (Pdb) p self.scheduler.running
# (Pdb) p self.scheduler.waiting
# (Pdb) n  # Next line
# (Pdb) c  # Continue
```

**Debugging Exercise 2: C++ Breakpoint**

```bash
# Start with gdb
gdb --args python debug_test.py

# In gdb:
(gdb) break paged_attention_v1_kernel  # Break in CUDA kernel launcher
(gdb) run

# When hit:
(gdb) backtrace  # Show call stack
(gdb) info locals  # Show local variables
(gdb) continue
```

**Understanding the call stack**:

```
#0  paged_attention_v1_kernel at csrc/attention/attention_kernels.cu:123
#1  paged_attention_v1 at vllm/attention/ops/paged_attn.py:45
#2  forward at vllm/attention/backends/flash_attn.py:67
#3  execute_model at vllm/model_executor/model_runner.py:234
#4  step at vllm/engine/llm_engine.py:156
#5  generate at vllm/entrypoints/llm.py:89
```

### Task 5: Introduction to Profiling (60 min)

**Tool**: Nsight Systems (nsys)

```bash
# Install if not present
# Download from: https://developer.nvidia.com/nsight-systems

# Verify installation
nsys --version
```

**Profile your first run**:

```bash
# Basic profile
nsys profile \
    --output=vllm_first_run \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    python debug_test.py

# Output: vllm_first_run.nsys-rep
```

**Open in GUI**:

```bash
# If on remote server, download the .nsys-rep file
scp user@server:~/vllm/vllm_first_run.nsys-rep .

# Open in Nsight Systems GUI (on local machine)
nsys-ui vllm_first_run.nsys-rep
```

**📊 What to Look For**:

1. **CUDA Kernels**: Timeline showing kernel execution
   - `paged_attention_v1_kernel`
   - `copy_blocks_kernel`
   - GEMM operations (matrix multiplication)

2. **CPU Activity**: Python/C++ execution
   - Scheduling overhead
   - Data transfer time

3. **Memory Transfers**: Host ↔ Device
   - Should be minimal if done correctly

4. **Gaps/Idle Time**: Optimization opportunities

**Understanding the Timeline**:

```
Time (ms) →
0    50   100  150  200  250  300
|----|----|----|----|----|----|
[Model Load.....................]
                [Schedule]
                [Copy Input to GPU]
                [GEMM: Q×K^T]
                [Softmax]
                [GEMM: Attn×V]
                [Copy Output]
                         [Sample Token]
                         [Repeat...]
```

### Task 6: Hands-On Exercises (90 min)

**Exercise 1: Build with Different Optimization Levels**

```bash
# Clean previous build
rm -rf build/

# Build 1: Debug (-O0)
export CXXFLAGS="-g -O0"
pip install -e . -v

# Time inference
time python debug_test.py
# Note: ~X seconds

# Build 2: Optimized (-O3)
rm -rf build/
export CXXFLAGS="-g -O3"
pip install -e . -v

# Time again
time python debug_test.py
# Note: ~Y seconds (should be faster)

# Compare: How much faster is optimized build?
```

**Exercise 2: Add Custom Profiling Markers**

```python
# File: debug_test_with_nvtx.py

import torch.cuda.nvtx as nvtx

def main():
    nvtx.range_push("Model Initialization")
    llm = LLM(model="facebook/opt-125m")
    nvtx.range_pop()

    nvtx.range_push("Inference")
    outputs = llm.generate([prompt], sampling_params)
    nvtx.range_pop()

    # These markers will show in Nsight Systems!
```

Profile with markers:
```bash
nsys profile --output=vllm_with_nvtx python debug_test_with_nvtx.py
```

**Exercise 3: Trace Memory Allocation**

```python
# File: memory_trace.py

import torch

def main():
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    llm = LLM(model="facebook/opt-125m")
    print(f"After model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    print(f"After inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Expected output:
    # Initial: 0.00 GB
    # After load: 0.24 GB (model weights)
    # After inference: 0.26 GB (+ KV cache)
```

**Exercise 4: Compare Batch Sizes**

```python
# File: batch_comparison.py

import time
from vllm import LLM, SamplingParams

def benchmark_batch(batch_size):
    llm = LLM(model="facebook/opt-125m", max_model_len=256)
    prompts = ["Hello, my name is"] * batch_size

    start = time.time()
    outputs = llm.generate(prompts, SamplingParams(max_tokens=20))
    elapsed = time.time() - start

    throughput = batch_size / elapsed
    print(f"Batch {batch_size}: {elapsed:.2f}s, {throughput:.2f} req/s")

# Run for different batch sizes
for bs in [1, 4, 8, 16, 32]:
    benchmark_batch(bs)

# Analyze: Does throughput scale linearly?
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Build System**
- Building vLLM from source with debug symbols
- Understanding setup.py and CMake configuration
- Debugging common build issues

✅ **Debugging Tools**
- gdb for C++ debugging
- cuda-gdb for CUDA kernel debugging
- VS Code configuration for mixed Python/C++ debugging

✅ **Profiling Basics**
- Using Nsight Systems for performance analysis
- Understanding kernel timelines
- Adding NVTX markers for custom profiling

✅ **Practical Skills**
- Setting breakpoints in both Python and C++
- Tracing execution through call stacks
- Measuring performance with different configurations

### Knowledge Check (Quiz)

**Question 1**: What compiler flag enables CUDA kernel debugging?
<details>
<summary>Answer</summary>
`-G` flag for nvcc disables optimizations and enables device code debugging. Full command: `nvcc -g -G kernel.cu`
</details>

**Question 2**: How do you install vLLM in development mode?
<details>
<summary>Answer</summary>
`pip install -e .` (editable install) - changes to Python code take effect immediately without reinstalling. C++/CUDA changes require recompilation.
</details>

**Question 3**: What environment variable makes CUDA execution synchronous for easier debugging?
<details>
<summary>Answer</summary>
`export CUDA_LAUNCH_BLOCKING=1` - forces CPU to wait for each CUDA kernel to complete before continuing, making errors easier to locate.
</details>

**Question 4**: In Nsight Systems, what are the main types of events you can trace?
<details>
<summary>Answer</summary>
- CUDA: kernel launches, memory transfers
- NVTX: custom markers/ranges
- OS Runtime (osrt): CPU threads, syscalls
- Python: function calls (with python trace)
</details>

**Question 5**: Why might a debug build be 10x slower than optimized?
<details>
<summary>Answer</summary>
Debug builds (-O0 -g -G):
- No compiler optimizations
- Extra bounds checking
- Symbol tables included
- Disabled GPU optimizations
All necessary for debugging but impact performance.
</details>

### Daily Reflection

**What went well?**
- [ ] Successfully built vLLM from source
- [ ] Set up debugging tools
- [ ] Ran first profiling session

**What was challenging?**
- [ ] Build system complexity
- [ ] Configuring multi-language debugging
- [ ] Understanding profiler output

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 3

Tomorrow you'll dive into:
- **Complete Request Lifecycle**: From API call to token output
- **Detailed Code Walkthrough**: Line-by-line through key components
- **Scheduler Deep Dive**: How requests are batched and scheduled
- **First Modification**: Add custom logging to trace requests

**Preparation**:
- Review your notes on LLMEngine.step() from Day 1
- Read `vllm/core/scheduler.py` header comments
- Ensure your debug build is working

---

## 📚 Additional Resources

**Documentation**:
- [ ] [vLLM Development Guide](https://docs.vllm.ai/en/latest/dev/contributing.html)
- [ ] [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [ ] [GDB Quick Reference](https://darkdust.net/files/GDB%20Cheat%20Sheet.pdf)

**Build System Deep Dive**:
- [ ] Read `setup.py` completely
- [ ] Explore `CMakeLists.txt` if present
- [ ] Check `pyproject.toml` for dependencies

**Optional Practice**:
- [ ] Profile different models (opt-125m vs opt-1.3b)
- [ ] Experiment with different CUDA architectures
- [ ] Try building with different PyTorch versions

---

**Congratulations on mastering the build system! 🎉**

**You now have the tools to debug and profile vLLM like a pro!**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
