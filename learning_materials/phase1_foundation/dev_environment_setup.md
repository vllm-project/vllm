# 🛠️ vLLM Development Environment Setup Guide

> **Goal**: Set up a complete development environment for learning vLLM internals
> **Time Required**: 2-4 hours (depending on your system)
> **Target**: Ubuntu 20.04/22.04 with NVIDIA GPU (can adapt for other systems)

---

## 📋 Prerequisites

Before starting:
- [ ] NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer)
- [ ] Ubuntu 20.04 or 22.04 (or similar Linux distribution)
- [ ] At least 32GB RAM (64GB recommended for large models)
- [ ] 100GB free disk space
- [ ] Sudo/root access for system packages

---

## 🎯 Setup Objectives

You'll set up:
1. ✅ **Build tools**: Compilers, CUDA toolkit, CMake
2. ✅ **Python environment**: Python 3.10+, venv/conda
3. ✅ **vLLM from source**: Debug build with symbols
4. ✅ **IDE configuration**: VSCode with C++/CUDA/Python support
5. ✅ **Profiling tools**: Nsight Systems, Nsight Compute
6. ✅ **Testing infrastructure**: Pytest, test models
7. ✅ **Debugging tools**: GDB for C++, pdb for Python

---

## Step 1: System Dependencies

### Install Build Essentials

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ninja-build \
    ccache \
    pkg-config

# Install Python development headers
sudo apt install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip

# Verify installations
gcc --version      # Should be 9.x or higher
cmake --version    # Should be 3.18 or higher
python3.10 --version
```

**Expected Output**:
```
gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
cmake version 3.22.1
Python 3.10.12
```

---

## Step 2: NVIDIA CUDA Toolkit

### Install CUDA 12.1 (or compatible version)

```bash
# Download CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Install (this will take 10-15 minutes)
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

**Expected Output**:
```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.1, V12.1.105
```

### Troubleshooting CUDA Installation

**Issue**: `nvidia-smi` not found
```bash
# Install NVIDIA driver
sudo apt install nvidia-driver-530
sudo reboot
```

**Issue**: CUDA version mismatch
```bash
# Check installed CUDA versions
ls -l /usr/local/ | grep cuda
# Use the appropriate version in your PATH
```

---

## Step 3: Python Environment Setup

### Option A: Using venv (Recommended for Learning)

```bash
# Create virtual environment
cd ~
python3.10 -m venv vllm-env

# Activate environment
source ~/vllm-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Verify
which python  # Should point to vllm-env
python --version  # Should be 3.10.x
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n vllm-env python=3.10 -y
conda activate vllm-env

# Install pip
conda install pip -y
```

### Install PyTorch with CUDA Support

```bash
# For CUDA 12.1
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Expected Output**:
```
PyTorch: 2.1.2+cu121
CUDA available: True
CUDA version: 12.1
```

---

## Step 4: Build vLLM from Source

### Clone vLLM Repository

```bash
# Clone the repo
cd ~/projects  # or your preferred directory
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Check out specific version (optional, for stability)
git checkout v0.6.0  # or use main for latest

# View directory structure
ls -la
```

### Build vLLM with Debug Symbols

```bash
# Set build flags for debugging
export VLLM_INSTALL_PUNICA_KERNELS=1
export MAX_JOBS=8  # Adjust based on your CPU cores
export CUDAARCHS="80"  # For A100, use "70" for V100, "89" for RTX 4090

# Build with debug symbols and verbose output
NVCC_PREPEND_FLAGS='-Xcompiler -g' \
CFLAGS="-g -O2" \
CMAKE_BUILD_TYPE=RelWithDebInfo \
VLLM_TARGET_DEVICE=cuda \
pip install -e . -v

# This will take 20-40 minutes depending on your hardware
```

**Build Flags Explained**:
- `-e`: Editable install (changes to code reflected immediately)
- `-g`: Debug symbols for GDB
- `RelWithDebInfo`: Optimized with debug info
- `-v`: Verbose output to see compilation

### Verify vLLM Installation

```bash
# Test import
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Run simple inference test
python -c "
from vllm import LLM
# This will download a small model
llm = LLM(model='facebook/opt-125m', max_model_len=512)
output = llm.generate('Hello, my name is', max_tokens=20)
print(output[0].outputs[0].text)
"
```

**Expected Output**:
```
vLLM version: 0.6.0+
Processed prompts: 100%|████████| 1/1 [00:01<00:00,  1.23s/it]
```

### Rebuild After Code Changes

```bash
# Quick rebuild (only changed files)
pip install -e . -v

# Full rebuild (if you encounter issues)
rm -rf build/
pip install -e . --no-build-isolation -v
```

---

## Step 5: IDE Setup - VSCode

### Install VSCode

```bash
# Download and install VSCode
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code -y

# Launch VSCode
code ~/projects/vllm
```

### Install Essential VSCode Extensions

```bash
# Install extensions via command line
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cpptools-extension-pack
code --install-extension ms-vscode.cmake-tools
code --install-extension nvidia.nsight-vscode-edition
code --install-extension GitHub.copilot  # Optional but helpful

# Restart VSCode
```

### Configure VSCode Settings

Create `.vscode/settings.json` in vLLM directory:

```bash
mkdir -p ~/projects/vllm/.vscode
cat > ~/projects/vllm/.vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "~/vllm-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.analysis.typeCheckingMode": "basic",

    "C_Cpp.default.compilerPath": "/usr/local/cuda/bin/nvcc",
    "C_Cpp.default.includePath": [
        "${workspaceFolder}/csrc",
        "/usr/local/cuda/include",
        "${workspaceFolder}/build"
    ],
    "C_Cpp.default.defines": [
        "__CUDACC__",
        "CUDA_VERSION=12010"
    ],

    "files.associations": {
        "*.cu": "cuda-cpp",
        "*.cuh": "cuda-cpp"
    },

    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "files.trimTrailingWhitespace": true
}
EOF
```

### Create VSCode Launch Configuration for Debugging

Create `.vscode/launch.json`:

```bash
cat > ~/projects/vllm/.vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        },
        {
            "name": "Python: vLLM Inference Test",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/examples/offline_inference.py",
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "C++: Attach to Python Process",
            "type": "cppdbg",
            "request": "attach",
            "program": "~/vllm-env/bin/python",
            "processId": "${command:pickProcess}",
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
EOF
```

---

## Step 6: Profiling Tools Setup

### Install Nsight Systems

```bash
# Download Nsight Systems
cd ~/Downloads
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2023_4_1/nsight-systems-2023.4.1_linux-public.deb

# Install
sudo dpkg -i nsight-systems-2023.4.1_linux-public.deb
sudo apt --fix-broken install -y

# Verify
nsys --version
```

### Install Nsight Compute

```bash
# Download Nsight Compute
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2023_3_0/nsight-compute-linux-2023.3.0.15-33411502.run

# Install
sudo sh nsight-compute-linux-2023.3.0.15-33411502.run --accept --installpath=/usr/local/nsight-compute

# Add to PATH
echo 'export PATH=/usr/local/nsight-compute:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
ncu --version
```

### Profile vLLM with Nsight Systems

```bash
# Profile a simple inference run
nsys profile \
    --output=vllm_profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    python examples/offline_inference.py

# View profile (opens GUI)
nsys-ui vllm_profile.nsys-rep

# Or generate text report
nsys stats vllm_profile.nsys-rep
```

### Profile Specific Kernel with Nsight Compute

```bash
# Profile all kernels
ncu --set full --export vllm_kernel_profile python examples/offline_inference.py

# Profile specific kernel pattern
ncu --kernel-name paged_attention --set full python examples/offline_inference.py

# View in GUI
ncu-ui vllm_kernel_profile.ncu-rep
```

---

## Step 7: Testing Infrastructure

### Install Testing Dependencies

```bash
# Install pytest and plugins
pip install pytest pytest-asyncio pytest-benchmark pytest-cov pytest-xdist

# Install development dependencies
pip install -r requirements-dev.txt

# Optional: Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Run vLLM Test Suite

```bash
# Run all tests (takes ~30 minutes)
pytest tests/

# Run specific test file
pytest tests/core/test_block_manager.py -v

# Run with coverage
pytest tests/ --cov=vllm --cov-report=html

# Run tests in parallel (faster)
pytest tests/ -n auto

# Run only CUDA tests
pytest tests/ -k "cuda" -v
```

### Create Custom Test Script

Create `~/projects/vllm/my_test.py`:

```python
#!/usr/bin/env python3
"""
Quick test script for vLLM experimentation
"""
import torch
from vllm import LLM, SamplingParams

def test_basic_inference():
    """Test basic inference"""
    # Create LLM
    llm = LLM(model="facebook/opt-125m", max_model_len=512, gpu_memory_utilization=0.5)

    # Prepare prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In a galaxy far, far away",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated: {generated_text!r}\n")

if __name__ == "__main__":
    test_basic_inference()
    print("✅ Test passed!")
```

Run it:
```bash
python my_test.py
```

---

## Step 8: Debugging Setup

### GDB for C++/CUDA Debugging

```bash
# Install GDB
sudo apt install gdb -y

# Create GDB init file for better output
cat > ~/.gdbinit << 'EOF'
set print pretty on
set print object on
set print static-members on
set print vtbl on
set print demangle on
set demangle-style gnu-v3
set print sevenbit-strings off

# CUDA-specific
set cuda break_on_launch application
EOF

# Test GDB with Python
gdb --args python my_test.py
# Inside GDB:
# (gdb) break csrc/attention/attention_kernels.cu:123
# (gdb) run
# (gdb) bt  # backtrace
```

### Using cuda-gdb for Kernel Debugging

```bash
# Launch with cuda-gdb
cuda-gdb --args python my_test.py

# Example debugging session
(cuda-gdb) break paged_attention_v1_kernel
(cuda-gdb) run
(cuda-gdb) cuda thread (0,0,0)  # Switch to specific thread
(cuda-gdb) print query[0]
(cuda-gdb) info cuda threads
```

### Python Debugging with pdb

Add to your Python code:
```python
import pdb; pdb.set_trace()  # Debugger will stop here
```

Or use iPython debugger (better):
```bash
pip install ipdb

# In code:
import ipdb; ipdb.set_trace()
```

---

## Step 9: Download Test Models

### Download Small Models for Testing

```bash
# Create models directory
mkdir -p ~/models
cd ~/models

# Download using HuggingFace CLI
pip install huggingface_hub

# Small models for quick testing
huggingface-cli download facebook/opt-125m
huggingface-cli download facebook/opt-1.3b
huggingface-cli download gpt2

# Medium models for realistic testing
huggingface-cli download meta-llama/Llama-2-7b-hf  # Requires HF token
huggingface-cli download mistralai/Mistral-7B-v0.1

# Set HuggingFace cache location (optional)
echo 'export HF_HOME=~/models' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 10: Create Useful Aliases and Scripts

### Add to ~/.bashrc

```bash
cat >> ~/.bashrc << 'EOF'

# vLLM Development Aliases
alias vllm-env='source ~/vllm-env/bin/activate'
alias vllm-cd='cd ~/projects/vllm'
alias vllm-build='pip install -e . -v'
alias vllm-test='pytest tests/ -v'
alias vllm-profile='nsys profile --trace=cuda,nvtx --output=profile'

# CUDA Utilities
alias cuda-watch='watch -n 1 nvidia-smi'
alias cuda-mem='nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# Quick vLLM test
vllm-quick-test() {
    python -c "from vllm import LLM; llm = LLM(model='facebook/opt-125m'); print(llm.generate('Hello', max_tokens=10))"
}

# Profile and view
vllm-prof() {
    nsys profile --output=profile_$1 --trace=cuda,nvtx python $1
    nsys-ui profile_$1.nsys-rep &
}

EOF

source ~/.bashrc
```

---

## Step 11: Verification Checklist

Run through this checklist to ensure everything works:

### System Check
```bash
# Check CUDA
nvidia-smi
nvcc --version

# Check Python
python --version
which python  # Should point to virtual environment

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check vLLM
python -c "import vllm; print(vllm.__version__)"
```

### Build Check
```bash
# Rebuild vLLM
cd ~/projects/vllm
pip install -e . -v

# Run single test
pytest tests/core/test_block_manager.py::test_block_allocation -v
```

### IDE Check
```bash
# Open VSCode
code ~/projects/vllm

# Verify:
# - Python extension recognizes vllm-env
# - C++ IntelliSense works in csrc/
# - Can set breakpoint and debug Python
```

### Profiling Check
```bash
# Quick profile
nsys profile --trace=cuda python my_test.py
ls -lh *.nsys-rep  # Should see profile file

# Kernel profile
ncu --kernel-name ".*" --launch-count 1 python my_test.py
```

---

## 🎓 What You've Accomplished

✅ **Complete C++/CUDA development environment**
✅ **vLLM built from source with debug symbols**
✅ **IDE configured for Python and C++ development**
✅ **Profiling tools installed and tested**
✅ **Testing infrastructure ready**
✅ **Debugging tools configured**
✅ **Test models downloaded**

---

## 🚀 Next Steps

1. **Verify** everything works with the checklist above
2. **Explore** the vLLM codebase structure
3. **Run** examples from `examples/` directory
4. **Start** Day 1 learning plan: `daily_plans/day01_codebase_overview.md`

---

## 🐛 Common Issues & Solutions

### Issue: CUDA out of memory during build
**Solution**:
```bash
# Reduce parallel jobs
export MAX_JOBS=2
pip install -e . -v
```

### Issue: nvcc not found
**Solution**:
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: Import error "cannot import name '_custom_ops'"
**Solution**:
```bash
# Full rebuild
rm -rf build/
pip uninstall vllm -y
pip install -e . --no-build-isolation -v
```

### Issue: Nsight Systems won't launch GUI
**Solution**:
```bash
# Text-based analysis instead
nsys stats profile.nsys-rep --report cuda_gpu_trace
```

### Issue: Test models won't download
**Solution**:
```bash
# Manual download
git clone https://huggingface.co/facebook/opt-125m
export HF_HOME=~/models
```

---

## 📚 Additional Resources

- **vLLM Docs**: https://docs.vllm.ai/
- **CUDA Setup**: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- **Nsight Systems**: https://developer.nvidia.com/nsight-systems
- **VSCode C++**: https://code.visualstudio.com/docs/languages/cpp

---

**Environment setup complete! Ready to dive into vLLM internals! 🚀**
