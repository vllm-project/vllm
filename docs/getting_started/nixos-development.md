# NixOS Development Environment

This guide shows how to set up a reproducible development environment for vLLM on NixOS using Nix flakes with full CUDA support.

## Prerequisites

- NixOS system or Nix package manager installed
- NVIDIA GPU with CUDA support
- At least 16GB RAM (32GB+ recommended for building)
- ~10GB free disk space for dependencies
- For Docker usage: NVIDIA Container Toolkit properly configured

## Quick Start

1. **Enter the development shell:**
   ```bash
   cd vllm/
   nix develop
   ```

2. **First-time setup (inside the shell):**
   ```bash
   # Option A: Use the provided build script (recommended)
   ./build_vllm.sh
   
   # Option B: Manual installation
   uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
   uv pip install -e .
   ```

3. **Test the installation:**
   ```bash
   python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## What's Included

The Nix flake provides:

### Development Tools
- **Python 3.12** with pip, setuptools, wheel
- **uv** - Fast Python package manager
- **Git** with git-lfs
- **Pre-commit** hooks and **ruff** linting
- **Zed editor** - Modern code editor with AI assistance

### CUDA Toolkit (12.8)
- `nvcc` compiler
- CUDA runtime libraries (cudart, cupti, nvrtc)
- Math libraries (cublas, cufft, curand, cusolver, cusparse)
- **cuDNN** and **NCCL** for deep learning

### Build System
- **CMake** and **Ninja** build systems
- **GCC 13** compiler toolchain
- System libraries (zlib, libGL, X11)

### Environment Configuration
The shell automatically configures:
- `CUDA_HOME` and `CUDA_PATH`
- `LD_LIBRARY_PATH` with CUDA libraries
- `PATH` with CUDA binaries
- `VLLM_TARGET_DEVICE=cuda`
- Virtual environment with uv

## Development Workflow

### Code Editing
```bash
# Open project in Zed editor (included in the environment)
zed .

# Or use your preferred editor
code .  # VS Code
vim .   # Vim
```

### Building vLLM

**Recommended: Use the build script**
```bash
# Normal build with optimal settings for your system
./build_vllm.sh

# Clean build (removes all build artifacts first)
./build_vllm.sh clean

# Fast build (attempts to skip some components)
./build_vllm.sh fast

# Custom parallel jobs
MAX_JOBS=8 ./build_vllm.sh
```

**Manual building**
```bash
# Clean rebuild after code changes
uv pip install -e . --no-build-isolation

# Parallel build (adjust MAX_JOBS for your system)
MAX_JOBS=10 uv pip install -e .
```

The build script (`build_vllm.sh`) provides:
- Automatic RAM-based MAX_JOBS calculation
- RTX 3080 Ti optimized CUDA architecture (sm_86)
- Proper NVIDIA driver library paths
- Pre-flight checks for CUDA/GPU setup
- Detailed build logging
- Error handling and troubleshooting hints

### Running Tests
```bash
# Basic functionality test
python -c "from vllm import LLM; print('vLLM imported successfully')"

# Run pytest (if available)
pytest tests/
```

### Running the Server
```bash
# Start OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/DialoGPT-small \
  --port 8000

# Or use the CLI
vllm serve microsoft/DialoGPT-small --port 8000
```

## Troubleshooting

### Common Issues

#### CUDA Not Found
```bash
# Check CUDA installation
nvcc --version
echo $CUDA_HOME

# Should show: /nix/store/.../cuda-merged-12.8
```

#### Build Out of Memory
```bash
# Reduce parallel jobs
MAX_JOBS=4 uv pip install -e .
```

#### Library Not Found Errors
```bash
# Check library paths
echo $LD_LIBRARY_PATH

# Should include CUDA lib directories
```

#### glibc Version Conflicts
The flake uses nixos-25.11 to avoid glibc 2.42 + CUDA 12.8 incompatibility.
If you encounter glibc errors, ensure you're using the pinned nixpkgs version.

### Performance Tuning

#### Build Performance
- **RAM**: 32GB+ recommended for `MAX_JOBS=20`
- **Storage**: Use SSD for faster builds
- **CPU**: More cores = faster parallel builds

#### Runtime Performance
- **GPU Memory**: Check with `nvidia-smi`
- **CUDA Kernels**: First run compiles kernels (slow)
- **Model Size**: Ensure GPU has enough VRAM

## Configuration

### Customizing the Environment

Edit `flake.nix` to:
- Change Python version
- Add additional packages
- Modify CUDA version
- Adjust build settings

### Environment Variables

Key variables set by the flake:
```bash
CUDA_HOME=/nix/store/.../cuda-merged-12.8
VLLM_TARGET_DEVICE=cuda
MAX_JOBS=20
VIRTUAL_ENV_PROMPT=vllm-dev
```

## Alternative: Docker

If Nix isn't suitable for your setup, consider using the official vLLM Docker images.

**Note**: Requires NVIDIA Container Toolkit to be properly installed and configured for GPU access.

```bash
# Pull official image
docker pull vllm/vllm-openai:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model microsoft/DialoGPT-small
```

## Contributing

When contributing to the Nix flake:

1. **Test thoroughly**: Verify on clean NixOS system
2. **Pin versions**: Keep nixpkgs pinned for reproducibility
3. **Document changes**: Update this guide for new features
4. **Check compatibility**: Ensure CUDA/glibc versions work together
5. **Commit flake.lock**: Keep lockfile in version control

## Resources

- [Nix Flakes Documentation](https://nixos.wiki/wiki/Flakes)
- [CUDA on NixOS](https://nixos.wiki/wiki/CUDA)
- [vLLM Documentation](https://docs.vllm.ai)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)