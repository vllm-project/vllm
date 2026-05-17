# Windows Compatibility Work

This fork tracks native Windows fixes needed by OmniChat's embedded vLLM experiments.

## Current Branch

Branch: `windows-compat`

## Changes

- `vllm.utils.network_utils.get_open_zmq_ipc_path()` returns a loopback TCP ZMQ endpoint on Windows, because pyzmq does not support `ipc://` transport on Windows. Unix behavior is unchanged.
- Windows CUDA source builds now auto-select `VLLM_TARGET_DEVICE=cuda` when CUDA-enabled PyTorch is present instead of forcing the empty backend.
- CMake path handling now normalizes Windows paths before passing them through Python/CMake/Torch discovery.
- Windows defaults to `spawn` for worker multiprocessing and avoids registering Windows process handles with `zmq.Poller`.
- `uvloop` call sites use `vllm.utils.uvloop_compat`, falling back to `asyncio.run` on Windows.
- CUDA backend selection treats missing `vllm-flash-attn` extensions as unavailable and falls back to FlashInfer/Triton instead of failing during model import or first attention call.
- Multimodal encoder backend selection skips FlashAttention when the varlen flash-attn op is unavailable, avoiding a Blackwell Windows `cudaErrorUnsupportedPtxVersion` path.
- Rotary embedding falls back to the native PyTorch implementation when `vllm_flash_attn.layers.rotary` is absent in the Windows build.
- MSVC/CUDA compile fixes cover core kernels, MoE top-k kernels, shared-memory alignment, `clock_gettime`, `ssize_t`, MSVC macro conflicts, and CUDA 13 / C++20 compatibility.

## Start From Scratch

These notes assume PowerShell on Windows, Visual Studio 2022 Build Tools, CUDA
13, and a CUDA-enabled PyTorch build.

1. Create the venv and install build helpers:

```powershell
py -3.12 -m venv C:\tmp\vllmvenv
C:\tmp\vllmvenv\Scripts\python.exe -m pip install -U pip setuptools wheel ninja cmake
```

2. Install PyTorch and runtime dependencies appropriate for the CUDA toolkit
   being used. The locally validated venv used:

```text
torch 2.11.0+cu130
torchvision 0.26.0+cu130
torchaudio 2.11.0+cu130
triton-windows 3.6.0.post26
flashinfer-python 0.6.8.post1
```

3. Clone and build the fork:

```powershell
cd C:\Users\ericl\Documents\ai-agents\Claude
git clone -b windows-compat https://github.com/ericleigh007/vllm-windows.git
cd vllm-windows

$env:CUDA_HOME = "C:\tmp\cuda13"
$env:CUDA_PATH = "C:\tmp\cuda13"
$env:CUDACXX = "C:\tmp\cuda13\bin\nvcc.exe"
$env:VLLM_TARGET_DEVICE = "cuda"
$env:MAX_JOBS = "4"
$env:NVCC_THREADS = "1"
$env:FETCHCONTENT_BASE_DIR = "C:\tmp\vllm_deps"
$env:CMAKE_ARGS = "-DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler"

C:\tmp\vllmvenv\Scripts\python.exe setup.py build_ext --inplace
C:\tmp\vllmvenv\Scripts\python.exe -m pip install -e .
```

Use a space-free CUDA path such as `C:\tmp\cuda13`. If CUDA is installed under
`C:\Program Files\...`, create a junction and point `CUDA_HOME`, `CUDA_PATH`,
and `CUDACXX` at the junction.

## CUDA 13 Build Notes

Validated locally with:

- CUDA Toolkit 13.0.3 installed as `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- Junction `C:\tmp\cuda13_system` pointing at the CUDA install to avoid spaces in toolchain paths
- PyTorch `2.11.0+cu130`
- NVIDIA RTX PRO 6000 Blackwell, compute capability `(12, 0)`
- Visual Studio 2022 Build Tools developer environment

Build command shape:

```powershell
set CUDA_HOME=C:\tmp\cuda13_system
set CUDA_PATH=C:\tmp\cuda13_system
set CUDACXX=C:\tmp\cuda13_system\bin\nvcc.exe
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=4
set NVCC_THREADS=1
set FETCHCONTENT_BASE_DIR=C:\tmp\vllm_deps
set CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler
C:\tmp\vllmvenv\Scripts\python.exe setup.py build_ext --inplace
```

## Current MSVC Limits

These optional acceleration paths are intentionally skipped under MSVC while the native build is brought up:

- `vllm-flash-attn` extensions
- AWQ stable GEMM kernels
- Marlin / Marlin-MOE generated kernels
- QuTLASS
- CUTLASS stable scaled-mm / MoE / FP4 / MLA kernels that currently fail under MSVC/CUDA 13

Core CUDA extensions, stable libtorch extension, MoE extension, triton-kernels packaging, CUDA memory allocator, and spinloop extension build and import.

## Validation

- `setup.py build_ext --inplace` completed successfully with CUDA 13.
- `python -m py_compile` passed for changed Python build/runtime helpers.
- Extension import smoke passed for `vllm._C`, `vllm._C_stable_libtorch`, `vllm._moe_C`, `vllm.cumem_allocator`, and `vllm.spinloop`.
- `scripts/windows_cuda_smoke.py` passed with default spawned V1 engine multiprocessing:
  - model: `facebook/opt-125m`
  - attention backend: FlashInfer
  - generated output included `I have a Windows CUDA`
- The build has also been used underneath `vllm-omni-windows` to run Qwen3-TTS
  streaming, Qwen3-Omni image/audio understanding, and Qwen3-Omni full
  audio-in/audio-out streaming on Windows.

## Runtime Evidence

This now goes beyond the earlier runtime monkeypatches: a native Windows source build can compile, import, start a spawned V1 CUDA engine, load a model, warm FlashInfer attention, and generate text without WSL or a client/server workaround.
