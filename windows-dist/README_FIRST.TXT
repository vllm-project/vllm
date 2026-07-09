================================================================
  vLLM for AMD ROCm - READ THIS FIRST
================================================================

BEFORE YOU RUN install.bat:

1. CHECK ROCm IS INSTALLED
   Open a command prompt and run:
       hipcc --version
   If you see "HIP version: 7.13" (or similar), you're good.
   If not, install ROCm for Windows from:
       https://rocm.docs.amd.com

2. CHECK PYTHON 3.12 IS INSTALLED
   Run:
       python --version
   If you see "Python 3.12.x", you're good.
   If not, download from:
       https://www.python.org/downloads/release/python-3129/

3. YOU HAVE AN AMD GPU
   RDNA3 (RX 7000 series) or RDNA4 (RX 9000 series)
   At least 8 GB VRAM (16 GB recommended)

4. INTERNET CONNECTION
   The installer downloads ~3 GB of packages.

5. ENOUGH DISK SPACE
   At least 10 GB free on the install drive.

WHAT install.bat WILL DO:
- Ask where to install
- Ask if you want a Python virtual environment
- Ask if you want to install PyTorch with ROCm
- Clone vLLM Windows port from GitHub
- Install vLLM and copy the pre-built _C.pyd binary

AFTER INSTALL:
- Run vllm.exe to start the GUI server
- Or run: python -m vllm.entrypoints.openai.api_server --model <path> --enforce-eager
- Open http://localhost:8001/ in your browser

================================================================
