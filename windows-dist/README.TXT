================================================================
  vLLM for AMD ROCm - HOW TO INSTALL
================================================================

STEP 1: Install Python 3.12
  Download from: https://www.python.org/downloads/release/python-3129/
  Run the installer. Check "Add Python to PATH".

STEP 2: Install ROCm 7.13
  Download from: https://rocm.docs.amd.com
  Or use: https://repo.amd.com/rocm/whl/gfx120X-all/ (browse in browser)

STEP 3: Extract this zip
  Right-click vllm-windows-rocm-dist.zip -> Extract All
  Open the extracted folder.

STEP 4: Run the installer
  Double-click install.bat

  It will ask:
  - Where to install (press Enter for default E:\VLLM)
  - Create Python venv? (press Enter for Yes)
  - Install PyTorch? (press Enter for Yes)

STEP 5: Download the wheel files
  The installer will show you download URLs like:
    https://repo.radeon.com/rocm/windows/.../torch-....whl
    https://repo.radeon.com/rocm/windows/.../torchvision-....whl
    https://repo.radeon.com/rocm/windows/.../rocm_sdk_core-....whl
    https://repo.radeon.com/rocm/windows/.../rocm_sdk_devel-....whl

  Open each URL in your browser. Right-click -> Save As.
  Save all .whl files into one folder (e.g. C:\wheels).

STEP 6: Install the wheels
  Open a command prompt in the install folder:
    .venv\Scripts\activate
    pip install --no-index --find-links C:\wheels torch torchvision rocm-sdk-core rocm-sdk-devel

STEP 7: Finish
  Go back to the installer window, type DONE and press Enter.
  The installer will clone vLLM and set everything up.

STEP 8: Run
  Open the install folder, double-click vllm.exe
  Click Browse to select your model folder
  Click Start Server
  Open http://localhost:8001/ in your browser

================================================================
  NEED A MODEL?
================================================================

  Download Qwen2.5-3B-Instruct (works great on 16 GB VRAM):
    1. Open Command Prompt in your install folder
    2. .venv\Scripts\activate
    3. pip install huggingface_hub
    4. python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct', local_dir=r'F:\VLLM-Models\Qwen2.5-3B-Instruct')"
    5. In vllm.exe, Browse to F:\VLLM-Models\Qwen2.5-3B-Instruct

================================================================
