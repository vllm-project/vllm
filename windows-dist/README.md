# vLLM for AMD ROCm — One Click Install

Extract the zip, double-click `install.bat`, walk away.

It downloads Python, Git, PyTorch with ROCm, clones vLLM, installs everything. When done, run `vllm.exe` to start the server.

## What You Get

| File | Purpose |
|------|---------|
| `install.bat` | **Double-click this.** Installs everything automatically. |
| `vllm.exe` | GUI launcher — pick a model, click Start. |
| `_C.pyd` | Pre-built C++ extension (13.8 MB, skips 80s compile) |
| `build-harness/` | Source to rebuild _C.pyd yourself |

## Requirements

- **Windows 11**, **AMD RDNA3+ GPU** (RX 7000/9000), **8 GB+ VRAM**
- **Internet connection** (downloads ~3 GB of packages)

## What `install.bat` Does

1. Installs **Python 3.12** (if missing) — from python.org
2. Installs **Git** (if missing) — from git-scm.com
3. Creates a Python virtual environment
4. Installs **PyTorch + ROCm** from repo.amd.com/rocm/whl/gfx120X-all/
5. Clones **vLLM Windows port** from GitHub
6. Runs `pip install -e .` to set up vLLM
7. Copies `_C.pyd` into place

## How to Run

After install.bat finishes:

1. **Double-click `vllm.exe`** in this folder
2. Click **Browse**, select your model folder
3. Click **Start Server**
4. Open **http://localhost:8001/** in your browser

**Need a model?** Download one:
```
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct', local_dir=r'F:\VLLM-Models\Qwen2.5-3B-Instruct')"
```
Then select `F:\VLLM-Models\Qwen2.5-3B-Instruct` in the GUI.

## Also in This Zip

| File | Purpose |
|------|---------|
| `setup.bat` | Lightweight installer (assumes Python + Git already installed) |
| `install.ps1` | PowerShell version |
| `run.bat` | Quick launcher from terminal |
| `launcher.cs` | Source code for vllm.exe (compile with `csc launcher.cs`) |
| `build-harness/` | Source to rebuild _C.pyd if needed |

## License

Apache 2.0
