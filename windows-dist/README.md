# vLLM for AMD ROCm

## One Install to Rule Them All

Extract the zip, double-click `install.bat`, answer 3 questions, done.

```
Where to install? [E:\VLLM]: 
Create Python venv? (Y/N) [Y]: 
Install PyTorch with ROCm? (Y/N) [Y]: 
```

That's it. It clones vLLM, installs everything, copies `_C.pyd`.

## After Install

Double-click `vllm.exe` in your install folder, pick a model, click Start.
Open http://localhost:8001/

## What's in the Box

| File | Size | What |
|------|------|------|
| `_C.pyd` | 13.8 MB | Pre-built C++ extension (no compile needed) |
| `vllm.exe` | 17 KB | GUI launcher |
| `install.bat` | — | Interactive installer (asks 3 questions, does the rest) |
| `build-harness/` | — | Source to rebuild _C.pyd if needed |

## Requirements

- Windows 11, AMD RDNA3+ GPU, 8+ GB VRAM
- Internet (downloads ~3 GB of packages)
- ROCm 7.13 system-installed (or `pip install` from the AMD repo)
