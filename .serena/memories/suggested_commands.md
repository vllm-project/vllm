# Suggested Commands for vLLM Development

## Installation & Setup

### Clone Repository
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

### Install vLLM

#### Python-only development (uses precompiled binaries)
```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

#### Full development (compile C++/CUDA kernels)
```bash
uv pip install -e .
```

### Install Development Dependencies
```bash
# Lint dependencies
uv pip install -r requirements/lint.txt

# Test dependencies (CUDA only)
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# Test dependencies (hardware agnostic)
uv pip install pytest pytest-asyncio

# Documentation dependencies
uv pip install -r requirements/docs.txt
```

## Linting and Formatting

### Setup Pre-commit Hooks
```bash
uv pip install pre-commit
pre-commit install
```

### Run Pre-commit Manually
```bash
pre-commit run              # runs on staged files
pre-commit run -a           # runs on all files
pre-commit run --hook-stage manual markdownlint  # run CI-only hooks
pre-commit run --hook-stage manual mypy-3.10     # run mypy for Python 3.10
```

### Bypass Pre-commit (use sparingly)
```bash
git commit --no-verify                    # bypass all hooks
SKIP=<hook-id> git commit                 # skip specific hook
```

## Testing

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific test file with verbose output
pytest -s -v tests/test_logger.py

# Run tests in a specific directory
pytest tests/kernels/

# Run with specific markers
pytest -m slow_test tests/
pytest -m core_model tests/
```

## Documentation

### Serve Documentation Locally
```bash
mkdocs serve                           # with API ref (~10 minutes)
API_AUTONAV_EXCLUDE=vllm mkdocs serve  # without API ref (~15 seconds)
```

Access at: http://127.0.0.1:8000/

## Running vLLM

### CLI Entry Point
```bash
vllm                                   # main CLI entry point
```

### Python API Examples
```bash
python examples/<example_script>.py
```

## Git Workflow

### Commit with Sign-off (DCO requirement)
```bash
git commit -s -m "Your commit message"
```

### Check Git Status
```bash
git status
git diff
git log --oneline -n 10
```

## macOS (Darwin) Specific Notes

On macOS:
- VLLM_TARGET_DEVICE automatically set to `cpu` (no GPU support)
- Standard Unix commands work (ls, cd, grep, find, etc.)
- Use Homebrew for installing dependencies if needed
```bash
brew install cmake ninja
```

## Build & Compilation

### Incremental Compilation (for kernel development)
See docs/contributing/incremental_build.md for optimized workflow when iterating on C++/CUDA kernels.

### Clean Build
```bash
rm -rf build/
uv pip install -e . --force-reinstall --no-deps
```

## Useful Development Commands

### Check Python Version
```bash
python --version
```

### List Installed Packages
```bash
uv pip list
```

### Find Files
```bash
find . -name "*.py" -type f
```

### Search Code
```bash
grep -r "pattern" vllm/
```
