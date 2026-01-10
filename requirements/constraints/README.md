# Dependency Constraints

This directory contains pinned dependency constraints for deterministic installations of vLLM.

## Overview

The constraints files in this directory provide **exact version pins** for all dependencies, ensuring deterministic builds.

**Default Behavior:**

- **Docker builds**: Constraints are **automatically applied** for reproducibility
- **`pip install vllm`**: Uses **flexible requirements** (keeps compatible existing packages)
- **Manual install**: Constraints are **opt-in** for users who want determinism

## Usage

### Installing vLLM Package

**Standard installation** (recommended for most users):

```bash
pip install vllm
```

- Uses flexible requirements from `requirements/*.txt`
- Keeps your existing compatible packages
- Gets latest compatible versions on fresh install

**Deterministic installation** (for reproducible environments):

```bash
# Clone the repository first
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install with constraints
pip install -r requirements/cuda.txt -c requirements/constraints/cuda.txt
```

- Uses exact pinned versions from constraints
- Guarantees same versions across installs
- May upgrade/downgrade existing packages to match pins

### Docker Builds

**All official Dockerfiles automatically use constraints** for deterministic builds:

- `docker/Dockerfile` (CUDA)
- `docker/Dockerfile.rocm` (ROCm)
- `docker/Dockerfile.tpu` (TPU)
- `docker/Dockerfile.xpu` (Intel XPU)

No changes needed - constraints are built-in.

### How It Works

- **requirements/*.txt** files contain flexible version constraints (e.g., `transformers >= 4.56.0, < 5`)
- **requirements/constraints/*.txt** files contain exact pins (e.g., `transformers==4.57.3`)
- **`pip install vllm`** reads from `requirements/*.txt` (flexible)
- **Docker builds** use both `-r requirements/*.txt -c requirements/constraints/*.txt` (deterministic)
- **Manual installs** can opt-in to constraints for reproducibility

## Special Cases

### CPU-only and dev Requirements

The following files **cannot** be compiled to standard constraints due to platform-specific torch builds:

- `cpu.txt` - Uses `torch==2.9.1+cpu` (CPU-only build)
- `cpu-build.txt` - Uses `torch==2.9.1+cpu` (CPU-only build)
- `dev.txt` - Includes `test.txt` which references CUDA builds

For these files, the requirements file itself already specifies exact versions, so use them directly:

```bash
# Use the requirements file alone for cpu/dev
pip install -r requirements/cpu.txt
pip install -r requirements/dev.txt
```

## Updating Dependencies

### Adding a New Dependency

1. Add it to the appropriate `requirements/*.txt` file with flexible constraints:

   ```requirements
   new-package >= 1.2.0
   ```

2. Regenerate the constraints:

   ```bash
   cd requirements
   ./compile_constraints.sh
   ```

3. Commit both the updated requirements file and the regenerated constraints file

### Updating a Dependency Version

1. Update the constraint in `requirements/<name>.txt`:

   ```diff
   - transformers >= 4.56.0, < 5
   + transformers >= 4.58.0, < 5
   ```

2. Regenerate constraints:

   ```bash
   cd requirements
   ./compile_constraints.sh
   ```

3. Review the updated pins in the constraints file

### Refreshing All Pins to Latest Compatible

To update all pins to the latest versions that satisfy the requirements:

```bash
cd requirements
./compile_constraints.sh
```

This will resolve all dependencies to their latest compatible versions available on PyPI.

## File Structure

```text
requirements/
├── common.txt              # Base dependencies (flexible)
├── cuda.txt                # CUDA-specific (flexible)
├── ...
├── constraints/
│   ├── common.txt          # Base dependencies (exact pins)
│   ├── cuda.txt            # CUDA-specific (exact pins)
│   ├── ...
│   └── README.md           # This file
└── compile_constraints.sh  # Script to regenerate constraints
```

## Requirements Files Coverage

Constraint files are available for:

- ✅ `common.txt` - Base dependencies
- ✅ `cuda.txt` - CUDA GPU support
- ✅ `rocm.txt` - AMD ROCm GPU support
- ✅ `tpu.txt` - Google TPU support
- ✅ `xpu.txt` - Intel XPU support
- ✅ `build.txt` - Build dependencies
- ✅ `docs.txt` - Documentation dependencies
- ✅ `lint.txt` - Linting/formatting tools
- ✅ `test.txt` - Testing dependencies
- ✅ `kv_connectors.txt` - KV cache connectors
- ✅ `nightly_torch_test.txt` - Nightly PyTorch tests
- ✅ `rocm-build.txt` - ROCm build dependencies
- ✅ `rocm-test.txt` - ROCm testing dependencies

Files without constraints (use requirements directly):

- ⚠️ `cpu.txt` - Platform-specific torch builds
- ⚠️ `cpu-build.txt` - Platform-specific torch builds
- ⚠️ `dev.txt` - References test.txt with special builds

## CI/CD Integration

### Option 1: Use Docker (Recommended)

Docker builds automatically use constraints:

```yaml
# .github/workflows/ci.yml
- name: Build and test
  run: docker build -f docker/Dockerfile .
```

### Option 2: Direct pip install with constraints

For deterministic non-Docker CI:

```yaml
# .github/workflows/ci.yml
- name: Install dependencies
  run: pip install -r requirements/cuda.txt -c requirements/constraints/cuda.txt
```

## Technical Details

- Constraints are generated using [uv](https://github.com/astral-sh/uv) pip compile
- Compilation targets Python 3.12
- Platform-specific compilations for CUDA (x86_64-manylinux_2_28)
- Uses `--index-strategy unsafe-best-match` for ROCm and XPU custom indexes
- Git-based dependencies are preserved with exact commit hashes

## Benefits

1. **Reproducible builds** - Same dependencies every time
2. **Dependency conflict prevention** - All transitive dependencies resolved together
3. **Security** - Pin known-good versions, review changes explicitly
4. **Performance** - Skip unnecessary reinstalls when packages already satisfy requirements
5. **Flexibility** - Can still use latest versions by omitting `-c` flag
