# Persistent top-k overflow reproducer

This reproducer compares the installed vLLM v0.27.0 persistent top-k operator
with the implementation in this PR. It runs an 87-case matrix and compares
selected values exactly with `torch.topk`.

The PR source files used by the isolated extension are:

- `csrc/libtorch_stable/persistent_topk.cuh`
- `csrc/libtorch_stable/topk_histogram_4096.cuh`

## Docker reproduction

Build from the repository root. The Dockerfile copies both PR headers into
`/repro`, so the commands below work unchanged.

```bash
docker build \
  -f tests/kernels/persistent_topk_reproducer/Dockerfile \
  -t pr52149-overflow-repro .
```

Run both backends:

```bash
docker run --rm \
  --gpus 'device=0' \
  --shm-size=16g \
  --ulimit memlock=-1:-1 \
  pr52149-overflow-repro 'bash ./run.sh' \
  2>&1 | tee persistent_topk_overflow_repro.log
```

`run.sh` executes these exact commands:

```bash
# Installed v0.27.0 baseline.
python3 repro_persistent_topk.py \
  --backend persistent --full --repeats 20

# PR headers compiled into the isolated extension.
TORCH_CUDA_ARCH_LIST=10.3 MAX_JOBS=4 \
PATCHED_TOPK_SOURCE="$PWD/patched_persistent_topk_ext.cu" \
PATCHED_TOPK_INCLUDE="$PWD" \
python3 repro_persistent_topk.py \
  --backend overflow-extension --full --repeats 20
```
Expected result in the validated NVIDIA B300 environment:

- installed v0.27.0 reproduces the long-vector overflow failures;
- the PR extension reports `0/87` failing cases, no invalid or duplicate
  indices, and exact selected-value agreement with `torch.topk`.

## Run from an existing vLLM v0.27.0 environment

From this directory, point the include path at the PR checkout:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"

python3 repro_persistent_topk.py \
  --backend persistent --full --repeats 20

TORCH_CUDA_ARCH_LIST=10.3 MAX_JOBS=4 \
PATCHED_TOPK_SOURCE="$PWD/patched_persistent_topk_ext.cu" \
PATCHED_TOPK_INCLUDE="$REPO_ROOT/csrc/libtorch_stable" \
python3 repro_persistent_topk.py \
  --backend overflow-extension --full --repeats 20
```
