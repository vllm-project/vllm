# P550 SBC Validation Dependency Report

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
SPDX-FileCopyrightText: Copyright (c) 2026 zyz
-->

This document records the dependency work discovered while validating the
`p550dev` branch on a second SiFive P550 SBC board. The goal of this validation
was to prove that the branch can be deployed from a fresh clone and can run
Qwen2.5 0.5B Instruct through the vLLM CPU backend.

Board addresses, SSH credentials, model cache paths, and local wheel cache paths
are intentionally omitted from this document.

## Validation Result

The SBC validation completed successfully:

- Source checkout: `p550dev`
- Validated commit: `f35915f86`
- Target backend: vLLM CPU backend
- Attention path: `CPU_ATTN`
- Triton: not installed and not used
- Model: Qwen2.5 0.5B Instruct, loaded from a local model directory
- Runtime baseline: scalar RISC-V CPU path with `VLLM_RVV_VLEN=0`
- CPU usage target: four P550 cores through `OMP_NUM_THREADS=4`
- Final chat validation: 10 short chat requests, 10 passed

Observed short-request latency was about 19 to 21 seconds per request. The
service remained healthy after the final request.

## Why Extra Downloads Were Needed

The vLLM repository contains the source code needed for the P550 CPU backend,
but it does not vendor Python runtime packages, native Python extension wheels,
system libraries, PyTorch, or model weights.

On the SBC board, the normal upstream vLLM dependency set was also not directly
usable because:

- RISC-V wheels are not available for every package on standard PyPI.
- Some packages build from source and require Rust/C toolchains.
- The tested board's default Rust toolchain is older than some modern Python
  packages expect.
- The board has limited or unreliable access to some external sites, so large
  artifacts are better downloaded from Windows or WSL and uploaded to the board.
- Upstream vLLM currently declares a newer RISC-V PyTorch requirement than the
  board-compatible PyTorch build used for this validation.

## System Packages Downloaded

The following packages were installed from the Ubuntu package repositories.
They are not part of the vLLM source tree.

### Base Build and Runtime Packages

```text
ca-certificates
curl
git
build-essential
gcc
g++
cmake
ninja-build
python3-dev
python3-venv
libnuma-dev
libtcmalloc-minimal4
```

### Native Extension Build Support

```text
cargo
rustc
pkg-config
```

These were needed for packages such as `blake3`, `pydantic-core`, and other
native Python extensions when a prebuilt RISC-V wheel was unavailable.

### Native Library Dependencies

```text
libzmq3-dev
libsodium-dev
libsentencepiece-dev
sentencepiece
libprotobuf-dev
protobuf-compiler
```

These were needed to make packages such as `pyzmq`, `sentencepiece`, and
protobuf-related imports installable or usable on RISC-V.

### System Python Packages Used Through the Virtual Environment

The P550 virtual environment was configured with access to system site packages.
The following Ubuntu Python packages were used to avoid unnecessary source
builds:

```text
python3-numpy
python3-psutil
python3-requests
python3-uvloop
```

`python3-uvloop` was useful because installing `uvloop` from source on the SBC
was slow and unreliable.

## Python Packages Downloaded

The validation environment used a P550-specific virtual environment and then
installed the following extra Python packages. These packages are runtime or
build dependencies; they are not shipped in the vLLM repository.

### Build Helpers

```text
cmake
ninja
packaging
wheel
jinja2
regex
setuptools==77.0.3
setuptools-scm
setuptools-rust>=1.9.0
maturin
blake3==1.0.6
```

`blake3==1.0.6` was used because newer `blake3` source releases can require a
newer Rust/Cargo lockfile format than the tested SBC toolchain provides.

### Core Model and Tokenizer Stack

```text
torch==2.4.1
transformers==4.56.2
tokenizers==0.23.0rc0
huggingface_hub==0.36.0
safetensors==0.4.3
sentencepiece==0.2.0
```

Important notes:

- `torch==2.4.1` was used as the board-compatible RISC-V PyTorch build.
  Standard upstream vLLM dependency metadata expects a newer PyTorch version on
  RISC-V, so `pip check` reports this as incompatible even though the validated
  Qwen path works.
- `tokenizers==0.23.1` failed with `transformers==4.56.2` because Transformers
  requires `tokenizers>=0.22.0,<=0.23.0`. The available working RISC-V option was
  `tokenizers==0.23.0rc0`.
- `safetensors>=0.6.2` is declared by upstream vLLM, but the tested environment
  used `safetensors==0.4.3` because the newer version did not have a suitable
  wheel and source build was not practical. The local Qwen2.5 safetensors model
  loaded successfully with `0.4.3`.
- `sentencepiece==0.2.1` did not work against the SBC's available system
  SentencePiece library, while `sentencepiece==0.2.0` installed successfully.
  Qwen2.5 uses the fast tokenizer path from `tokenizer.json`, so SentencePiece
  was not on the critical path for this model.

### vLLM Import and Runtime Dependencies

These were installed after import or startup failures exposed missing modules:

```text
cloudpickle==3.1.2
pydantic==2.13.4
pydantic-core==2.46.4
annotated-types==0.7.0
typing-inspection==0.4.2
msgspec
protobuf
pyyaml
six
typing_extensions
filelock
partial-json-parser
diskcache
lark==1.2.2
pyzmq==27.1.0
```

Observed hard failures and fixes:

| Missing or incompatible item | Symptom | Resolution |
| --- | --- | --- |
| `tokenizers==0.23.1` | `transformers` import rejected the version | Install `tokenizers==0.23.0rc0` |
| `cloudpickle` | `from vllm import LLM` failed | Install `cloudpickle` |
| `pydantic_core` | `pydantic` import failed | Install `pydantic-core==2.46.4` plus `annotated-types` and `typing-inspection` |
| `cbor2` | vLLM hashing import failed | Install `cbor2==5.6.5` |
| `uvloop` | vLLM V1 utility import failed | Use Ubuntu `python3-uvloop` or a working RISC-V install |
| `openai` | vLLM chat utility import failed | Install `openai>=2.0.0` |
| `llguidance` | structured-output type import failed during vLLM V1 import | Install `llguidance` RISC-V wheel |

### Additional Lightweight vLLM Runtime Packages

The following packages were installed to keep the minimal Qwen text path stable
after `pip check` and vLLM import-chain inspection:

```text
cbor2==5.6.5
einops==0.8.2
gguf==0.19.0
ijson==3.5.0
py-cpuinfo==9.0.0
pybase64==1.4.3
python-json-logger==4.1.0
setproctitle==1.3.7
prometheus-client==0.25.0
fsspec
networkx
sympy
aiohttp
fastapi
uvicorn
watchfiles
openai>=2.0.0
llguidance
```

`fastapi` and `uvicorn` are not used by the P550 minimal HTTP wrapper, which
uses Python's `http.server`. They were installed because vLLM imports some
entrypoint and utility modules eagerly.

## Packages Not Required for This Minimal Validation

The following upstream vLLM dependencies were not needed for the successful
Qwen2.5 0.5B text-only validation:

```text
anthropic
compressed-tensors
depyf
lm-format-enforcer
mcp
mistral_common
model-hosting-container-standards
numba
openai-harmony
opencv-python-headless
opentelemetry-api
opentelemetry-exporter-otlp
opentelemetry-sdk
opentelemetry-semantic-conventions-ai
outlines_core
prometheus-fastapi-instrumentator
tiktoken
xgrammar
triton
triton_kernels
ray
```

These may be required for other vLLM modes, such as the full OpenAI API server,
structured output, multimodal models, telemetry, quantization, distributed
serving, or GPU paths. They were not part of the SBC acceptance criteria.

## Large Artifacts Downloaded Outside the Board

Because the SBC may not have proxy access to all external sites, large artifacts
were downloaded from Windows or WSL and then uploaded to the board.

### PyTorch Wheel

The board-compatible PyTorch wheel was downloaded off-board and uploaded into a
local wheel cache before installation:

```text
torch-2.4.1-cp312-cp312-manylinux_2_35_riscv64.whl
```

### Qwen2.5 0.5B Instruct Model

The Qwen2.5 model snapshot was also prepared off-board and uploaded to the SBC.
The local model directory contained:

```text
config.json
generation_config.json
merges.txt
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
```

Model files are not committed to the repository.

## Runtime Settings Used for the Passing Validation

The passing service used these settings:

```bash
VLLM_TARGET_DEVICE=cpu
VLLM_RVV_VLEN=0
OMP_NUM_THREADS=4
VLLM_CPU_OMP_THREADS_BIND=nobind
VLLM_WORKER_MULTIPROC_METHOD=fork
VLLM_P550_MAX_MODEL_LEN=128
VLLM_P550_KV_CACHE_BYTES=536870912
```

The SBC has four CPU cores, so `OMP_NUM_THREADS=4` is the intended setting for
this validation. The graphical system was temporarily stopped during testing to
free memory; this is an operational optimization and not a code dependency.

## Reproduction Guidance

For another fresh P550 SBC, the recommended dependency strategy is:

1. Install the system packages listed above.
2. Create the `.venv-p550` virtual environment with system site packages
   enabled.
3. Install the board-compatible PyTorch wheel first.
4. Install the Python dependency baseline and then install vLLM with:

   ```bash
   VLLM_TARGET_DEVICE=cpu VLLM_RVV_VLEN=0 \
       python -m pip install -e . --no-build-isolation --no-deps
   ```

5. Upload the Qwen2.5 0.5B Instruct snapshot into a local model directory.
6. Start `tools/p550_start_vllm_service.sh` with the runtime settings above.
7. Validate with 10 short chat requests through `/v1/chat/completions`.

## Restricted Network Deployment Plan

If a freshly imaged P550 board is only allowed to access GitHub and a Tsinghua
PyPI mirror, do not rely on the board to resolve all dependencies online. The
validated path needs several artifacts that are not reliably available from a
standard PyPI mirror on `riscv64`.

The recommended approach is to publish or otherwise stage a P550 artifact bundle
that the board can fetch from GitHub:

```text
p550-wheelhouse/
  torch-2.4.1-cp312-cp312-manylinux_2_35_riscv64.whl
  tokenizers-0.23.0rc0-*.whl
  llguidance-*-riscv64.whl
  pyzmq-*-riscv64.whl
  pydantic_core-*-riscv64.whl
  pybase64-*-riscv64.whl
  watchfiles-*-riscv64.whl
  aiohttp-*-riscv64.whl
  yarl-*-riscv64.whl
  propcache-*-riscv64.whl
  frozenlist-*-riscv64.whl
  blake3-1.0.6-*.whl
  other pure-Python wheels from the pinned list

models/
  qwen2.5-0.5b-instruct.tar.zst
```

Then install with a wheelhouse-first policy:

```bash
python -m pip install --no-index --find-links ./p550-wheelhouse \
    torch==2.4.1 tokenizers==0.23.0rc0 llguidance pyzmq pydantic-core

python -m pip install --find-links ./p550-wheelhouse \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers==4.56.2 huggingface_hub==0.36.0 \
    cloudpickle cbor2==5.6.5 einops gguf ijson py-cpuinfo pybase64 \
    python-json-logger setproctitle prometheus-client fsspec networkx sympy \
    aiohttp fastapi uvicorn watchfiles openai
```

Build and install vLLM from the GitHub checkout without dependency resolution:

```bash
VLLM_TARGET_DEVICE=cpu VLLM_RVV_VLEN=0 \
    python -m pip install -e . --no-build-isolation --no-deps
```

### Expected Pitfalls

- **PyTorch is the first hard blocker.** The Tsinghua PyPI mirror is unlikely to
  provide the validated `riscv64` PyTorch wheel. Stage the PyTorch wheel in the
  GitHub-hosted wheelhouse.
- **Model files are not pip packages.** Qwen2.5 0.5B must be staged separately,
  for example as a compressed tarball attached to a GitHub release. Do not
  commit model weights to the repository.
- **System packages may still be required.** If Ubuntu package mirrors are also
  blocked, create an offline `.deb` bundle for the system packages listed above.
  Python wheels alone are not enough for `libnuma`, `tcmalloc`, ZeroMQ,
  SentencePiece, protobuf, compiler, and Python header dependencies.
- **Rust package versions can fail on the board.** Some current packages require
  newer Cargo lockfile formats or Rust 2024 edition support. Prefer known-good
  wheels and pinned versions such as `blake3==1.0.6` and `cbor2==5.6.5`.
- **`tokenizers` version selection matters.** `transformers==4.56.2` rejected
  `tokenizers==0.23.1`; use `tokenizers==0.23.0rc0` or adjust both packages
  together after testing on RISC-V.
- **`safetensors>=0.6.2` may not be practical.** The validation used
  `safetensors==0.4.3` successfully for the local Qwen2.5 safetensors file.
  Treat newer safetensors as a separate RISC-V wheel-building task.
- **Full vLLM server dependencies are broader than the minimal wrapper.** The
  P550 acceptance path uses `tools/p550_start_vllm_service.sh`, not the full
  upstream OpenAI API server. Installing every upstream optional dependency can
  introduce avoidable RISC-V build failures.
- **Use `--no-deps` for vLLM editable install.** Letting pip resolve upstream
  vLLM dependencies on RISC-V can replace validated board-compatible packages
  with versions that do not build or do not exist for the architecture.
- **Use four CPU cores deliberately.** Set `OMP_NUM_THREADS=4`; also pass
  four-way build parallelism where possible when compiling local extensions.
- **Memory pressure can affect startup.** Stopping the graphical system during
  validation can free enough memory to make model loading more reliable.

