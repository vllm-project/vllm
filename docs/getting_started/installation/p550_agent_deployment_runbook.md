# P550 Autonomous vLLM Deployment Runbook

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
SPDX-FileCopyrightText: Copyright (c) 2026 zyz
-->

This runbook is written for an AI coding agent that has cloned the
`supercatking/vllm` repository and has been asked to deploy vLLM on a SiFive
P550 board, start Qwen2.5 0.5B inference, and validate 10 successful chat
rounds.

Do not stop after environment inspection. Continue through deployment,
service startup, and 10-round validation unless a hard external artifact is
unavailable.

## Trigger

Use this runbook when the request mentions P550 deployment, P550 vLLM
validation, Qwen0.5B/Qwen2.5 0.5B on P550, or 10-round chat validation. Before
acting, also read:

- `docs/getting_started/installation/p550_sbc_validation_dependencies.md`
- `docs/getting_started/installation/p550_vllm_implementation.md`
- `docs/getting_started/installation/cpu.p550.inc.md`

## Operating Rules

- Never write board addresses, SSH usernames, passwords, private tokens, model
  cache paths, or company network details into git-tracked files.
- Do not commit model weights, wheel caches, virtual environments, logs, or
  generated local artifacts.
- Prefer deployment fixes over source changes. If source code changes become
  unavoidable, first write a short Markdown plan explaining the missing
  component and why dependency-only remediation failed.
- The P550 deployment environment intentionally uses `.venv-p550` and
  `python -m pip`; this P550 operational runbook overrides the generic
  development-environment guidance in `AGENTS.md` for board deployment tasks.
- Keep working until the service is healthy and the 10 validation requests pass.

## Required Inputs

The agent must know how to access the target board. Accept either mode:

1. Running directly on the P550 board.
2. Running from another machine with an SSH target supplied out of band.

Use environment variables or local operator input for access details. Do not
hardcode them in files.

Recommended variables:

```bash
export P550_REPO_DIR="${P550_REPO_DIR:-$HOME/vllm}"
export P550_ARTIFACT_DIR="${P550_ARTIFACT_DIR:-$PWD/p550_artifacts}"
export P550_WHEELHOUSE="${P550_WHEELHOUSE:-$P550_ARTIFACT_DIR/p550-wheelhouse}"
export P550_MODEL_TARBALL="${P550_MODEL_TARBALL:-$P550_ARTIFACT_DIR/qwen2.5-0.5b-instruct.tar.zst}"
```

If the target board can only access GitHub and a Tsinghua PyPI mirror, the
artifact directory must already be populated from a GitHub release or other
GitHub-hosted artifact source.

## Network-Restricted Deployment Strategy

If the board can only reach GitHub and the Tsinghua PyPI mirror, do not attempt
to download from Hugging Face, external PyTorch indexes, or standard PyPI.

Prepare these GitHub-hosted artifacts before deployment:

```text
p550-wheelhouse/
  torch-2.4.1-cp312-cp312-manylinux_2_35_riscv64.whl
  tokenizers-0.23.0rc0-*.whl, llguidance-*-riscv64.whl
  pyzmq/pydantic_core/pybase64/watchfiles/aiohttp/yarl/propcache/frozenlist wheels
  blake3-1.0.6 wheel and pure-Python wheels listed in the dependency report
qwen2.5-0.5b-instruct.tar.zst
```

If these artifacts are missing, the agent cannot honestly complete the
validation under the restricted network policy. Report the exact missing
artifact names and stop only at that point.

## Deployment Steps

### 1. Clone or Enter the Repository

```bash
if [ ! -d "$P550_REPO_DIR/.git" ]; then
  git clone --branch p550dev https://github.com/supercatking/vllm.git "$P550_REPO_DIR"
fi
cd "$P550_REPO_DIR"
git branch --show-current
git rev-parse --short HEAD
```

Expected branch: `p550dev`. Do not run destructive git commands.

### 2. Install System Dependencies

Use the board's allowed Ubuntu package source or an offline `.deb` bundle.

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates curl git build-essential gcc g++ \
  cmake ninja-build python3-dev python3-venv \
  libnuma-dev libtcmalloc-minimal4 cargo rustc pkg-config \
  libzmq3-dev libsodium-dev libsentencepiece-dev sentencepiece \
  libprotobuf-dev protobuf-compiler \
  python3-numpy python3-psutil python3-requests python3-uvloop
```

If graphical services are running and memory is tight, stop the display manager
temporarily for the validation window.

### 3. Create the P550 Virtual Environment

```bash
cd "$P550_REPO_DIR"
python3 -m venv --system-site-packages .venv-p550
. .venv-p550/bin/activate
python -m pip install --upgrade pip
```

The `--system-site-packages` flag is intentional. It lets the environment use
Ubuntu's working `numpy`, `psutil`, `requests`, and `uvloop` packages on
RISC-V.

### 4. Install Wheelhouse-First Python Dependencies

If a wheelhouse is present, use it first:

```bash
python -m pip install --no-index --find-links "$P550_WHEELHOUSE" \
  torch==2.4.1 tokenizers==0.23.0rc0 llguidance pyzmq pydantic-core
```

Then install the remaining baseline:

```bash
python -m pip install --find-links "$P550_WHEELHOUSE" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  cmake ninja packaging wheel jinja2 regex \
  setuptools==77.0.3 setuptools-scm "setuptools-rust>=1.9.0" \
  maturin blake3==1.0.6 \
  transformers==4.56.2 huggingface_hub==0.36.0 safetensors==0.4.3 \
  sentencepiece==0.2.0 cloudpickle==3.1.2 \
  pydantic==2.13.4 annotated-types==0.7.0 typing-inspection==0.4.2 \
  msgspec protobuf pyyaml six typing_extensions filelock \
  partial-json-parser diskcache lark==1.2.2 \
  cbor2==5.6.5 einops==0.8.2 gguf==0.19.0 ijson==3.5.0 \
  py-cpuinfo==9.0.0 pybase64==1.4.3 python-json-logger==4.1.0 \
  setproctitle==1.3.7 prometheus-client==0.25.0 \
  fsspec networkx sympy aiohttp fastapi uvicorn watchfiles openai
```

Expected pits:

- Do not let pip replace `torch==2.4.1` with an unavailable RISC-V torch build.
- Do not install `tokenizers==0.23.1` with `transformers==4.56.2`.
- Avoid `cbor2` 6.x source builds on old Cargo; use `cbor2==5.6.5`.
- Avoid forcing `safetensors>=0.6.2` unless a tested RISC-V wheel exists.

### 5. Build vLLM from Source

```bash
cd "$P550_REPO_DIR"
. .venv-p550/bin/activate
export VLLM_TARGET_DEVICE=cpu
export VLLM_RVV_VLEN=0
export MAX_JOBS=4
python -m pip install -e . --no-build-isolation --no-deps
```

Verify imports:

```bash
python - <<'PY'
import blake3
import torch
import vllm
import vllm._C as C
print("blake3", getattr(blake3, "__version__", "unknown"))
print("torch", torch.__version__)
print("vllm", getattr(vllm, "__version__", "unknown"), vllm.__file__)
print("vllm _C", C.__file__)
PY
```

The `cpuinfo` unsupported-architecture message is a known warning.

### 6. Install the Qwen2.5 0.5B Model Locally

Model files must not be committed to git.

```bash
mkdir -p "$P550_REPO_DIR/.p550_models"
tar --zstd -xf "$P550_MODEL_TARBALL" -C "$P550_REPO_DIR/.p550_models"
test -f "$P550_REPO_DIR/.p550_models/qwen2.5-0.5b-instruct/model.safetensors"
test -f "$P550_REPO_DIR/.p550_models/qwen2.5-0.5b-instruct/tokenizer.json"
```

### 7. Start the Minimal vLLM Service

```bash
cd "$P550_REPO_DIR"
. .venv-p550/bin/activate

if [ -f /tmp/p550_qwen_vllm.pid ]; then
  old_pid="$(cat /tmp/p550_qwen_vllm.pid 2>/dev/null || true)"
  [ -n "$old_pid" ] && kill "$old_pid" 2>/dev/null || true
fi

setsid env \
  VLLM_P550_HOME="$P550_REPO_DIR" \
  VLLM_P550_MODEL="$P550_REPO_DIR/.p550_models/qwen2.5-0.5b-instruct" \
  VLLM_P550_SERVED_MODEL_NAME=qwen2.5-0.5b-instruct \
  VLLM_P550_MAX_MODEL_LEN=128 \
  VLLM_P550_KV_CACHE_BYTES=536870912 \
  VLLM_TARGET_DEVICE=cpu \
  VLLM_RVV_VLEN=0 \
  OMP_NUM_THREADS=4 \
  VLLM_CPU_OMP_THREADS_BIND=nobind \
  VLLM_WORKER_MULTIPROC_METHOD=fork \
  bash tools/p550_start_vllm_service.sh \
  </dev/null >/tmp/p550_qwen_vllm.log 2>&1 &

echo $! > /tmp/p550_qwen_vllm.pid
```

Wait for health:

```bash
for i in $(seq 1 300); do
  if curl -fsS http://127.0.0.1:8000/health; then
    break
  fi
  sleep 1
done
```

If startup fails, inspect `/tmp/p550_qwen_vllm.log`. Fix missing runtime
dependencies and restart. Do not declare success until health passes.

### 8. Run 10 Chat Validation Rounds

Run this from the P550 board or from a machine that can reach the service:

```bash
python - <<'PY'
import json, time, urllib.request
base = "http://127.0.0.1:8000"
model = "qwen2.5-0.5b-instruct"
tests = [
    ("What is 1+2? Answer with only the number.", "3"),
    ("What is 2*3? Answer with only the number.", "6"),
    ("What is 10-4? Answer with only the number.", "6"),
    ("What is 7+5? Answer with only the number.", "12"),
    ("How many days are in a week? Answer with only the number.", "7"),
    ("How many sides does a square have? Answer with only the number.", "4"),
    ("What is the capital of France? Answer with one word.", "Paris"),
    ("Name one color of the sky on a clear day. Answer with one word.", "Blue"),
    ("Answer yes or no: Is ice cold?", "Yes"),
    ("What month comes after January? Answer with one word.", "February"),
]
passed = 0
for idx, (prompt, expected) in enumerate(tests, 1):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 24, "temperature": 0.0}
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        method="POST")
    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.loads(response.read().decode())
    answer = data["choices"][0]["message"]["content"].strip()
    ok = expected.lower() in answer.lower()
    passed += int(ok)
    print(f"{idx:02d} {'PASS' if ok else 'FAIL'} "
          f"{time.time() - start:.1f}s expected={expected!r} answer={answer!r}")
print(f"SUMMARY {passed}/{len(tests)}")
raise SystemExit(0 if passed == len(tests) else 1)
PY
```

Acceptance criteria:

- All 10 requests return HTTP 200.
- All 10 answers contain the expected substring, case-insensitive.
- `/health` still returns OK after the final request.
- `/tmp/p550_qwen_vllm.log` includes `CPU_ATTN`.
- The log does not contain `fla_stub` unsupported-op errors.
- The log shows Triton is not installed or not used.
