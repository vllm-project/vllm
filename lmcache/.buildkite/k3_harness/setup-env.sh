#!/usr/bin/env bash
# Per-job environment setup: installs vLLM nightly + LMCache from source.
# Called at the start of every CI job.
set -euo pipefail

# Print the failing command and line number on any error.
trap 'echo "ERROR: setup-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# ── GPU health pre-check ────────────────────────────────────
# Fail fast if GPUs are occupied by stale host processes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80

# Resolve which vLLM nightly to install. Sets PINNED_VLLM_VERSION (empty
# string means "use latest nightly", any other value means
# "install vllm==<that exact version>"). See script header for the full
# resolution order and override knobs.
source "${REPO_ROOT}/.buildkite/k3_harness/resolve-pinned-vllm.sh"

echo "--- :broom: Pre-install bytecode/cache eviction"
# The CI base image pre-installs packages from requirements/*.txt at image
# build time. We've observed k3 jobs (integration + correctness) fail with
#   ImportError: cannot import name 'GenerationConfig' from 'transformers'
# after `uv pip install -U vllm ...` upgrades transformers, even though the
# installed .py files clearly expose GenerationConfig (verified both 5.5.0
# and 5.5.4). The same install recipe replayed in a fresh venv outside CI
# always succeeds, so the failure is tied to base-image filesystem state
# (stale __pycache__, partial upgrades via overlayfs, etc.). Evict all
# bytecode caches and uv's download cache up front so later steps operate
# on clean ground; this is cheap (few seconds) and idempotent.
find /opt/venv/lib/python3.12/site-packages -type d -name __pycache__ \
    -exec rm -rf {} + 2>/dev/null || true
uv cache clean 2>/dev/null || true

echo "--- :python: Installing vLLM nightly (pinned to cu130 index)"
# The base image is nvidia/cuda:13.0.2-devel-ubuntu24.04 (system nvcc 13).
# vLLM's generic nightly index (wheels.vllm.ai/nightly/vllm/) non-deterministically
# resolves to either a cu128 or a cu130 torch wheel depending on which wheel
# vLLM's nightly CI happened to publish that day. When the resolver picks a
# cu128 torch, torch.utils.cpp_extension._check_cuda_version aborts the
# LMCache editable install with:
#   RuntimeError: The detected CUDA version (13.0) mismatches the version
#   that was used to compile PyTorch (12.8).
#
# Pin to the cu130 sub-index so torch.version.cuda is always "13.0" and
# matches the base image. This also lets us drop the HTML-scraping + apt
# cuda-compiler alignment dance that lived here before.
# (See https://docs.vllm.ai/ install tips → Nightly → CUDA 13.0.)
#
# --reinstall-package for transformers / tokenizers / huggingface-hub /
# safetensors / vllm forces uv to uninstall-and-reinstall those packages
# even when it thinks the existing install is up to date. That is the
# minimum set to put the full `vllm serve` import chain on a freshly
# extracted wheel, which bypasses whatever filesystem-level mismatch in
# the base image was causing the GenerationConfig ImportError.
#
# When PINNED_VLLM_VERSION is non-empty (resolved by resolve-pinned-vllm.sh
# from the `buildkite_latest_tested_vllm` branch), pin to that exact wheel
# so every CI job matches the version most recently verified by the
# canary build. Empty falls back to "latest nightly".
#
# vLLM's nightly index (wheels.vllm.ai/nightly/<cuda>/) only keeps the
# *latest* wheel; older versions get rolled off within a day or two and
# pinning them there fails with "no version of vllm==X". Historical
# wheels are still served at wheels.vllm.ai/<full-commit-sha>/<cuda>/,
# which is a PEP 503 simple index. The canary records that archive URL
# directly in the pin file, so the common path needs zero extra API
# calls. As a fallback (e.g. an old-format pin file written before the
# canary started recording metadata, or a manual override via
# PINNED_VLLM_VERSION), we still expand the short SHA via the public
# GitHub commits API; GITHUB_TOKEN is honoured (5000 req/h) but optional.
PINNED_VLLM_INDEX_ARGS=()
if [[ -n "${PINNED_VLLM_VERSION:-}" ]]; then
    VLLM_INSTALL_SPEC="vllm[runai,tensorizer,flashinfer]==${PINNED_VLLM_VERSION}"
    echo "Installing vLLM pinned: ${VLLM_INSTALL_SPEC}"
    archive_url="${PINNED_VLLM_ARCHIVE_INDEX_URL:-}"
    if [[ -z "${archive_url}" ]]; then
        # Old-format pin file or per-build override: resolve on demand.
        short_sha="${PINNED_VLLM_VERSION##*+g}"
        if [[ "${short_sha}" != "${PINNED_VLLM_VERSION}" \
                && "${short_sha}" =~ ^[0-9a-f]+$ ]]; then
            gh_auth_args=()
            if [[ -n "${GITHUB_TOKEN:-}" ]]; then
                gh_auth_args=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
            fi
            full_sha=""
            for attempt in 1 2 3; do
                full_sha="$(curl -fsSL --connect-timeout 5 --max-time 10 \
                    -H "Accept: application/vnd.github+json" \
                    "${gh_auth_args[@]+"${gh_auth_args[@]}"}" \
                    "https://api.github.com/repos/vllm-project/vllm/commits/${short_sha}" \
                    2>/dev/null \
                    | awk -F'"' '/"sha":/ {print $4; exit}')" || true
                if [[ "${full_sha}" =~ ^[0-9a-f]{40}$ ]]; then
                    break
                fi
                echo "[INFO] GitHub commit lookup attempt ${attempt} for" \
                     "${short_sha} returned no SHA; retrying..." >&2
                sleep 2
            done
            if [[ "${full_sha}" =~ ^[0-9a-f]{40}$ ]]; then
                archive_url="https://wheels.vllm.ai/${full_sha}/cu130"
            else
                echo "[WARN] could not resolve full SHA for ${short_sha}; pip" \
                     "may fail if the wheel has rolled off the nightly index" >&2
            fi
        fi
    fi
    if [[ -n "${archive_url}" ]]; then
        echo "Adding commit-archived index: ${archive_url}"
        PINNED_VLLM_INDEX_ARGS+=(--extra-index-url "${archive_url}")
    fi
else
    # ">=0.0.0.dev0" explicitly limits vLLM to be the only pre-release
    # package.
    VLLM_INSTALL_SPEC="vllm[runai,tensorizer,flashinfer]>=0.0.0.dev0"
    echo "Installing latest vLLM nightly (no pin)"
fi
uv pip install -U "${VLLM_INSTALL_SPEC}" \
    --reinstall-package transformers \
    --reinstall-package tokenizers \
    --reinstall-package huggingface-hub \
    --reinstall-package safetensors \
    --reinstall-package vllm \
    "${PINNED_VLLM_INDEX_ARGS[@]+"${PINNED_VLLM_INDEX_ARGS[@]}"}" \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match

# Pre-import transformers on the main thread via sitecustomize.py so
# vllm's BG-thread preload can't race ahead of _LazyModule init.
cat > /opt/venv/lib/python3.12/site-packages/sitecustomize.py <<'PY'
try:
    import transformers  # noqa: F401
except Exception:
    pass
PY

# Probe the vLLM CLI by invoking `vllm --help` as a subprocess. This is the
# only probe that exercises the full import chain that `vllm serve` runs:
# vllm.entrypoints.cli.main.main() triggers `import vllm.entrypoints.cli.
# benchmark.main` *inside* the function body, which in turn loads
# vllm.config.model -> vllm.transformers_utils.config -> `from transformers
# import GenerationConfig, PretrainedConfig`. A plain `from vllm.entrypoints.
# cli.main import main` only resolves the `main` symbol; it never executes
# the function, so it silently passes even when the CLI is broken. Shared
# between the pre-install auto-heal loop below and the post-install hard
# probe at the end of this script.
probe_vllm_cli() {
    vllm --help 2>&1 >/dev/null
}

# vLLM nightlies periodically add eager imports of packages that aren't in
# their declared deps (e.g. `pandas` from vllm/_aiter_ops.py). Auto-install
# any ModuleNotFoundError modules so the job keeps going. Capped to avoid
# infinite loops; every auto-install is logged so the drift is visible in
# the build output. ImportError with a missing top-level name (e.g. a
# transformers/vLLM API break) bails immediately since reinstalling the
# package wouldn't recover.
dump_transformers_state() {
    # Called whenever a CLI probe fails with anything other than a clean
    # ModuleNotFoundError. This failure mode has been reproducible only on
    # the K3s pods, never on a fresh local venv with identical versions,
    # so we need a direct view of the running pod's filesystem + Python
    # state to make progress.
    echo "=================== DIAGNOSTIC DUMP ===================" >&2
    echo "--- uv pip list (relevant packages) ---" >&2
    uv pip list 2>/dev/null | grep -iE "^(transformers|tokenizers|huggingface|safetensors|vllm|torch) " >&2 || true
    local tf_dir=/opt/venv/lib/python3.12/site-packages/transformers
    echo "--- transformers directory listing ---" >&2
    ls -la "${tf_dir}/__init__.py" "${tf_dir}/__pycache__/__init__.cpython-312.pyc" 2>&1 >&2 || true
    echo "--- dist-info METADATA version ---" >&2
    grep -m1 "^Version:" /opt/venv/lib/python3.12/site-packages/transformers-*.dist-info/METADATA 2>/dev/null >&2 || true
    echo "--- transformers/__init__.py: __version__ line ---" >&2
    grep -n "^__version__" "${tf_dir}/__init__.py" 2>/dev/null >&2 || true
    echo "--- transformers/__init__.py: 'generation' key in _import_structure ---" >&2
    awk '/"generation":/,/\]/' "${tf_dir}/__init__.py" 2>/dev/null | head -20 >&2 || true
    echo "--- Python sees: ---" >&2
    python - <<'PY' >&2 2>&1 || true
import sys, importlib
print(f"sys.executable = {sys.executable}")
print(f"sys.path = {sys.path}")
try:
    import transformers
    print(f"transformers.__file__ = {transformers.__file__}")
    print(f"transformers.__version__ = {transformers.__version__}")
    print(f"type(transformers) = {type(transformers).__name__}")
    print(f"'GenerationConfig' in dir(transformers) = {'GenerationConfig' in dir(transformers)}")
    cs2m = getattr(transformers, "_class_to_module", None)
    print(f"has _class_to_module = {cs2m is not None}")
    if cs2m is not None:
        print(f"GenerationConfig in _class_to_module = {'GenerationConfig' in cs2m}")
        print(f"first 5 keys = {list(cs2m.keys())[:5]}")
    import_struct = getattr(transformers, "_import_structure", None)
    print(f"has _import_structure = {import_struct is not None}")
    if import_struct is not None:
        print(f"'generation' in _import_structure = {'generation' in import_struct}")
        print(f"_import_structure['generation'] = {import_struct.get('generation')}")
    try:
        from transformers import GenerationConfig
        print("DIRECT IMPORT of GenerationConfig WORKED")
    except Exception as e:
        print(f"DIRECT IMPORT failed: {type(e).__name__}: {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
PY
    echo "=================== END DIAGNOSTIC DUMP ===================" >&2
}

MAX_AUTO_INSTALL=5
for i in $(seq 1 "$MAX_AUTO_INSTALL"); do
    if err=$(probe_vllm_cli); then
        break
    fi
    mod=$(printf '%s\n' "$err" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | head -1)
    if [[ -z "$mod" ]]; then
        echo "vLLM import failed with a non-ModuleNotFoundError:" >&2
        echo "$err" >&2
        dump_transformers_state
        exit 1
    fi
    if [[ "$i" == "$MAX_AUTO_INSTALL" ]]; then
        echo "Hit $MAX_AUTO_INSTALL auto-install retries; last missing module: $mod" >&2
        echo "$err" >&2
        exit 1
    fi
    echo "Auto-installing missing vLLM runtime dep: $mod"
    uv pip install "$mod"
done

echo "--- :mag: Verifying torch CUDA matches system nvcc"
# Sanity check: fail fast with a clear message if the cu130 pin above
# somehow didn't produce a cu13x torch. Previously this mismatch surfaced
# deep inside ninja as a cryptic `cusparse.h: No such file or directory`;
# catching it here makes the failure mode obvious.
python - <<'PY'
import subprocess, sys, torch
tc = torch.version.cuda or ""
try:
    nv = subprocess.check_output(["nvcc", "--version"], text=True)
    sys_major = next(
        (line.split("release ")[1].split(",")[0].split(".")[0]
         for line in nv.splitlines() if "release " in line),
        "",
    )
except Exception:
    sys_major = ""
torch_major = tc.split(".")[0] if tc else ""
print(f"torch.version.cuda={tc!r}; system nvcc major={sys_major!r}")
if torch_major and sys_major and torch_major != sys_major:
    sys.exit(
        f"CUDA major mismatch: torch={torch_major} vs nvcc={sys_major}. "
        "Check the vLLM nightly cu130 index pin in setup-env.sh."
    )
PY

echo "--- :python: Installing LMCache from source"
# Skip setuptools_scm git describe; the repo carries non-PEP-440 tags
# (nightly, nightly-cu13) that crash the newer vcs_versioning backend.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
uv pip freeze | sort > /tmp/env-before-lmcache.txt
uv pip install -e . --no-build-isolation
uv pip freeze | sort > /tmp/env-after-lmcache.txt
if ! diff -q /tmp/env-before-lmcache.txt /tmp/env-after-lmcache.txt >/dev/null; then
    echo "--- :warning: Packages changed during LMCache install"
    diff /tmp/env-before-lmcache.txt /tmp/env-after-lmcache.txt || true
fi

echo "--- :broom: Post-install bytecode eviction"
# Belt-and-suspenders companion to the pre-install eviction above: clear
# __pycache__ again after the LMCache editable install, which may have
# triggered fresh imports (setuptools_scm, pyproject build backend) and
# deposited new .pyc files that reference now-downgraded packages.
find /opt/venv/lib/python3.12/site-packages -type d -name __pycache__ \
    -exec rm -rf {} + 2>/dev/null || true

echo "--- :mag: Post-install CLI chain probe"
# The LMCache editable install can downgrade transitive deps to honor the
# caps in requirements/common.txt. If that leaves the env in a state where
# `vllm --help` cannot complete its import chain (vllm.entrypoints.cli.
# main.main() -> vllm.entrypoints.cli.benchmark.main -> vllm.config ->
# vllm.transformers_utils.config -> `from transformers import ...`), the
# only other signal is a 180s `wait_for_server` timeout inside each test
# harness. Re-probe the full chain here so broken envs fail fast with the
# actual traceback instead of a generic timeout.
if err=$(probe_vllm_cli); then
    echo "vLLM CLI import chain OK post-install."
else
    echo "FATAL: vLLM CLI import chain broken after LMCache install." >&2
    echo "--- Traceback ---" >&2
    echo "$err" >&2
    echo "--- Installed packages ---" >&2
    uv pip freeze >&2 || true
    exit 1
fi

echo "--- :white_check_mark: Environment ready"
python -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation')"
