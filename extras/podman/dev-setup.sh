#!/usr/bin/env bash#!/usr/bin/env bash

set -euo pipefail# Robust setup entrypoint: prefer extras/dev-setup.sh,

# otherwise use the image-provided /home/vllmuser/setup_vllm_dev.sh.

echo "[dev-setup] Strict editable install starting"set -euo pipefail

python - <<'PY'

import os, textwrap, sysSCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd)

site_dir = next(p for p in sys.path if p.endswith('site-packages'))EXTRAS_DIR=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)

sc = os.path.join(site_dir,'sitecustomize.py')

if not os.path.exists(sc):try_exec() {

    with open(sc,'w',encoding='utf-8') as f: f.write('# placeholder\n')	local target="$1"

# Append shims idempotently	if [[ -f "$target" ]]; then

content = open(sc,'r',encoding='utf-8').read()		# Normalize CRLF and avoid chmod on mounted FS

if '# VLLM_STRICT_EDITABLE_SHIM' not in content:		local tmp

    content += '\n# VLLM_STRICT_EDITABLE_SHIM\n'		tmp="$(mktemp /tmp/dev-setup-target.XXXX.sh)"

    content += textwrap.dedent('''\		tr -d '\r' < "$target" > "$tmp" 2>/dev/null || cp "$target" "$tmp"

import os, errno, builtins		chmod +x "$tmp" 2>/dev/null || true

import distutils.file_util as _dfu		exec "$tmp" "$@"

from functools import wraps	fi

}

# Swallow PermissionError on utime/chmod common on mounted FS

for _name in ('utime','chmod'):# 1) Current canonical path

    _orig = getattr(os,_name)if [[ -f "${EXTRAS_DIR}/dev-setup.sh" ]]; then

    def _wrap(fn,_o=_orig):	try_exec "${EXTRAS_DIR}/dev-setup.sh" "$@"

        @wraps(_o)fi

        def inner(*a,**k):

            try: return _o(*a,**k)# 2) Fallback: perform a minimal editable install inline (avoid chmod on /tmp)

            except PermissionError: return Noneecho "🔧 Setting up vLLM (inline fallback)..."

        return innercd /workspace

    setattr(os,_name,_wrap(_name,_orig))

# Ensure patches applied before building

# Intercept copy_fileif command -v apply-vllm-patches >/dev/null 2>&1; then

if hasattr(_dfu,'copy_file'):	apply-vllm-patches || true

    _orig_copy=_dfu.copy_filefi

    def copy_file(src,dst,*a,**k):

        try: return _orig_copy(src,dst,*a,**k)# Prefer /opt/work/tmp (mounted volume) if available, else /tmp

        except PermissionError: return (dst,0)if [[ -d /opt/work ]]; then

    _dfu.copy_file=copy_file	export TMPDIR=/opt/work/tmp

''')else

    open(sc,'w',encoding='utf-8').write(content)	export TMPDIR=/tmp

PYfi

mkdir -p "$TMPDIR" || true

# Ensure build deps minimal

export PIP_DISABLE_PIP_VERSION_CHECK=1# Build env knobs

export PIP_NO_CACHE_DIR=1export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}

export VLLM_INSTALL_PUNICA_KERNELS=${VLLM_INSTALL_PUNICA_KERNELS:-0}

# Install in editable strict modeexport MAX_JOBS=${MAX_JOBS:-4}

python -m pip install -e . --no-deps --no-build-isolation --config-settings editable-legacy=true -v# CUDA 13 toolchain dropped SM70/75; ensure we don't pass them to nvcc

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0 8.6 8.9 9.0 12.0 13.0"}

echo "[dev-setup] Done"export CUDAARCHS=${CUDAARCHS:-"80;86;89;90;120"}


# Install Python deps from repo (torch stack already in image)
if [[ -f requirements/common.txt ]]; then
	pip install -r requirements/common.txt || true
fi

# Avoid slow git describe during setuptools_scm by providing a pretend version
export SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0+local}

# Always install in editable mode. On some host filesystems (Windows mounts), setuptools may fail on os.utime;
# inject a sitecustomize shim to ignore PermissionError from utime during the build/copy step.
SITE_SHIM_DIR="$TMPDIR/pyshim"
mkdir -p "$SITE_SHIM_DIR"
cat > "$SITE_SHIM_DIR/sitecustomize.py" <<'PY'
"""sitecustomize: make editable install resilient on restrictive mounts.
Silently ignore PermissionError from utime/chmod and copy_file metadata ops.
NO FALLBACK: If editable still fails, build stops (user requested strict mode).
"""
import os, sys

_orig_utime = os.utime
_orig_chmod = os.chmod
def _safe_utime(*a, **k):
	try:
		return _orig_utime(*a, **k)
	except PermissionError:
		return None
def _safe_chmod(*a, **k):
	try:
		return _orig_chmod(*a, **k)
	except PermissionError:
		return None
os.utime = _safe_utime  # type: ignore
os.chmod = _safe_chmod  # type: ignore

def _patch_copy_file(module):
	if not module: return
	cf = getattr(module, 'copy_file', None)
	if not cf: return
	def _wrapped(src, dst, *a, **k):
		try:
			return cf(src, dst, *a, **k)
		except PermissionError:
			# Minimal fallback: raw bytes copy without metadata
			import shutil
			shutil.copyfile(src, dst)
			return (dst, 1)
	module.copy_file = _wrapped

for _mod_name in ('distutils.file_util', 'setuptools._distutils.file_util'):  # both paths
	try:
		import importlib
		_m = importlib.import_module(_mod_name)
		_patch_copy_file(_m)
	except Exception:
		pass

if '/workspace' not in sys.path:
	sys.path.insert(0, '/workspace')
PY

echo "📦 Installing vLLM in strict editable mode (no fallback)..."
PYTHONPATH="$SITE_SHIM_DIR:${PYTHONPATH:-}" FETCHCONTENT_BASE_DIR="$TMPDIR/deps" \
	pip install -e . --no-deps --no-build-isolation --verbose --config-settings editable-legacy=true || {
		echo "❌ Editable install failed (strict mode)."; exit 1; }
echo "✅ vLLM installed in editable mode."

python - <<'PY'
import os, vllm
print("vLLM version:", getattr(vllm, "__version__", "unknown"))
print("FA3_MEMORY_SAFE_MODE:", os.environ.get("FA3_MEMORY_SAFE_MODE"))
PY
