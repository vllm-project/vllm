#!/usr/bin/env bash
set -euo pipefail

echo "[dev-setup] Strict editable install starting"

# Choose writable tmp
if [[ -d /opt/work ]]; then
	export TMPDIR=/opt/work/tmp
else
	export TMPDIR=/tmp
fi
mkdir -p "$TMPDIR" || true

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0+local}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}
export MAX_JOBS=${MAX_JOBS:-4}
export VLLM_INSTALL_PUNICA_KERNELS=${VLLM_INSTALL_PUNICA_KERNELS:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0 8.6 8.9 9.0 12.0 13.0"}
export CUDAARCHS=${CUDAARCHS:-"80;86;89;90;120"}

# sitecustomize shim to swallow PermissionError on utime/chmod/copy_file
SITE_SHIM_DIR="$TMPDIR/pyshim"
mkdir -p "$SITE_SHIM_DIR"
cat > "$SITE_SHIM_DIR/sitecustomize.py" <<'PY'
import os, sys, importlib
_ou, _oc = os.utime, os.chmod
def _wrap(fn):
	def inner(*a, **k):
		try: return fn(*a, **k)
		except PermissionError: return None
	return inner
os.utime = _wrap(_ou); os.chmod = _wrap(_oc)
def _patch_copy(mod_name):
	try:
		m = importlib.import_module(mod_name)
	except Exception:
		return
	cf = getattr(m, 'copy_file', None)
	if not cf: return
	def _cf(src, dst, *a, **k):
		try: return cf(src, dst, *a, **k)
		except PermissionError:
			import shutil; shutil.copyfile(src,dst); return (dst,1)
	m.copy_file = _cf
for _n in ('distutils.file_util','setuptools._distutils.file_util'): _patch_copy(_n)
if '/workspace' not in sys.path: sys.path.insert(0,'/workspace')
PY

echo "📦 Installing vLLM (editable, strict) ..."
PYTHONPATH="$SITE_SHIM_DIR:${PYTHONPATH:-}" FETCHCONTENT_BASE_DIR="$TMPDIR/deps" \
	pip install -e . --no-deps --no-build-isolation --verbose --config-settings editable-legacy=true || {
		echo "❌ Editable install failed (strict mode)."; exit 1; }
echo "✅ Editable install complete"

python - <<'PY'
import os, vllm
print('vLLM version:', getattr(vllm,'__version__','unknown'))
print('FA3_MEMORY_SAFE_MODE:', os.environ.get('FA3_MEMORY_SAFE_MODE'))
PY

echo "[dev-setup] Done"
