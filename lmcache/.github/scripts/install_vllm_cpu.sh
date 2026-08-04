#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the prebuilt CPU-only vLLM wheel (`vllm-cpu-nightly`) plus a
# `vllm-<ver>+cpu.dist-info` alias.
#
# Why the alias: the wheel installs the `vllm/` package but registers
# its dist metadata under `vllm-cpu-nightly`. vLLM's CLI / internal
# callers do `importlib.metadata.version("vllm")` (distribution name,
# not import name); without the alias that raises PackageNotFoundError
# and `vllm serve` won't start. The `+cpu` local label is also needed
# so `cpu_platform_plugin()` activates the CPU platform (it greps the
# dist metadata for the substring "cpu"); our build strips `+cpu`
# before PyPI upload because PyPI rejects local versions.
#
# Usage:
#   install_vllm_cpu.sh           # use plain `pip`
#   PIP_BIN="uv pip" install_vllm_cpu.sh
#                                 # use `uv pip` and pass extra flags
#                                 # via PIP_INSTALL_EXTRA_ARGS
#   VLLM_CPU_NIGHTLY_SPEC="vllm-cpu-nightly==0.25.2.dev202607230832" \
#     install_vllm_cpu.sh         # pin a specific nightly build (used
#                                 # by the macOS CI leg to dodge a
#                                 # broken upstream wheel; see the
#                                 # workflow for context)
#
# Idempotent: re-running just rewrites the alias.

set -euo pipefail

PIP_BIN="${PIP_BIN:-pip}"
PIP_INSTALL_EXTRA_ARGS="${PIP_INSTALL_EXTRA_ARGS:-}"
VLLM_CPU_NIGHTLY_SPEC="${VLLM_CPU_NIGHTLY_SPEC:-vllm-cpu-nightly}"

# `--extra-index-url` is required because the wheel pins torch==2.11.0
# which only lives on the pytorch CPU index. Harmless on macOS.
${PIP_BIN} install "numpy<2"
# Cap apache-tvm-ffi below 0.1.13 (published 2026-07-30): xgrammar's
# macOS arm64 wheels were built against the 0.1.12 ABI and segfault at
# import (xgrammar::__TVMFFIStaticInitFunc0) when the 0.1.13 dylib is
# loaded, which kills `vllm serve` before it can bind its port. Linux
# happens to load 0.1.13 fine, but both legs get the cap so the whole
# matrix runs the validated version. Lift the cap once xgrammar ships
# wheels built against apache-tvm-ffi>=0.1.13.
# shellcheck disable=SC2086
${PIP_BIN} install "${VLLM_CPU_NIGHTLY_SPEC}" "apache-tvm-ffi<0.1.13" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  ${PIP_INSTALL_EXTRA_ARGS}

python - <<'PY'
import importlib.metadata as md
import pathlib
import shutil

dist = md.distribution("vllm-cpu-nightly")
ver = dist.version
fake_ver = f"{ver}+cpu"
site_root = pathlib.Path(dist.locate_file(""))
info_name = next(
    p.parts[0] for p in (dist.files or [])
    if p.parts and p.parts[0].endswith(".dist-info")
)
src = site_root / info_name
dst = src.with_name(f"vllm-{fake_ver}.dist-info")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
meta = dst / "METADATA"
txt = meta.read_text()
txt = txt.replace("Name: vllm-cpu-nightly", "Name: vllm", 1)
txt = txt.replace(f"Version: {ver}", f"Version: {fake_ver}", 1)
meta.write_text(txt)
print(f"Aliased {src.name} -> {dst.name}")
print("vllm version (via importlib.metadata):", md.version("vllm"))
PY

python -c "import vllm, torch; \
print('vllm:', vllm.__version__, 'torch:', torch.__version__, \
      'cuda:', torch.cuda.is_available())"
