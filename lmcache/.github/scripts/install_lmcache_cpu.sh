#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install lmcache in editable, CPU-only mode (NO_GPU_EXT=1).
# Assumes vLLM (and therefore torch) was already installed by
# install_vllm_cpu.sh, so we use --no-deps to keep that pinned torch
# in place.
#
# Usage:
#   install_lmcache_cpu.sh           # plain pip
#   PIP_BIN="uv pip" install_lmcache_cpu.sh

set -euo pipefail

PIP_BIN="${PIP_BIN:-pip}"

export NO_GPU_EXT=1
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0}"

${PIP_BIN} install --upgrade pip
${PIP_BIN} install -r requirements/build.txt
${PIP_BIN} install -r requirements/common.txt
${PIP_BIN} install -r requirements/cli.txt
${PIP_BIN} install -e . --no-deps --no-build-isolation

python -c "import lmcache, vllm; \
print('lmcache:', lmcache.__version__, 'vllm:', vllm.__version__)"
