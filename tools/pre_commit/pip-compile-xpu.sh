#!/bin/bash
# TODO(xpu-wheel-testing): this wrapper exists only to bypass the corporate
# proxy for the temporary internal test index (10.239.182.107) that hosts the
# `triton` xpu compat shim. Once triton-xpu ships a public `triton` package
# on a permanent index, drop this wrapper and switch pip-compile-xpu back to
# the plain astral-sh/uv-pre-commit `pip-compile` hook.
set -euo pipefail

export NO_PROXY="${NO_PROXY:-},10.239.182.107"
export no_proxy="${no_proxy:-},10.239.182.107"

exec uv pip compile \
  requirements/test/xpu.in \
  -c requirements/xpu.txt \
  -o requirements/test/xpu.txt \
  --index-strategy unsafe-best-match \
  --python-platform x86_64-manylinux_2_39 \
  --python-version "3.12"
