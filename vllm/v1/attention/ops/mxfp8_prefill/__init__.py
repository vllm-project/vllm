# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal MXFP8 prefill: operand preparation, normal bulk, boundary merge."""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

# sdpa_mxfp8(causal_exclude_own_block=32), NVIDIA/cudnn-frontend#2563.
MIN_CUDNN_FRONTEND = "1.33.0"


def cudnn_frontend_problem() -> str | None:
    """Why the installed cuDNN Frontend cannot run MXFP8 prefill, or None."""
    try:
        found = version("nvidia-cudnn-frontend")
    except PackageNotFoundError:
        return f"requires nvidia-cudnn-frontend>={MIN_CUDNN_FRONTEND}"
    if Version(found) < Version(MIN_CUDNN_FRONTEND):
        return f"requires nvidia-cudnn-frontend>={MIN_CUDNN_FRONTEND} (found {found})"
    return None
