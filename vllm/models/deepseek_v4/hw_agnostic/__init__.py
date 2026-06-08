# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The hw-agnostic DeepSeek V4 path must never dispatch to the NVIDIA-only
# CUTeDSL kernels in ``nvidia/ops/``, even when ``cutlass`` happens to be
# importable. Disable the shared dispatch flag at package import time so the
# common ops and the compressor always pick their Triton fallbacks.
from vllm.models.deepseek_v4.common.cutedsl_flag import set_cutedsl_enabled

set_cutedsl_enabled(False)
