# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 decode MonoKernel for ROCm (MI350X, TP8): one fused kernel per MoE decoder
layer on pure-decode steps.

``kernel/`` is a vendored copy of ROCm/ATOM PR #2435's ``atom/model_ops/monokernel``
(GLM subset, head 45e4b55d) with imports rewritten to this package. Tests and golden
references live in ``tests/models/deepseek_v32/amd_mono/``."""
