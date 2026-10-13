# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5 MonoKernel integration (ROCm gfx950): one launch per decoder layer of a
pure decode step. The kernel source lives in ``..ops.mono``."""
