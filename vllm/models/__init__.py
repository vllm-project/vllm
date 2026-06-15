# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-specific model implementations.

Each sub-package targets a single checkpoint and trades generality for
hand-tuned kernels and fused execution paths. The packages here currently
host only the custom kernels; the specialized model wiring is added
separately.
"""
