# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-on small-row MoE fusion over the Qwen3.5 XPU eager backbone.

Post-load hooks bind eligible weights; unsupported calls use the stock layer.
This package does not replace the model's whole decode step.
"""

MAX_DECODE_ROWS = 8
MAX_ROUTED_ROWS = 512
