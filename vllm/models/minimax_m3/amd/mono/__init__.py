# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 fused per-layer decode of up to ``config.MAX_TOKENS`` tokens; entered
through ``dispatch.MonoDecode`` (``VLLM_ROCM_MINIMAX_M3_MONO=1``)."""
