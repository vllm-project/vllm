# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in small-row MoE fusion over the Qwen3.5 XPU eager backbone.

Post-load hooks bind qualified weights; ineligible steps use the stock layer.
This package does not replace the model's whole decode step.
"""

MAX_DECODE_ROWS = 8
MAX_ROUTED_ROWS = 512


def is_requested(vllm_config) -> bool:
    """Select the optional provider through the existing additional-config API."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend not in ("auto", "latency_specialized"):
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    return backend == "latency_specialized"
