# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSv4-specific non-attention layer customizations.

Holds vendored layer files whose math or carve-outs are DSv4-specific
(``rotary_embedding`` is restricted to ``DeepseekV4ScalingRotaryEmbedding``;
``mhc`` is multi-head compression used only by DSv4). Generic layer
files live under ``../shared/layers/``.
"""
