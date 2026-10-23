# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL sources of the Muse Spark K3 decode monokernel.

``k3.build_vllm(S, W, next_rope)`` is the build vLLM launches: TP = EP = 8,
the router output multiplier folded into the router weight, q without the QK
scale, bf16 KV pages in vLLM's LBHNC layout, and vLLM's shuffled MXFP4 expert
weights and scales read in place. Importing these modules needs FlyDSL and
AITER's FlyDSL helpers.
"""
