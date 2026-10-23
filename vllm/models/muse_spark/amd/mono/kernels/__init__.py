# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL sources of the Muse Spark decode monokernel (see ../README.md).

``k3.build_vllm(S, W, next_rope)`` (8 / 16 rows), ``k5.build_vllm`` (HA / HB)
and ``k5m.build_vllm`` (HM) for 32-128 rows are the builds vLLM launches:
TP = EP = 8, the router output multiplier folded into the router weight, q
without the QK scale, bf16 KV pages in vLLM's LBHNC layout, and vLLM's
shuffled MXFP4 expert weights and scales read in place. With ``attn="swa"`` /
``"full"`` the K3 / HA builds also run the layer's verify attention at their
head (``attn_frag.make(..., kvl="lbhnc")``: vLLM's bf16 KV cache, block table
and seq_lens). Importing these modules needs FlyDSL and AITER's FlyDSL
helpers.
"""
