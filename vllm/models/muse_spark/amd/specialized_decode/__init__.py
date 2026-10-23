# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark specialized decode for ROCm (MI355X, TP8), layer mode (RFC #60904):
one persistent FlyDSL MonoKernel launch (K2) per routed-MXFP4 decoder layer on
pure decode / DSpark verify steps, inside the stock layer loop. Enabled by
``--monokernel`` or ``additional_config={"decode_backend": "latency_specialized"}``.

Phase A: vLLM runs each kernel layer's attention up to o_proj's input; K2 runs
o_proj, its all-reduce, the attention boundary, the router and routing, the
pooler, the MXFP4 experts, the reduce, the post-expert norm and the unpooler.

``hooks.py`` holds the model-side hooks (``SpecializedDecodeMixin``);
``vllm/models/muse_spark/amd/model_state.py`` the model runner state that builds
the executor. ``runtime/`` is the host side: the config check
(``runtime/__init__.py``), the executor (``runtime/decode.py``), the kernel
layers, the per-rank launch state and the later refusals. ``mono/`` is the
kernel source; ``mono/spec.py`` is the contract its packages share. Tests live in
``tests/models/muse_spark/amd_specialized_decode/``."""
