# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.2 specialized decode for ROCm (MI350X, TP8), layer mode: one fused FlyDSL
MonoKernel per MoE decoder layer on pure-decode steps, inside the stock layer loop.
Enabled by ``additional_config={"decode_backend": "latency_specialized"}``.

``runtime/`` is the host side: the config check (``runtime/__init__.py``), the executor
(``runtime/decode.py``) and the per-layer state. ``flydsl/`` is the vendored FlyDSL
kernel source (provenance in each file's header). Tests live in
``tests/models/deepseek_v32/amd_specialized_decode/``."""
