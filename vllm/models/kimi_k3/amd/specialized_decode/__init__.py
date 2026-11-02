# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 specialized decode for ROCm (MI355X, TP8), layer mode: FlyDSL
MonoKernels per MoE decoder layer on pure spec-verify steps (K1 / K2 on the KDA
layers, M1a / M1b / K2 on the MLA layers, DCP1 or DCP8), inside the stock layer
loop. Enabled by ``additional_config={"decode_backend": "latency_specialized"}``.

``hooks.py`` holds the model-side hooks (``SpecializedDecodeMixin``) and
``model_state.py`` the model runner state that builds the executor. ``runtime/`` is
the host side: the config check (``runtime/__init__.py``), the executor
(``runtime/decode.py``), the layer kernels, the per-rank launch state and the later
refusals. ``mono/`` is the kernel source. Tests live in
``tests/models/kimi_k3/amd_specialized_decode/``."""
