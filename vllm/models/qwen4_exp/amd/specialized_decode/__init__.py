# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next specialized decode for ROCm (MI355X, TP2), step mode:
FlyDSL MonoKernels for each decoder layer's blocks on pure decode steps of at
most 16 rows. Enabled by ``--monokernel``.

``hooks.py`` holds the model-side hooks (``SpecializedDecodeMixin``); the model
runner state that builds the executor is ``qwen4_exp/amd/model_state.py``.
``runtime/`` is the host side: the config check (``runtime/__init__.py``), the
executor (``runtime/decode.py``), the launches (``runtime/layers.py``) and the
model and cache refusals (``runtime/guards.py``). ``mono/`` is the kernel
source. Tests live in ``tests/models/qwen4_exp/amd_specialized_decode/``."""
