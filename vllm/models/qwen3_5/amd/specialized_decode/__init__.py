# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 (2.4T-A95B) specialized decode for ROCm (MI355X, TP8), step mode:
FlyDSL MonoKernels for each decoder layer on pure decode steps of at most 16
rows. Enabled by ``--monokernel``.

``hooks.py`` holds the model-side hooks (``SpecializedDecodeMixin``),
``model_state.py`` the model runner state that builds the executor.
``runtime/`` is the host side: the config check (``runtime/__init__.py``), the
executor (``runtime/decode.py``), the launches (``runtime/layers.py``) and the
model and cache refusals (``runtime/guards.py``). ``mono/`` is the kernel
source. Tests live in ``tests/models/qwen3_5/amd_specialized_decode/``."""
