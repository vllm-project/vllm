# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono decode layer of the DeepSeek-V4 family: one decoder layer of a
decode step in two persistent FlyDSL launches, its TP all-reduces in-kernel
(README.md). A model's widths are a ``spec.MonoSpec``; the host side is
``runner``, vLLM's side ``decode`` (docked by ``deepseek_v41/amd/mono_decode``
and ``deepseek_v4/amd/mono_decode``)."""
