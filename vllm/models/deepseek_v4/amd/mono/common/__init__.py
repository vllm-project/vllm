# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/__init__.py
"""The mono kernels' shared mechanisms: ``ops`` (wave / math primitives, the TP
group's peer buffers), ``sync`` (the tagged mailbox), ``mx`` (FP8 / MX
arithmetic), ``plan`` (execution model, layouts, build keys) and
``peer_memory`` (symmetric peer buffers)."""
