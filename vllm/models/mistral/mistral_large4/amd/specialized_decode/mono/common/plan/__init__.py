# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM), MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Build-time contracts of a mono kernel, plain Python (no torch, no FlyDSL):
`trace` records a kernel's mailbox accesses while it is traced, `check` proves
the recorded hand-offs against the regions' declarations."""
