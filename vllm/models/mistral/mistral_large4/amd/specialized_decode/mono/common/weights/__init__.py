# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM), MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Host-side weight preparation every mono kernel needs: the storage
vocabulary (`config`), MX format conversion (`formats`), and the packing
that turns a loaded tensor into what a kernel reads (`packing`, `layer`).
"""
