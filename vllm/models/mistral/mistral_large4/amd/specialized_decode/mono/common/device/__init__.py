# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM), MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Device-side mechanisms every mono kernel shares, one module each: `ops`
(wave / math / packing primitives) and `sync` (the tagged mailbox)."""
