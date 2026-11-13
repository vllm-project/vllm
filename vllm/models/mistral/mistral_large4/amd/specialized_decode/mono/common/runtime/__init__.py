# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM), MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Host-side mechanisms of a mono runner, one module each (imported directly, so
the ones without an AITER dependency load without it): `consensus` (TP-uniform
decisions), `peer_memory` (symmetric peer buffers), `abi` (kernel argument
tables), `compile` (compile-only builds)."""
