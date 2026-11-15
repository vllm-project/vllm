# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM), MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Mono decode: a model's decode step as one persistent kernel per layer, with the
tensor-parallel reductions inside the kernel.

This package holds what every mono model shares -- the mechanisms: `runtime`
(host: TP consensus, peer memory, kernel argument tables, compile-only builds),
`plan` (plain Python: the execution model, build keys, the traced hand-off
contract) and `device` (traced inside kernels: the tagged mailbox, wave / math
primitives). The kernels, their layouts and schedules stay in the model's own
`mono` package. It imports no model and holds no model constant.
"""
