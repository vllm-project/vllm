# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono decode: a model's decode step as one persistent kernel per layer, with the
tensor-parallel reductions inside the kernel.

This package holds what every mono model shares -- the mechanisms: ``runtime``
(host: TP consensus, peer memory, kernel argument tables, compile-only builds),
``plan`` (plain Python: the execution model, build keys, the traced hand-off
contract) and ``device`` (traced inside kernels: the tagged mailbox, wave / math
primitives). The kernels, their layouts and schedules stay in the model's own
``mono`` package. It imports no model and holds no model constant.
"""
