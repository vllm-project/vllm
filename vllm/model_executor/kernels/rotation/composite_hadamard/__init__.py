# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

from .tilelang import composite_hadamard
from .torch import composite_hadamard_torch
from .utils import factor_hadamard_dim, resolve_hadamard_dim

__all__ = [
    "composite_hadamard",
    "composite_hadamard_torch",
    "factor_hadamard_dim",
    "resolve_hadamard_dim",
]
