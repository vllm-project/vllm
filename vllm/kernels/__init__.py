# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel implementations for vLLM."""

from vllm.utils.import_utils import has_helion

from . import aiter_ops, oink_ops, vllm_c, xpu_ops

if has_helion():
    from .helion import ir_ops as _helion_ir_ops  # noqa: F401

__all__ = ["vllm_c", "aiter_ops", "oink_ops", "xpu_ops"]
