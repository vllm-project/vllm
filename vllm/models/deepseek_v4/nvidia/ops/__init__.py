# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA-only (cutedsl/cutlass) kernels for DeepSeek V4.

These modules import ``cutlass``/``cutedsl`` at module top level, so they must
not be imported on non-CUDA platforms. Callers should gate on
``vllm.utils.import_utils.has_cutedsl()`` before importing from here.

This ``__init__`` deliberately imports nothing: re-exporting the cutedsl
modules here would eagerly ``import cutlass`` (initializing the CUDA driver) for
anyone who imports ``vllm.models.deepseek_v4``, breaking forked subprocesses.
Import the leaf modules directly under a ``has_cutedsl()``/``is_cuda()`` gate.
"""
