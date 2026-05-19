# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA-only (cutedsl/cutlass) kernels for DeepSeek V4.

These modules import ``cutlass``/``cutedsl`` at module top level, so they must
not be imported on non-CUDA platforms. Callers should gate on
``vllm.utils.import_utils.has_cutedsl()`` before importing from here.
"""
