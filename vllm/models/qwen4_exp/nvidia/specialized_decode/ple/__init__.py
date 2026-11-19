# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PLE layer of the Qwen4Exp specialized decode at TP > 1: the
vocab-parallel FP8 lookup gathered and dequantized over NVLink in one kernel
(`gather.py`). Nothing is imported here: the module holds a CuTe DSL kernel
(CUDA builds only)."""
