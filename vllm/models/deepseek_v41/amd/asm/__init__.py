# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-written gfx950 (MI355X) decode kernels for DeepSeek-V4.1-Flash at TP4:
the decoder layers' decode steps of 17-256 rows (``decode.AsmDecodeLayer``)
next to the mono decode layer, which keeps those of up to 16. The kernels are
generated as assembly at first use, built with LLVM (clang, ld.lld) into code
objects cached under $VLLM_CACHE_ROOT/dsv41_asm and launched through HIP."""
