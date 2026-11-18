# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL helpers copied from Kimi K3's specialized decode (`host`, `policy`,
`primitives`, the split-K GEMM in `splitk` and `projection`) and from DeepSeek-V4.1's
(`finalize_publish`), unchanged apart from imports, so this package does not depend
on those models' trees.
"""
