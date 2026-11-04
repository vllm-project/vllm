# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Building blocks shared by the CuTe DSL kernels that fuse a MoE layer's tail
(top-k finalize, TP all-reduce, norms) across models."""
