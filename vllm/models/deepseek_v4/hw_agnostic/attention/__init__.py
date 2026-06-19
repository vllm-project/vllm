# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSv4 hw-agnostic attention path.

Holds the DSv4 MLA layer, sparse-MLA / SWA / indexer / compressor
backends, and the Triton kernels under ``kernels/``. Vendored copies
that don't reach into vendor-specific kernels.
"""
