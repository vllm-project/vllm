# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor on Hopper (SM90): the executor's schedule
(fused all-reduce + mHC boundaries, side streams, tables shared across layers,
recorded metadata builds) over the Hopper stock kernels, with one function per
kernel choice (``kernels``)."""
