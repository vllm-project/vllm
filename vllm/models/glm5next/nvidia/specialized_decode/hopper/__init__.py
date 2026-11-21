# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor on Hopper (SM90): the GB300 executor's plan,
layer loop and boundaries (`runtime/`) over the Hopper stage kernels, chosen per
token count by `policy.HOPPER`. See `AGENTS.md`."""
