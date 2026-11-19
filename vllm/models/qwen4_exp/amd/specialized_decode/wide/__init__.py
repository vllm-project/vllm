# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide decode kernels: one persistent launch per decoder layer, each a
sequence of grid-wide phases over every row of the step."""
