# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono MoE launch's shared mechanisms: ``plan`` (constants, the workspace
and control-word layout), ``ops`` (device primitives) and ``sync`` (tickets,
counters and flags between workgroups)."""
