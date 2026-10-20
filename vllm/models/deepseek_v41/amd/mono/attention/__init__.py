# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono layer's attention stages: ``front`` (K1: wqkv, q / kv norms and the
KV insert, wq_b) and ``back`` (K2: split attention, combine, wo_a, wo_b)."""
