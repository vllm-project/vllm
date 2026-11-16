# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next specialized decode on MI355X (`--monokernel`).

`hooks` holds the model's entry points, `runtime` the refusals, the plan of
each step and the executor, `wide` the persistent kernels for decode steps
of 1 to 192 rows at TP2, 1 to 64 at TP1 (one launch per GDN layer, two around
the QSA attention on a QSA layer). See `README.md`."""
