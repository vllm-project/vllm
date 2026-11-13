# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-2.4T-A95B specialized decode on MI355X (TP8, `--monokernel`).

`hooks` holds the model's entry points, `model_state` the model runner
state that builds the executor, `runtime` the refusals, the plan of each
step and the executor, `wide` the persistent kernels for decode steps of 1
to 64 rows, `mono` #60833's layer and GDN blocks for steps of up to 16. See
`README.md`."""
