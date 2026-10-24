# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone own-KV runtime for target-conditioned speculators.

The standalone process plays the scheduler and model runner for the draft
model alone, so the existing speculators can run there unchanged:

- kv: the draft-only KV block pool; blocks are never shared with the target.
- step: per-sequence bookkeeping that turns protocol rows into the per-step
  fields a model runner hands a speculator (positions, sequence lengths,
  slot mapping, block table updates, sampled/rejected counts).
- adapter: the server adapter that drives the planner and a device-side
  step executor.
"""
