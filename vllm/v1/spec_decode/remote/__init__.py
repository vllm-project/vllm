# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated-device placement for target-conditioned own-KV speculators.

This package contains the placement-independent building blocks for running
EAGLE / P-EAGLE / native-MTP style speculators in a standalone process on a
dedicated same-host GPU outside the target tensor-parallel group:

- capabilities: what a speculator adapter/checkpoint supports and which
  target features it needs every round.
- protocol: control-plane message types exchanged between the verifier and
  the standalone speculator server.
- state: client-side handle/sequence state machines and the
  placement-neutral batch/result types the session exposes to the runner.
- transport: the framed inline stream transport (debug data plane).
- session: the verifier-side session (handshake, sequence lifecycle,
  dispatch/collect, failure latching).
- server: the standalone server loop, idempotent sequence registry, the
  adapter contract, and a deterministic fake adapter for tests.

Only ``session`` operates on tensors. The server, protocol, registry, and
state logic exchange raw data-plane frames, so they need no GPU and can be
unit tested in isolation.
"""
