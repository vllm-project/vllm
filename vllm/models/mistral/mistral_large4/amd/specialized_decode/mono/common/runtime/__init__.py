# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side mechanisms of a mono runner, one module each (imported directly, so
the ones without an AITER dependency load without it): ``consensus`` (TP-uniform
decisions), ``peer_memory`` (symmetric peer buffers), ``abi`` (kernel argument
tables), ``compile`` (compile-only builds)."""
