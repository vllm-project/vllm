# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Internal building blocks for the Mooncake encoder-cache connector.

The public connector remains in :mod:`mooncake_ec_connector`.  This package
separates configuration, control-plane messaging, transfer state, registered
memory, and worker/scheduler orchestration so that each component has one
owner and can be tested independently.
"""
