# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine.online import OnlineEngineClient as V1OnlineEngineClient

OnlineEngineClient = V1OnlineEngineClient  # type: ignore
"""The `OnlineEngineClient` class is an alias of 
[vllm.v1.engine.online.OnlineEngineClient][]."""
