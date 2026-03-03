# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine.launch import LaunchEngineClient as V1LaunchEngineClient

LaunchEngineClient = V1LaunchEngineClient  # type: ignore
"""The `LaunchEngineClient` class is an alias of
[vllm.v1.engine.launch.LaunchEngineClient][]."""
