# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Stage-1 MoE CPU offload expert paging CLI/config wiring.
# Importing this module patches EngineArgs with the opt-in flags only. It does
# not modify runtime execution behavior.
import vllm.engine.moe_offload_cli  # noqa: F401
