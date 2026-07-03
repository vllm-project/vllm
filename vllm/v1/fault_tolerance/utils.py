# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import msgspec

# All2all backends that support fault-tolerant timeout + rank masking,
# required for FT under DP+EP MoE deployments.
FT_BACKEND_SET = frozenset({"deepep_low_latency", "nixl_ep"})


class FaultToleranceResult(msgspec.Struct):
    request_id: str
    success: bool
    reason: str | None = None


class FaultToleranceRequest(msgspec.Struct):
    instruction: str
    params: dict[str, Any]
    request_id: str = ""
