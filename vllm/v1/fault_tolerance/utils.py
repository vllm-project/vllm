# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import msgspec

# Instructions accepted from external callers. The API router and the engine
# both validate against this; keep it the single source of truth so a new
# instruction cannot be allowed at one layer and rejected at the other.
ALLOWED_FT_INSTRUCTIONS = frozenset({"retry", "scale_down"})


class FaultToleranceResult(msgspec.Struct):
    request_id: str
    success: bool
    reason: str | None = None


class FaultToleranceRequest(msgspec.Struct):
    instruction: str
    params: dict[str, Any]
    request_id: str = ""
