# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from vllm.entrypoints.serve.profile.api_router import start_profile, stop_profile


@pytest.mark.parametrize(
    ("endpoint", "method", "message"),
    [
        (start_profile, "start_profile", "Proton unavailable"),
        (stop_profile, "stop_profile", "Proton finalize failed"),
    ],
)
def test_profile_errors_are_returned_to_api_callers(endpoint, method, message):
    engine = SimpleNamespace(**{method: AsyncMock(side_effect=RuntimeError(message))})
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=engine))
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(request))

    assert exc_info.value.status_code == 500
    assert message in exc_info.value.detail
