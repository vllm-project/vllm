# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def require_rocm_cdna4() -> None:
    if not current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            pytest.skip("requires CDNA4")
