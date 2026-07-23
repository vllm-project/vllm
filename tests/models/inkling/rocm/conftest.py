# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm.platforms import current_platform


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            pytest.skip("requires CDNA4")
        return

    rocm_test_dir = Path(__file__).parent
    skip_rocm = pytest.mark.skip(reason="requires ROCm")
    for item in items:
        if item.path.is_relative_to(rocm_test_dir):
            item.add_marker(skip_rocm)
