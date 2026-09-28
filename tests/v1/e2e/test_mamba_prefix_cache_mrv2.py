# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.utils import create_new_process_for_each_test
from tests.v1.e2e._mamba_prefix_cache import _run_mamba_prefix_cache_mrv2


@create_new_process_for_each_test()
def test_mamba_prefix_cache_mrv2(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv2(monkeypatch, async_scheduling=False)


@create_new_process_for_each_test()
def test_mamba_prefix_cache_mrv2_async(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv2(monkeypatch, async_scheduling=True)
