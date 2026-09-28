# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.utils import create_new_process_for_each_test
from tests.v1.e2e._mamba_prefix_cache import _run_mamba_prefix_cache_mrv1


@create_new_process_for_each_test("spawn")
def test_mamba_prefix_cache_mrv1(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv1(monkeypatch, async_scheduling=False)


@create_new_process_for_each_test("spawn")
def test_mamba_prefix_cache_mrv1_async(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv1(monkeypatch, async_scheduling=True)
