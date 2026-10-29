# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("CPU-only test", allow_module_level=True)

from vllm.v1.worker.cpu_worker import _get_cpushm_dist_ident  # noqa: E402


def test_get_cpushm_dist_ident_preserves_worker_rendezvous():
    idents = [
        _get_cpushm_dist_ident(f"tcp://127.0.0.1:{port}") for port in [11001, 11002]
    ]
    assert idents == ["11001", "11002"]


def test_get_cpushm_dist_ident_uses_file_store_basename():
    assert _get_cpushm_dist_ident("file:///tmp/vllm_dist_ident") == "vllm_dist_ident"
