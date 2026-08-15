# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.platforms.interface import Platform

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("disable_hybrid_manager", "expected_align_calls"),
    [(False, 1), (True, 0)],
)
def test_hybrid_block_alignment_respects_manager_mode(
    monkeypatch, disable_hybrid_manager, expected_align_calls
):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=64,
            user_specified_block_size=True,
            kv_cache_dtype_skip_layers=None,
        ),
        model_config=SimpleNamespace(is_hybrid=True),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid_manager
        ),
    )
    align_calls = []

    monkeypatch.setattr(
        Platform,
        "_find_non_ssm_backend",
        classmethod(lambda cls, vllm_config: object()),
    )
    monkeypatch.setattr(
        Platform,
        "_align_hybrid_block_size",
        classmethod(
            lambda cls, vllm_config, backend_cls: align_calls.append(vllm_config)
        ),
    )

    Platform.update_block_size_for_backend(config)

    assert len(align_calls) == expected_align_calls
