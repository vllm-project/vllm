# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.worker import cp_utils


class _DummyImpl:
    need_to_return_lse_for_decode = False
    supports_pcp = True
    supports_mtp_with_cp_non_trivial_interleave_size = True


class _DummyLayer:
    impl = _DummyImpl()


def test_dcp_error_includes_attention_backend_hint(monkeypatch: pytest.MonkeyPatch):
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
    )
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *_: {"layer.0": _DummyLayer()},
    )

    with pytest.raises(AssertionError, match="VLLM_ATTENTION_BACKEND") as exc:
        cp_utils.check_attention_cp_compatibility(vllm_config)
    assert "_DummyImpl" in str(exc.value)
