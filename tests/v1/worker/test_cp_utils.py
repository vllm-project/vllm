# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.worker import cp_utils


def test_dcp_unsupported_backend_error_mentions_attention_backend_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedDCPImpl:
        need_to_return_lse_for_decode = False
        supports_mtp_with_cp_non_trivial_interleave_size = True
        supports_pcp = True

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
        lambda *_args, **_kwargs: {
            "layer.0": SimpleNamespace(impl=UnsupportedDCPImpl())
        },
    )

    with pytest.raises(AssertionError) as exc_info:
        cp_utils.check_attention_cp_compatibility(vllm_config)

    message = str(exc_info.value)
    assert "Decode Context Parallelism (DCP)" in message
    assert "--attention-backend" in message
    assert "VLLM_ATTENTION_BACKEND" in message
