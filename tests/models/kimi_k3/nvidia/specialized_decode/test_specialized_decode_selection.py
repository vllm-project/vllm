# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which decode a Kimi K3 deployment gets from `decode_backend` (CPU only): the
default `auto` runs the specialized decode where `unsupported` finds no reason
and falls back to the stock decode where it does, `stock` never runs it, and
`latency_specialized` refuses an unsupported deployment."""

from types import SimpleNamespace

import pytest

from vllm.models.kimi_k3.nvidia.specialized_decode import runtime


def _config(**extra):
    return SimpleNamespace(
        additional_config=extra,
        speculative_config=None,
        model_config=SimpleNamespace(
            max_model_len=4096, hf_text_config=SimpleNamespace()
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4]),
    )


@pytest.fixture
def support(monkeypatch):
    """Sets what `unsupported` reports; records whether it was consulted."""
    state = SimpleNamespace(reason=None, calls=0)

    def unsupported(vllm_config):
        state.calls += 1
        return state.reason

    monkeypatch.setattr(runtime, "unsupported", unsupported)
    monkeypatch.setattr(runtime, "dsa_fi_kernels", lambda vllm_config: ())
    monkeypatch.setattr(runtime, "prims_ts_rows", lambda vllm_config: ())
    return state


@pytest.mark.parametrize("extra", [{}, {"decode_backend": "auto"}])
def test_auto_runs_the_specialized_decode_where_supported(support, extra):
    config = runtime.specialized_decode_config(_config(**extra))
    assert isinstance(config, runtime.DecodeConfig)
    assert config.capture_sizes == (1, 2, 4)


def test_auto_falls_back_to_stock_where_unsupported(support):
    support.reason = "this GPU (SM100/103)"
    assert runtime.specialized_decode_config(_config()) is None


def test_stock_never_builds_it(support):
    assert runtime.specialized_decode_config(_config(decode_backend="stock")) is None
    assert support.calls == 0


def test_latency_specialized_refuses_an_unsupported_deployment(support):
    support.reason = "this GPU (SM100/103)"
    with pytest.raises(ValueError, match="does not support this GPU"):
        runtime.specialized_decode_config(_config(decode_backend="latency_specialized"))


def test_unknown_backend_is_rejected(support):
    with pytest.raises(ValueError, match="auto/stock/latency_specialized"):
        runtime.specialized_decode_config(_config(decode_backend="fast"))
