# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.layers.mamba.ops import ssu_dispatch
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FLASHINFER_REPLAYSSM_AUTO_TACTIC,
    FlashInferReplaySSMBackend,
    FlashInferReplaySSMTactic,
    use_flashinfer_replayssm_tactic,
)
from vllm.model_executor.warmup import flashinfer_replayssm_warmup as warmup


def _fake_backend() -> FlashInferReplaySSMBackend:
    backend = FlashInferReplaySSMBackend.__new__(FlashInferReplaySSMBackend)
    backend._mamba_config = MambaConfig(backend=MambaBackendEnum.FLASHINFER)
    backend._kernel = Mock(return_value=torch.empty(1))
    backend._tactic = FLASHINFER_REPLAYSSM_AUTO_TACTIC
    return backend


def _fake_layer(cache_slots: int = 5):
    nheads, headdim, dstate, ngroups = 2, 4, 8, 1
    return SimpleNamespace(
        kv_cache=(
            torch.empty(0),
            torch.empty(cache_slots, nheads, headdim, dstate),
            torch.empty(cache_slots, nheads, 17, headdim),
            torch.empty(cache_slots, nheads, 17),
            torch.empty(cache_slots, ngroups, 17, dstate),
        ),
        A=torch.empty(nheads),
        D=torch.empty(nheads),
        dt_bias=torch.empty(nheads),
        mamba_config=SimpleNamespace(
            enable_stochastic_rounding=False,
            stochastic_rounding_philox_rounds=None,
        ),
    )


def test_replayssm_explicit_tactic_validation():
    tactic = FlashInferReplaySSMTactic("two-kernel", 1, 4, precompute_heads_per_cta=8)
    assert tactic.name == "two_kernel_s1_c4_h8"
    with pytest.raises(ValueError, match="does not accept precompute"):
        FlashInferReplaySSMTactic("monolith", precompute_heads_per_cta=8)


def test_replayssm_tactic_scope_restores_direct_launch_controls(monkeypatch):
    backend = _fake_backend()
    monkeypatch.setattr(ssu_dispatch, "_replayssm_backend", backend)
    tactic = FlashInferReplaySSMTactic("two-kernel", 2, 16, precompute_heads_per_cta=8)

    with (
        pytest.raises(RuntimeError, match="sentinel"),
        use_flashinfer_replayssm_tactic(tactic),
    ):
        assert backend._tactic is tactic
        raise RuntimeError("sentinel")

    assert backend._tactic is FLASHINFER_REPLAYSSM_AUTO_TACTIC


def test_replayssm_backend_forwards_explicit_tactic(monkeypatch):
    backend = _fake_backend()
    monkeypatch.setattr(ssu_dispatch, "_replayssm_backend", backend)
    tensor = torch.empty(1)

    with use_flashinfer_replayssm_tactic(
        FlashInferReplaySSMTactic("two-kernel", 1, 4, precompute_heads_per_cta=8)
    ):
        backend(
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
        )

    kwargs = backend._kernel.call_args.kwargs
    assert kwargs["algorithm"] == "two-kernel"
    assert kwargs["precompute_heads_per_cta"] == 8
    assert kwargs["main_pipeline_stages"] == 1
    assert kwargs["main_ctas_per_sm"] == 4


def test_replayssm_tuning_call_uses_private_production_layout():
    layer = _fake_layer()
    call = warmup._ReplaySSMTuningCall(layer, 3)

    assert call.state.shape == (4, *layer.kv_cache[1].shape[1:])
    assert call.state.stride() == layer.kv_cache[1].stride()
    assert call.state.data_ptr() != layer.kv_cache[1].data_ptr()
    assert call.x_cache.data_ptr() != layer.kv_cache[2].data_ptr()
    assert call.indices.tolist() == [1, 2, 3]
    assert call.ring_start.shape == (4,)
    assert call.prev_num_accepted.tolist() == [0, 1, 2, 3]
    assert call.dt.shape == (3, 2, 4)
    assert call.dt.stride(-1) == 0


def test_replayssm_tuning_trigger_uses_largest_supported_decode_batch(monkeypatch):
    layer = _fake_layer(cache_slots=5)
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        cudagraph_dispatcher=SimpleNamespace(
            get_capture_descs=lambda: [
                (
                    None,
                    [
                        BatchDescriptor(num_tokens=2, num_reqs=2, uniform=True),
                        BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True),
                    ],
                )
            ]
        ),
    )
    observed = SimpleNamespace(batch=None, ran=False)

    class FakeCall:
        def __init__(self, _layer, batch):
            assert _layer is layer
            observed.batch = batch

        def run(self):
            observed.ran = True

    monkeypatch.setattr(warmup, "_find_replayssm_layers", lambda _runner: (layer,))
    monkeypatch.setattr(
        warmup, "_distributed_layers_are_compatible", lambda _layers: True
    )
    monkeypatch.setattr(warmup, "_distributed_min", lambda value: value)
    monkeypatch.setattr(warmup, "_ReplaySSMTuningCall", FakeCall)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    warmup.trigger_flashinfer_replayssm_autotune(runner)

    assert observed.batch == 4
    assert observed.ran
