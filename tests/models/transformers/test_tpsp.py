# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPSP interface contracts, using local models and a simulated P2P transport."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from vllm.distributed import parallel_state
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.transformers.fuser import get_fuser
from vllm.model_executor.models.transformers.fusers import RMSNormFuser
from vllm.model_executor.models.transformers.tpsp_dense_adapter import (
    TransformersDenseTPSP,
)
from vllm.model_executor.models.transformers.tpsp_runtime import (
    TPSPOpsGroup,
    UnsupportedTPSPModelError,
)
from vllm.platforms import current_platform, tpsp_cuda


@pytest.fixture
def transport(monkeypatch):
    """Replace only CUDA/collective boundaries; run the real backend lifecycle."""
    device_group = SimpleNamespace(group_name="tp")
    control_group = object()
    group = SimpleNamespace(
        world_size=2, device_group=device_group, cpu_group=control_group
    )
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        tpsp_cuda.c10d, "_resolve_process_group", lambda _: device_group
    )
    monkeypatch.setattr(tpsp_cuda.dist, "get_world_size", lambda _: 2)
    monkeypatch.setattr(tpsp_cuda.dist, "get_rank", lambda _: 0)
    monkeypatch.setattr(tpsp_cuda.dist, "all_reduce", Mock())
    monkeypatch.setattr(tpsp_cuda.dist, "barrier", Mock())
    monkeypatch.setattr(
        tpsp_cuda.dist,
        "all_gather",
        lambda outputs, value, **_: [output.copy_(value) for output in outputs],
    )
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    live_handles = set()

    def allocate(*args):
        live_handles.add(1)
        return 1

    native = SimpleNamespace(
        init_tpsp_p2p=Mock(side_effect=allocate),
        get_tpsp_p2p_handles=Mock(return_value=torch.zeros(128, dtype=torch.uint8)),
        map_tpsp_p2p=Mock(return_value=True),
        destroy_tpsp_p2p=Mock(side_effect=live_handles.remove),
        tpsp_matmul_reduce_scatter_norm_all_gather=Mock(),
    )
    for name, operation in vars(native).items():
        monkeypatch.setattr(torch.ops._C, name, operation, raising=False)
    return SimpleNamespace(native=native, live_handles=live_handles)


@pytest.fixture
def backend(transport):
    backend = tpsp_cuda.CudaTPSPBackend("tp", torch.device("cuda:0"))
    yield backend
    backend.close()
    assert not transport.live_handles


@pytest.fixture
def open_options():
    return dict(
        dtype=torch.bfloat16,
        tp_size=2,
        hidden_size=16,
        max_batched_tokens=1024,
        group_name="tp",
        device=torch.device("cuda:0"),
    )


@pytest.fixture
def ops_groups(default_vllm_config):
    with torch.device("meta"):
        return [TPSPOpsGroup("proj", nn.Linear(8, 16), RMSNorm(16))]


@pytest.fixture
def adapter_config():
    return SimpleNamespace(
        quant_config=None,
        lora_config=None,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            tpsp_microchunk_tokens=None,
            tpsp_threshold_tokens=None,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        device_config=SimpleNamespace(device="cuda:0"),
    )


@pytest.fixture
def model_factory(default_vllm_config):
    def create(model_type="llama"):
        config = AutoConfig.for_model(
            model_type,
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            bos_token_id=1,
            eos_token_id=2,
        )
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
            # Adapter installation follows the regular backend's norm fusion.
            for name, module in list(model.named_modules()):
                if "norm" not in type(module).__name__.lower():
                    continue
                fuser = get_fuser(module, RMSNormFuser)
                if fuser is not None:
                    model.set_submodule(
                        name, fuser.fuse(module, name, default_vllm_config)
                    )
        return model

    return create


@pytest.fixture
def adapter_backend(monkeypatch, backend):
    monkeypatch.setattr(
        current_platform, "get_tpsp_backend_cls", lambda: lambda *args: backend
    )
    return backend


# ============================================================================
# Parameter validation
# ============================================================================


@pytest.mark.parametrize(
    "microchunk,threshold,expected_chunk,expected_threshold",
    [(None, None, 128, 512), (8, 16, 4, 16), (2048, 2048, 512, 1024)],
    ids=["defaults", "overrides", "capped"],
)
def test_profile_accepts_token_settings(
    backend,
    open_options,
    ops_groups,
    microchunk,
    threshold,
    expected_chunk,
    expected_threshold,
):
    assert backend.open(**open_options)
    context = backend.profile(
        ops_groups, 1024, microchunk_tokens=microchunk, threshold_tokens=threshold
    )
    assert context is not None
    assert context.profile.enabled
    assert context.profile.threshold_tokens == expected_threshold
    assert context.handles["proj"].config.microchunk_tokens_per_rank == expected_chunk


@pytest.mark.parametrize(
    "settings,error",
    [
        ({"microchunk_tokens": 0}, "microchunk_tokens"),
        ({"threshold_tokens": -1}, "threshold_tokens"),
    ],
)
def test_profile_rejects_invalid_token_settings(
    backend, open_options, ops_groups, settings, error
):
    assert backend.open(**open_options)
    with pytest.raises(ValueError, match=error):
        backend.profile(ops_groups, 1024, **settings)


# ============================================================================
# TPSPBackend compatibility
# ============================================================================


@pytest.mark.parametrize(
    "overrides,supported",
    [({}, True), ({"dtype": torch.float16}, False), ({"tp_size": 1}, False)],
    ids=["bf16-tp2", "unsupported-dtype", "unsupported-tp"],
)
def test_backend_accepts_only_supported_configurations(
    backend, open_options, overrides, supported
):
    assert backend.open(**(open_options | overrides)) is supported


def test_backend_reports_unsupported_norm(backend, open_options, caplog_vllm):
    assert backend.open(**open_options)
    with torch.device("meta"):
        groups = [TPSPOpsGroup("proj", nn.Linear(8, 16), nn.LayerNorm(16))]
    assert backend.profile(groups, 1024) is None
    assert "unsupported norm types" in caplog_vllm.text


# ============================================================================
# Model compatibility
# ============================================================================


@pytest.mark.parametrize(
    "model_type,supported",
    [("llama", True), ("qwen3", True), ("gemma", False), ("gpt2", False)],
)
def test_adapter_accepts_only_supported_models(
    model_factory, adapter_config, adapter_backend, model_type, supported
):
    model = model_factory(model_type)
    if supported:
        adapter = TransformersDenseTPSP(model, adapter_config)
        adapter.profile()
        assert adapter.context is not None and adapter.context.profile.enabled
        assert set(adapter.context.handles) == {"o_proj", "down_proj"}
    else:
        forwards = {name: module.forward for name, module in model.named_modules()}
        with pytest.raises(UnsupportedTPSPModelError):
            TransformersDenseTPSP(model, adapter_config)
        assert all(
            module.forward == forwards[name] for name, module in model.named_modules()
        )


# ============================================================================
# Fallback and resource cleanup
# ============================================================================


@pytest.mark.parametrize("failure", ["peer-allocation", "mapping"])
def test_open_fallback_releases_p2p_resources(
    monkeypatch, transport, adapter_backend, model_factory, adapter_config, failure
):
    if failure == "peer-allocation":
        monkeypatch.setattr(
            tpsp_cuda.dist, "all_reduce", lambda ready, **_: ready.zero_()
        )
    else:
        transport.native.map_tpsp_p2p.return_value = False
    adapter = TransformersDenseTPSP(model_factory(), adapter_config)
    assert adapter.backend is None
    assert not transport.live_handles
    transport.native.destroy_tpsp_p2p.assert_called_once_with(1)
    adapter_backend.close()
    transport.native.destroy_tpsp_p2p.assert_called_once_with(1)


@pytest.mark.parametrize("failure", ["unsupported-norm", "configuration-error"])
def test_profile_failure_releases_contexts_and_p2p_resources(
    monkeypatch,
    transport,
    adapter_backend,
    model_factory,
    adapter_config,
    caplog_vllm,
    failure,
):
    adapter = TransformersDenseTPSP(model_factory(), adapter_config)
    configured = []
    set_config = adapter_backend.set_config

    def configure(handle, config):
        configured.append(handle)
        if len(configured) == 2:
            raise RuntimeError("configuration failed")
        set_config(handle, config)

    if failure == "unsupported-norm":
        monkeypatch.setattr(adapter_backend, "supported_norm_types", frozenset())
        adapter.profile()
        assert "status=fallback" in caplog_vllm.text
    else:
        monkeypatch.setattr(adapter_backend, "set_config", configure)
        with pytest.raises(RuntimeError, match="configuration failed"):
            adapter.profile()
        assert "status=failed" in caplog_vllm.text
    assert adapter.context is None and adapter.backend is None
    assert not transport.live_handles
    transport.native.destroy_tpsp_p2p.assert_called_once_with(1)
    for handle in configured:
        with pytest.raises(RuntimeError, match="closed"):
            set_config(handle, 8)
    adapter_backend.close()
    transport.native.destroy_tpsp_p2p.assert_called_once_with(1)
