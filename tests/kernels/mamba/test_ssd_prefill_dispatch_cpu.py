# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mamba.mamba_mixer2 as mixer2
import vllm.model_executor.layers.mamba.ops.ssd_prefill_dispatch as dispatch
from vllm.config.mamba import MambaPrefillBackendEnum
from vllm.model_executor.layers.mamba.ops.ssd_prefill_dispatch import (
    FlashInferMamba2PrefillRequest,
    Mamba2PrefillFallbackReason,
    get_mamba2_prefill_dispatch_stats,
    has_supported_mamba_state_cache_layout,
    reset_mamba2_prefill_dispatch_stats,
    run_flashinfer_mamba2_prefill,
    validate_flashinfer_mamba2_prefill,
)


def test_mamba_state_cache_layout_accepts_unified_page_stride():
    # Target logical row is H64/D64/N128 = 524,288 fp16 elements. The
    # Nano35 unified page row stride is 589,824 elements.
    raw = torch.empty(2 * 589824, dtype=torch.float16)
    state = torch.as_strided(
        raw,
        size=(2, 64, 64, 128),
        stride=(589824, 8192, 128, 1),
    )
    assert not state.is_contiguous()
    assert has_supported_mamba_state_cache_layout(state)


def test_mamba_state_cache_layout_rejects_noncompact_inner_axes():
    raw = torch.empty(2 * 600000, dtype=torch.float16)
    state = torch.as_strided(
        raw,
        size=(2, 64, 64, 128),
        stride=(600000, 9000, 128, 1),
    )
    assert not has_supported_mamba_state_cache_layout(state)


def test_mamba_state_cache_layout_rejects_misaligned_base_pointer():
    raw = torch.empty(2 * 589824 + 1, dtype=torch.float16)
    state = torch.as_strided(
        raw,
        size=(2, 64, 64, 128),
        stride=(589824, 8192, 128, 1),
        storage_offset=1,
    )
    assert state.data_ptr() % 16 != 0
    assert not has_supported_mamba_state_cache_layout(state)


def test_mamba_state_cache_layout_rejects_misaligned_row_stride():
    row_stride = 589825
    raw = torch.empty(row_stride + 64 * 64 * 128, dtype=torch.float16)
    state = torch.as_strided(
        raw,
        size=(2, 64, 64, 128),
        stride=(row_stride, 8192, 128, 1),
    )
    assert state.data_ptr() % 16 == 0
    assert state.stride(0) * state.element_size() % 16 != 0
    assert not has_supported_mamba_state_cache_layout(state)


def _cpu_request_with_missing_metadata() -> FlashInferMamba2PrefillRequest:
    tokens = 128
    x = torch.empty(tokens, 64, 64, dtype=torch.bfloat16)
    return FlashInferMamba2PrefillRequest(
        x=x,
        dt=torch.empty(tokens, 64, dtype=torch.bfloat16),
        A=torch.empty(64, dtype=torch.float32),
        B=torch.empty(tokens, 8, 128, dtype=torch.bfloat16),
        C=torch.empty(tokens, 8, 128, dtype=torch.bfloat16),
        D=torch.empty(64, dtype=torch.bfloat16),
        dt_bias=torch.empty(64, dtype=torch.float32),
        out=torch.empty_like(x),
        state_cache=torch.empty(1, 64, 64, 128, dtype=torch.float16),
        initial_states=None,
        seq_idx=None,
        token_dst_indices=None,
        chunk_indices=None,
        chunk_offsets=None,
        seq_chunk_cumsum=None,
        intermediate_state_indices=None,
        chunk_size=128,
        num_seqs=1,
        valid_seqlen=tokens,
        padded_seqlen=tokens,
    )


def test_missing_metadata_falls_back_with_layer_token_counter():
    reset_mamba2_prefill_dispatch_stats()
    reason = run_flashinfer_mamba2_prefill(_cpu_request_with_missing_metadata())
    assert reason == Mamba2PrefillFallbackReason.MISSING_METADATA

    stats = get_mamba2_prefill_dispatch_stats()
    assert stats.fallback_layer_invocations == {"missing_metadata": 1}
    assert stats.fallback_layer_tokens == {"missing_metadata": 128}
    assert stats.flashinfer_layer_invocations == 0
    assert stats.flashinfer_layer_tokens == 0


def test_serving_ssd_requests_fp32_processed_dt(monkeypatch):
    captured = {}

    class FakeSSD:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(dispatch, "_load_flashinfer_ssd_class", lambda: FakeSSD)
    dispatch._SSD_OBJECTS.clear()
    dispatch._get_ssd(_cpu_request_with_missing_metadata())

    assert captured["dt_processed_dtype"] == torch.float32
    assert "cumsum_block_size_h" not in captured
    assert captured["has_intermediate_states"] is True
    assert "intermediate_state_indices" in (
        FlashInferMamba2PrefillRequest.__dataclass_fields__
    )


def _static_config(*, prefill_backend, speculative_tokens=0):
    return SimpleNamespace(
        mamba_config=SimpleNamespace(prefill_backend=prefill_backend),
        num_speculative_tokens=speculative_tokens,
    )


def test_default_triton_does_not_probe_flashinfer(monkeypatch):
    def unexpected_probe():
        raise AssertionError("default Triton path must not import FlashInfer")

    monkeypatch.setattr(dispatch, "_load_flashinfer_ssd_class", unexpected_probe)
    validate_flashinfer_mamba2_prefill(
        vllm_config=_static_config(
            prefill_backend=MambaPrefillBackendEnum.TRITON
        ),
        model_config=None,
        cache_config=None,
        tp_size=2,
        nheads=1,
        headdim=1,
        dstate=1,
        ngroups=1,
        state_dtype=torch.float32,
        layer_name="test",
    )


def test_explicit_flashinfer_static_gate_aggregates_failures(monkeypatch):
    monkeypatch.setattr(
        dispatch,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda: (12, 0),
        ),
    )
    monkeypatch.setattr(dispatch, "_load_flashinfer_ssd_class", lambda: object)
    model_config = SimpleNamespace(
        dtype=torch.float16, get_mamba_chunk_size=lambda: 64
    )
    cache_config = SimpleNamespace(
        mamba_cache_mode="align", mamba_block_size=1000
    )

    with pytest.raises(ValueError) as exc_info:
        validate_flashinfer_mamba2_prefill(
            vllm_config=_static_config(
                prefill_backend=MambaPrefillBackendEnum.FLASHINFER,
                speculative_tokens=3,
            ),
            model_config=model_config,
            cache_config=cache_config,
            tp_size=2,
            nheads=32,
            headdim=128,
            dstate=64,
            ngroups=4,
            state_dtype=torch.bfloat16,
            layer_name="bad.layer",
        )

    message = str(exc_info.value)
    assert "SM100/SM103/SM110" in message
    assert "tensor parallel size must be 1" in message
    assert "model I/O dtype must be bfloat16" in message
    assert "SSM cache dtype must be float16" in message
    assert "Mamba chunk size must be 128" in message
    assert "H64/D64/N128/G8" in message
    assert "Mamba cache mode must be 'all'" in message
    assert "Mamba cache block size must be divisible by 128" in message
    assert "speculative decoding is not supported" in message


def test_explicit_flashinfer_static_gate_accepts_resolved_target(monkeypatch):
    monkeypatch.setattr(
        dispatch,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda: (10, 0),
        ),
    )
    monkeypatch.setattr(dispatch, "_load_flashinfer_ssd_class", lambda: object)
    model_config = SimpleNamespace(
        dtype=torch.bfloat16, get_mamba_chunk_size=lambda: 128
    )
    cache_config = SimpleNamespace(
        mamba_cache_mode="all", mamba_block_size=1152
    )

    validate_flashinfer_mamba2_prefill(
        vllm_config=_static_config(
            prefill_backend=MambaPrefillBackendEnum.FLASHINFER
        ),
        model_config=model_config,
        cache_config=cache_config,
        tp_size=1,
        nheads=64,
        headdim=64,
        dstate=128,
        ngroups=8,
        state_dtype=torch.float16,
        layer_name="good.layer",
    )


def test_mixer_validation_observes_post_load_block_resolution(monkeypatch):
    observed_block_sizes = []
    vllm_config = _static_config(
        prefill_backend=MambaPrefillBackendEnum.FLASHINFER
    )
    monkeypatch.setattr(
        mixer2,
        "get_current_vllm_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("warmup must use the retained config")
        ),
    )

    def record_validation(**kwargs):
        observed_block_sizes.append(kwargs["cache_config"].mamba_block_size)

    monkeypatch.setattr(
        mixer2, "validate_flashinfer_mamba2_prefill", record_validation
    )
    cache_config = SimpleNamespace(mamba_block_size=16)
    layer = SimpleNamespace(
        _use_flashinfer_prefill=True,
        _flashinfer_prefill_validated=False,
        _vllm_config=vllm_config,
        model_config=SimpleNamespace(),
        cache_config=cache_config,
        tp_size=1,
        num_heads=64,
        head_dim=64,
        ssm_state_size=128,
        n_groups=8,
        prefix="test.layer",
        get_state_dtype=lambda: (torch.bfloat16, torch.float16),
    )

    # Model construction sees the provisional default but must not validate.
    assert observed_block_sizes == []
    cache_config.mamba_block_size = 1152

    mixer2.MambaMixer2._ensure_flashinfer_prefill_validated(layer)
    mixer2.MambaMixer2._ensure_flashinfer_prefill_validated(layer)

    assert observed_block_sizes == [1152]
    assert layer._flashinfer_prefill_validated
