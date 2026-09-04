# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _qwen_gdn_warmup_config,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
    qwen_triton_warmup,
)


def _gdn_layer(
    *,
    kv_cache=(),
    include_state_api: bool = True,
    num_k_heads: int = 4,
    num_v_heads: int = 4,
    head_k_dim: int = 16,
    head_v_dim: int = 16,
    conv_kernel_size: int = 4,
    tp_size: int = 1,
) -> SimpleNamespace:
    conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    layer = SimpleNamespace(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        tp_size=tp_size,
        kv_cache=kv_cache,
        A_log=torch.zeros(num_v_heads, dtype=torch.float32),
        dt_bias=torch.zeros(num_v_heads, dtype=torch.float32),
        norm=SimpleNamespace(
            weight=torch.zeros(head_v_dim, dtype=torch.bfloat16),
            norm_before_gate=True,
            activation="silu",
        ),
    )
    if include_state_api:
        conv_shape = (conv_dim, conv_kernel_size - 1)
        ssm_shape = (num_v_heads, head_v_dim, head_k_dim)
        layer.get_state_shape = lambda: (conv_shape, ssm_shape)
        layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    return layer


def _model_config(model_type: str = "qwen3_5") -> SimpleNamespace:
    return SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type=model_type),
        hf_config=SimpleNamespace(model_type=model_type),
        dtype=torch.bfloat16,
    )


def _stub_qwen_warmup_helpers(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._qwen_gdn_warmup_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._warm_gated_rms_norm_kernel",
        lambda device, value, max_num_tokens, x_dtype=None: calls.append("rmsnorm"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._warm_causal_conv1d_fwd_kernel",
        lambda device, value: calls.append("conv"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._warm_fused_post_conv_kernel",
        lambda device, value: calls.append("post"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup"
        "._warm_fused_sigmoid_gating_delta_rule_update_kernel",
        lambda device, value: calls.append("decode"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._synchronize_device",
        lambda device: calls.append("sync"),
    )


def test_qwen_gdn_warmup_config_keeps_bound_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first",
        lambda: True,
    )
    conv = torch.empty(8, 128, 3)
    ssm = torch.empty(8, 4, 16, 16)
    layer = _gdn_layer(kv_cache=(conv, ssm), include_state_api=False)
    config = _qwen_gdn_warmup_config({"layer": layer})
    assert config is not None
    assert config.conv_state.shape[0] == 8
    assert config.hv == 4
    assert config.norm_before_gate is True
    assert config.norm_activation == "silu"


def test_qwen_gdn_warmup_config_norm_is_optional(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first",
        lambda: True,
    )
    conv = torch.empty(8, 128, 3)
    ssm = torch.empty(8, 4, 16, 16)
    layer = _gdn_layer(kv_cache=(conv, ssm), include_state_api=False)
    del layer.norm
    config = _qwen_gdn_warmup_config({"layer": layer})
    assert config is not None
    assert config.norm_weight_dtype is None
    assert config.conv_state.shape[0] == 8


def test_qwen_triton_warmup_runs_prefill_kernels_for_pooling(monkeypatch) -> None:
    calls: list[str] = []
    _stub_qwen_warmup_helpers(monkeypatch, calls)
    runner = SimpleNamespace(
        is_pooling_model=True,
        device=torch.device("cpu"),
        max_num_tokens=512,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    qwen_triton_warmup(runner, _model_config())
    assert calls == ["rmsnorm", "conv", "post", "sync"]


def test_qwen_triton_warmup_runs_generate_kernels(monkeypatch) -> None:
    calls: list[str] = []
    _stub_qwen_warmup_helpers(monkeypatch, calls)
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cpu"),
        max_num_tokens=512,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    qwen_triton_warmup(runner, _model_config())
    assert calls == ["rmsnorm", "conv", "post", "decode", "sync"]


def test_qwen_triton_warmup_skips_non_qwen_model_type(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("non-Qwen model_type must not inspect GDN layers")

    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._qwen_gdn_warmup_config",
        fail,
    )
    runner = SimpleNamespace(is_pooling_model=False)
    qwen_triton_warmup(runner, _model_config("custom"))


def test_qwen_triton_warmup_skips_when_gdn_config_missing(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("GDN kernels must not run without a GDN config")

    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._qwen_gdn_warmup_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._warm_gated_rms_norm_kernel",
        fail,
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cpu"),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    qwen_triton_warmup(runner, _model_config())


def test_warm_gated_rms_norm_uses_production_shape(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_warmup(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(
        "vllm.third_party.flash_linear_attention.ops.layernorm_guard"
        ".warmup_layer_norm_fwd",
        fake_warmup,
    )
    from vllm.model_executor.warmup.qwen_triton_warmup import _QwenGDNWarmupConfig

    config = _QwenGDNWarmupConfig(
        h=2,
        hv=32,
        k=16,
        v=128,
        conv_kernel_size=4,
        conv_state=torch.empty(1),
        conv_dtype=torch.bfloat16,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.empty(1),
        dt_bias=torch.empty(1),
        state_stride_token=1,
        state_dtype=torch.float32,
    )
    _warm_gated_rms_norm_kernel(torch.device("cpu"), config, max_num_tokens=512)
    assert captured
    assert captured[0]["rows_per_token"] == 32
    assert captured[0]["group_size"] == 128
    assert captured[0]["is_rms_norm"] is True
    assert captured[0]["norm_before_gate"] is True
    assert captured[0]["activation"] == "silu"


def test_warm_gated_rms_norm_skips_when_norm_missing(monkeypatch) -> None:
    called: list[int] = []

    def fail(*_args, **_kwargs):
        called.append(1)

    from vllm.model_executor.warmup.qwen_triton_warmup import _QwenGDNWarmupConfig

    monkeypatch.setattr(
        "vllm.third_party.flash_linear_attention.ops.layernorm_guard"
        ".warmup_layer_norm_fwd",
        fail,
    )
    config = _QwenGDNWarmupConfig(
        h=2,
        hv=32,
        k=16,
        v=128,
        conv_kernel_size=4,
        conv_state=torch.empty(1),
        conv_dtype=torch.bfloat16,
        norm_weight_dtype=None,
        norm_before_gate=False,
        norm_activation="",
        a_log=torch.empty(1),
        dt_bias=torch.empty(1),
        state_stride_token=1,
        state_dtype=torch.float32,
    )
    _warm_gated_rms_norm_kernel(torch.device("cpu"), config, max_num_tokens=512)
    assert called == []


def _cuda_gdn_config() -> object:
    from vllm.model_executor.warmup.qwen_triton_warmup import _QwenGDNWarmupConfig

    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(device, config, max_num_tokens=16)
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)
