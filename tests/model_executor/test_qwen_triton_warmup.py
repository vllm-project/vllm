# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.warmup.qwen_triton_warmup as warmup_module
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _qwen_gdn_warmup_config,
    _warm_batch_memcpy_kernel,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_post_conv_kernel,
    _warm_mrope,
    _warm_vision,
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
    )


def _stub_qwen_warmup_helpers(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._qwen_gdn_warmup_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._warm_gated_rms_norm_kernel",
        lambda device, value, max_num_tokens: calls.append("rmsnorm"),
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
        "vllm.model_executor.warmup.qwen_triton_warmup._warm_batch_memcpy_kernel",
        lambda device: calls.append("memcpy"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup"
        "._warm_fused_sigmoid_gating_delta_rule_update_kernel",
        lambda device, value: calls.append("decode"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._warm_vision",
        lambda model: calls.append("vision"),
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._warm_mrope",
        lambda runner, model: calls.append("mrope"),
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


def test_qwen_triton_warmup_runs_prefill_kernels_for_pooling(monkeypatch) -> None:
    calls: list[str] = []
    _stub_qwen_warmup_helpers(monkeypatch, calls)
    runner = SimpleNamespace(
        is_pooling_model=True,
        device=torch.device("cpu"),
        max_num_tokens=512,
        compilation_config=SimpleNamespace(static_forward_context={}),
        get_model=lambda: torch.nn.Module(),
    )
    qwen_triton_warmup(runner, _model_config())
    assert calls == [
        "rmsnorm",
        "conv",
        "post",
        "memcpy",
        "vision",
        "mrope",
        "sync",
    ]


def test_qwen_triton_warmup_runs_generate_kernels(monkeypatch) -> None:
    calls: list[str] = []
    _stub_qwen_warmup_helpers(monkeypatch, calls)
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cpu"),
        max_num_tokens=512,
        compilation_config=SimpleNamespace(static_forward_context={}),
        get_model=lambda: torch.nn.Module(),
    )
    qwen_triton_warmup(runner, _model_config())
    assert calls == [
        "rmsnorm",
        "conv",
        "post",
        "memcpy",
        "decode",
        "vision",
        "mrope",
        "sync",
    ]


def test_qwen_triton_warmup_skips_non_qwen_model_type(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("non-Qwen model_type must not inspect GDN layers")

    monkeypatch.setattr(
        "vllm.model_executor.warmup.qwen_triton_warmup._qwen_gdn_warmup_config",
        fail,
    )
    runner = SimpleNamespace(is_pooling_model=False, get_model=fail)
    qwen_triton_warmup(runner, _model_config("custom"))


def test_qwen_triton_warmup_runs_vl_helpers_when_gdn_config_missing(
    monkeypatch,
) -> None:
    calls = []
    model = torch.nn.Module()
    runner = SimpleNamespace(
        device=torch.device("cuda"),
        is_pooling_model=False,
        compilation_config=SimpleNamespace(static_forward_context={}),
        get_model=lambda: model,
    )

    monkeypatch.setattr(
        warmup_module, "_qwen_gdn_warmup_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        warmup_module, "_warm_vision", lambda value: calls.append(value)
    )
    monkeypatch.setattr(
        warmup_module,
        "_warm_mrope",
        lambda runner_value, model_value: calls.append((runner_value, model_value)),
    )
    monkeypatch.setattr(
        warmup_module, "_synchronize_device", lambda device: calls.append(device)
    )

    qwen_triton_warmup(runner, _model_config())
    assert calls == [model, (runner, model), runner.device]


def test_warm_gated_rms_norm_uses_production_m_shape(monkeypatch) -> None:
    captured_m: list[int] = []
    captured_kwargs: list[dict[str, object]] = []

    def fake_warmup(*args, **kwargs):
        captured_m.append(args[10])
        captured_kwargs.append(kwargs)

    monkeypatch.setattr(
        "vllm.third_party.flash_linear_attention.ops.layernorm_guard"
        ".calc_rows_per_block",
        lambda M, device: 1,
    )
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        layer_norm_fwd_kernel,
    )

    monkeypatch.setattr(layer_norm_fwd_kernel, "warmup", fake_warmup)
    from vllm.model_executor.warmup.qwen_triton_warmup import (
        _QwenGDNWarmupConfig,
        _warm_gated_rms_norm_kernel,
    )

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
    assert captured_m
    assert all(m % 32 == 0 for m in captured_m)
    assert captured_kwargs[0]["HAS_Z"] is True
    assert captured_kwargs[0]["IS_RMS_NORM"] is True
    assert captured_kwargs[0]["NORM_BEFORE_GATE"] is True
    assert captured_kwargs[0]["ACTIVATION"] == "silu"


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
    from vllm.model_executor.warmup.qwen_triton_warmup import (
        _warm_gated_rms_norm_kernel,
    )

    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(device, config, max_num_tokens=16)
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    _warm_batch_memcpy_kernel(device)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


def test_vision_warmup_calls_only_position_and_rotary_paths() -> None:
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    calls: list[tuple[str, object]] = []

    class FakeAttention:
        num_attention_heads_per_partition = 4
        hidden_size_per_attention_head = 8

        def apply_rotary_emb(self, qk, cos, sin):
            calls.append(("rotary", qk.shape))

    class FakeVisual(Qwen3_VisionTransformer):
        spatial_merge_size = 2

        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.blocks = [SimpleNamespace(attn=FakeAttention())]

        def fast_pos_embed_interpolate(self, grid_thw):
            calls.append(("position", grid_thw[0]))

        def rot_pos_emb(self, grid_thw):
            _, h, w = grid_thw[0]
            shape = (h * w, 4)
            return torch.empty(shape), torch.empty(shape)

        def forward(self, *args, **kwargs):
            raise AssertionError("vision warmup must not run the full tower")

    model = torch.nn.Module()
    model.visual = FakeVisual()
    _warm_vision(model)

    grids = [(1, 16, 16), (1, 16, 2), (1, 2, 16), (1, 2, 2)]
    assert [value for name, value in calls if name == "position"] == [
        list(grid) for grid in grids
    ]
    assert [value for name, value in calls if name == "rotary"] == [
        torch.Size((2, h * w, 4, 8)) for _, h, w in grids
    ]


def test_mrope_warmup_reads_model_config_on_v2_runner() -> None:
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    launched: list[tuple[torch.Size, torch.Size]] = []

    class FakeRope(MRotaryEmbedding):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.head_size = 8
            self.rotary_dim = 8
            self.mrope_section = [2, 3, 3]
            self.mrope_interleaved = False
            self.is_neox_style = True

        def forward(self, positions, query, key):
            launched.append((positions.shape, query.shape))
            return query, key

    model = torch.nn.Module()
    model.rotary_emb = FakeRope()
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            uses_mrope=True,
            get_num_attention_heads=lambda parallel_config: 4,
            get_num_kv_heads=lambda parallel_config: 2,
        ),
        parallel_config=object(),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    _warm_mrope(runner, model)
    assert [shape for shape, _ in launched] == [
        torch.Size((3, 1)),
        torch.Size((3, 2)),
        torch.Size((3, 16)),
    ]


def test_mrope_warmup_skips_when_model_config_disables_mrope() -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("M-RoPE warmup must not run when uses_mrope is false")

    runner = SimpleNamespace(
        model_config=SimpleNamespace(uses_mrope=False),
        get_model=fail,
    )
    _warm_mrope(runner, torch.nn.Module())
