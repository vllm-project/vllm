# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import torch


def _install_flash_attn_stub(monkeypatch) -> None:
    flash_attn_stub = ModuleType("vllm.vllm_flash_attn")
    flash_attn_stub.__dict__["flash_attn_varlen_func"] = lambda *args, **kwargs: None
    flash_attn_stub.__dict__["get_scheduler_metadata"] = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", flash_attn_stub)


def _import_qwen_gdn_module(monkeypatch):
    _install_flash_attn_stub(monkeypatch)
    from vllm.model_executor.layers.mamba.gdn import (
        qwen_gdn_linear_attn as qwen_gdn,
    )

    return qwen_gdn


def test_kernel_warmup_runs_hybrid_warmup(monkeypatch) -> None:
    _install_flash_attn_stub(monkeypatch)
    from vllm.model_executor.warmup import kernel_warmup

    calls: list[str] = []

    model = object()

    def fake_hybrid_warmup(*args, **kwargs) -> None:
        assert args == (model,)
        assert kwargs == {"model_dtype": torch.bfloat16}
        calls.append("hybrid")

    def fake_dummy_run(**kwargs) -> None:
        assert kwargs == {
            "num_tokens": 16,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        }
        calls.append("dummy")

    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(kernel_warmup, "has_flashinfer", lambda: False)
    monkeypatch.setattr(kernel_warmup, "qwen_triton_warmup", lambda *args: None)
    monkeypatch.setattr(
        kernel_warmup, "deepseek_v4_mhc_warmup", lambda *args, **kw: None
    )
    monkeypatch.setattr(
        kernel_warmup,
        "sparse_mla_triton_warmup_if_needed",
        lambda *args: None,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "flashinfer_sparse_mla_decode_autotune_warmup",
        lambda *args: None,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "deepseek_v4_sparse_mla_attention_warmup",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.hybrid_gdn_mamba_mrope_warmup"
        ".hybrid_gdn_mamba_mrope_warmup",
        fake_hybrid_warmup,
    )
    monkeypatch.setattr(
        "vllm.model_executor.warmup.minimax_m3_msa_warmup.minimax_m3_msa_warmup",
        lambda worker: None,
    )

    worker = SimpleNamespace(
        get_model=lambda: model,
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False),
            model_config=SimpleNamespace(),
        ),
        model_runner=SimpleNamespace(
            dtype=torch.bfloat16,
            _dummy_run=fake_dummy_run,
            is_pooling_model=True,
            attn_groups=[],
        ),
    )

    kernel_warmup.kernel_warmup(worker)

    assert calls == ["hybrid"]


def test_qwen_gdn_update_warmup_uses_bound_ssm_cache(monkeypatch) -> None:
    qwen_gdn = _import_qwen_gdn_module(monkeypatch)
    captured: dict[str, Any] = {}
    calls: list[str] = []

    def fake_update(**kwargs):
        calls.append("update")
        captured.update(kwargs)
        return torch.empty(0), kwargs["initial_state"]

    monkeypatch.setattr(
        qwen_gdn,
        "fused_sigmoid_gating_delta_rule_update",
        fake_update,
    )

    ssm_state = torch.empty((2, 3, 4, 5), dtype=torch.float32)
    layer = SimpleNamespace(
        kv_cache=(torch.empty(0), ssm_state),
        A_log=torch.empty(3, dtype=torch.float32),
        dt_bias=torch.empty(3, dtype=torch.float32),
        head_k_dim=5,
        head_v_dim=4,
        _continuous_batching_update_kernel_warmed_up=False,
    )

    qwen_gdn.QwenGatedDeltaNetAttention._warmup_continuous_batching_update_kernel(
        layer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_k_heads=2,
        num_v_heads=3,
    )

    assert captured["initial_state"] is ssm_state
    assert captured["A_log"] is layer.A_log
    assert captured["dt_bias"] is layer.dt_bias
    assert captured["inplace_final_state"] is True
    assert captured["use_qk_l2norm_in_kernel"] is True
    assert captured["q"].shape == (1, 1, 2, 5)
    assert captured["k"].shape == (1, 1, 2, 5)
    assert captured["v"].shape == (1, 1, 3, 4)
    assert captured["a"].shape == (1, 3)
    assert captured["b"].shape == (1, 3)
    assert captured["cu_seqlens"].tolist() == [0, 1]
    assert captured["ssm_state_indices"].tolist() == [0]
    assert layer._continuous_batching_update_kernel_warmed_up is True

    qwen_gdn.QwenGatedDeltaNetAttention._warmup_continuous_batching_update_kernel(
        layer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_k_heads=2,
        num_v_heads=3,
    )

    assert calls == ["update"]


def test_qwen_gdn_update_warmup_skips_without_bound_ssm_cache(
    monkeypatch,
) -> None:
    qwen_gdn = _import_qwen_gdn_module(monkeypatch)
    calls: list[str] = []

    def fake_update(**kwargs):
        calls.append("update")
        return torch.empty(0), kwargs["initial_state"]

    monkeypatch.setattr(
        qwen_gdn,
        "fused_sigmoid_gating_delta_rule_update",
        fake_update,
    )

    layer = SimpleNamespace(
        kv_cache=(torch.empty(0), torch.empty(0)),
        A_log=torch.empty(3, dtype=torch.float32),
        dt_bias=torch.empty(3, dtype=torch.float32),
        head_k_dim=5,
        head_v_dim=4,
        _continuous_batching_update_kernel_warmed_up=False,
    )

    qwen_gdn.QwenGatedDeltaNetAttention._warmup_continuous_batching_update_kernel(
        layer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_k_heads=2,
        num_v_heads=3,
    )

    layer.kv_cache = None
    qwen_gdn.QwenGatedDeltaNetAttention._warmup_continuous_batching_update_kernel(
        layer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_k_heads=2,
        num_v_heads=3,
    )

    assert calls == []
    assert layer._continuous_batching_update_kernel_warmed_up is False

    layer.kv_cache = (torch.empty(0), torch.empty((2, 3, 4, 5), dtype=torch.float32))
    qwen_gdn.QwenGatedDeltaNetAttention._warmup_continuous_batching_update_kernel(
        layer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_k_heads=2,
        num_v_heads=3,
    )

    assert calls == ["update"]
    assert layer._continuous_batching_update_kernel_warmed_up is True
