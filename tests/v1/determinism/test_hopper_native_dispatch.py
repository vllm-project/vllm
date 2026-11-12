# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.compiler_interface import set_inductor_config
from vllm.config.compilation import PassConfig
from vllm.config.utils import Range
from vllm.config.vllm import enable_allreduce_rms_fusion
from vllm.model_executor.determinism import batch_invariant_configs as bi_configs
from vllm.model_executor.layers import linear as linear_mod
from vllm.model_executor.layers import vocab_parallel_embedding as embed_mod
from vllm.model_executor.layers.fused_moe import fused_moe


def test_use_hopper_native_linear_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bi_configs, "supports_hopper_native_dispatch", lambda: True)

    weight = torch.empty((128, 64), dtype=torch.bfloat16)
    x = torch.empty((4, 64), dtype=torch.bfloat16)
    assert bi_configs.use_hopper_native_linear(x, weight, None)

    # Non-contiguous activations stay eligible so F.linear(x.contiguous(), w) keeps
    # the layer on one kernel family instead of switching between cuBLAS and Triton.
    non_contig_x = torch.empty((8, 64), dtype=torch.bfloat16)[::2]
    assert not non_contig_x.is_contiguous()
    assert bi_configs.use_hopper_native_linear(non_contig_x, weight, None)

    # Narrow projections (e.g., Mixtral router N=8) keep batch-invariant Triton matmul.
    narrow_weight = torch.empty((8, 64), dtype=torch.bfloat16)
    assert not bi_configs.use_hopper_native_linear(x, narrow_weight, None)

    # Bias, non-BF16, unaligned N/K, non-contiguous weight, or non-SM90 are excluded.
    bias = torch.zeros(128, dtype=torch.bfloat16)
    assert not bi_configs.use_hopper_native_linear(x, weight, bias)
    assert not bi_configs.use_hopper_native_linear(
        x.to(torch.float16), weight.to(torch.float16), None
    )
    assert not bi_configs.use_hopper_native_linear(
        x, torch.empty((132, 64), dtype=torch.bfloat16), None
    )
    assert not bi_configs.use_hopper_native_linear(
        torch.empty((4, 68), dtype=torch.bfloat16),
        torch.empty((128, 68), dtype=torch.bfloat16),
        None,
    )
    non_contig_w = torch.empty((64, 128), dtype=torch.bfloat16).t()
    assert not bi_configs.use_hopper_native_linear(x, non_contig_w, None)

    monkeypatch.setattr(bi_configs, "supports_hopper_native_dispatch", lambda: False)
    assert not bi_configs.use_hopper_native_linear(x, weight, None)


def test_hopper_native_linear_and_lm_head_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(linear_mod.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(embed_mod.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(bi_configs, "supports_hopper_native_dispatch", lambda: True)

    seen_f_linear: list[bool] = []
    seen_bi_linear: list[bool] = []
    seen_embed_bi: list[bool] = []

    def fake_f_linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
        seen_f_linear.append(x.is_contiguous())
        return torch.zeros((x.shape[0], weight.shape[0]), dtype=x.dtype)

    def fake_bi_linear(
        x: torch.Tensor, weight: torch.Tensor, bias=None
    ) -> torch.Tensor:
        seen_bi_linear.append(True)
        return torch.zeros((x.shape[0], weight.shape[0]), dtype=x.dtype)

    def fake_embed_bi(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
        seen_embed_bi.append(True)
        return torch.zeros((x.shape[0], weight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(torch.nn.functional, "linear", fake_f_linear)
    monkeypatch.setattr(linear_mod, "linear_batch_invariant", fake_bi_linear)
    monkeypatch.setattr(embed_mod, "linear_batch_invariant", fake_embed_bi)

    method = linear_mod.UnquantizedLinearMethod()
    layer = torch.nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
    non_contig_x = torch.empty((8, 64), dtype=torch.bfloat16)[::2]
    assert not non_contig_x.is_contiguous()
    method.apply(layer, non_contig_x)
    assert seen_f_linear == [True]
    assert seen_bi_linear == []

    # Narrow router falls back to linear_batch_invariant.
    router = torch.nn.Linear(64, 8, bias=False, dtype=torch.bfloat16)
    method.apply(router, torch.empty((4, 64), dtype=torch.bfloat16))
    assert seen_bi_linear == [True]

    # Embedding/LM-head projection also passes contiguous activations to F.linear.
    embed_method = embed_mod.UnquantizedEmbeddingMethod()
    embed_method.apply(layer, non_contig_x)
    assert seen_f_linear == [True, True]
    assert seen_embed_bi == []


def test_hopper_inductor_and_pass_config_disable_allreduce_rms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(bi_configs, "supports_hopper_native_dispatch", lambda: True)

    inductor_cfg: dict[str, object] = {}
    set_inductor_config(inductor_cfg, Range(start=1, end=1))
    assert inductor_cfg == bi_configs.HOPPER_DETERMINISTIC_INDUCTOR_CONFIG
    assert inductor_cfg["deterministic"] is True
    assert inductor_cfg["max_autotune"] is False
    assert inductor_cfg["coordinate_descent_tuning"] is False
    assert inductor_cfg["triton.autotune_pointwise"] is False

    # fuse_allreduce_rms is forced off when VLLM_BATCH_INVARIANT=1 on SM90.
    pass_cfg = PassConfig(fuse_allreduce_rms=True)
    assert pass_cfg.fuse_allreduce_rms is False
    assert not enable_allreduce_rms_fusion(None)  # type: ignore[arg-type]

    # Non-SM90 platforms retain explicit PassConfig(fuse_allreduce_rms=True).
    monkeypatch.setattr(bi_configs, "supports_hopper_native_dispatch", lambda: False)
    non_sm90_pass_cfg = PassConfig(fuse_allreduce_rms=True)
    assert non_sm90_pass_cfg.fuse_allreduce_rms is True

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    normal_pass_cfg = PassConfig(fuse_allreduce_rms=True)
    assert normal_pass_cfg.fuse_allreduce_rms is True


def test_hopper_moe_get_moe_configs_filters_split_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_moe.get_moe_configs.cache_clear()
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(envs, "VLLM_TUNED_CONFIG_FOLDER", "/tmp/unused-user-folder")
    monkeypatch.setattr(bi_configs, "supports_hopper_native_dispatch", lambda: True)
    monkeypatch.setattr(
        fused_moe,
        "get_config_file_name",
        lambda E, N, dtype=None, block_shape=None: (
            f"E={E},N={N},device_name=NVIDIA_H200.json"
        ),
    )

    # Validated H200 unquantized table loads cleanly and has no SPLIT_K != 1.
    cfg = fused_moe.get_moe_configs(128, 768, None)
    assert cfg is not None
    assert all(tile.get("SPLIT_K", 1) == 1 for tile in cfg.values())

    # Unaudited (E, N) or non-H200 tables are rejected.
    assert fused_moe.get_moe_configs(64, 640, None) is None
    assert fused_moe.get_moe_configs(8, 14336, None) is None
    monkeypatch.setattr(
        fused_moe,
        "get_config_file_name",
        lambda E, N, dtype=None, block_shape=None: (
            f"E={E},N={N},device_name=NVIDIA_H100.json"
        ),
    )
    fused_moe.get_moe_configs.cache_clear()
    assert fused_moe.get_moe_configs(128, 768, None) is None

    # Any tile with SPLIT_K != 1 must be rejected.
    monkeypatch.setattr(
        fused_moe,
        "get_config_file_name",
        lambda E, N, dtype=None, block_shape=None: (
            f"E={E},N={N},device_name=NVIDIA_H200.json"
        ),
    )
    monkeypatch.setattr(
        json,
        "load",
        lambda _f: {"1": {"BLOCK_SIZE_M": 16, "SPLIT_K": 2}},
    )
    fused_moe.get_moe_configs.cache_clear()
    assert fused_moe.get_moe_configs(128, 768, None) is None
    fused_moe.get_moe_configs.cache_clear()


def test_hopper_native_dispatch_is_dynamo_traceable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bi_configs, "_HOPPER_NATIVE_DISPATCH", None)
    monkeypatch.setattr(bi_configs.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        bi_configs.current_platform,
        "is_device_capability",
        lambda cap: cap == (9, 0),
    )
    assert bi_configs.supports_hopper_native_dispatch() is True

    # Simulate untraceable NVML/ctypes lookup if invoked during Dynamo tracing.
    def _fail_in_dynamo(_cap):
        raise RuntimeError("must not query device capability during torch.compile")

    monkeypatch.setattr(
        bi_configs.current_platform, "is_device_capability", _fail_in_dynamo
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(linear_mod.current_platform, "is_cuda_alike", lambda: True)

    method = linear_mod.UnquantizedLinearMethod()
    layer = torch.nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)

    @torch.compile(backend="eager", fullgraph=True)
    def run_compiled(x: torch.Tensor) -> torch.Tensor:
        return method.apply(layer, x)

    out = run_compiled(torch.ones((4, 64), dtype=torch.bfloat16))
    assert out.shape == (4, 128)
    monkeypatch.setattr(bi_configs, "_HOPPER_NATIVE_DISPATCH", None)
