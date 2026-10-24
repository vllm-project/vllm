# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for mHC warmup after the TileLang provider migration."""

import importlib
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest
import torch

from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_custom_op_warmup,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry


@pytest.mark.parametrize("model_kind", ["mtp", "dspark"])
@pytest.mark.parametrize("enabled", [False, True])
def test_draft_model_registers_its_own_head(
    monkeypatch: pytest.MonkeyPatch, model_kind: str, enabled: bool
) -> None:
    """Draft construction must not rely on the target's head registration."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _HC_HEAD_FUSED_TILELANG_KERNEL,
    )

    module = importlib.import_module(f"vllm.models.deepseek_v4.nvidia.{model_kind}")

    class DummyModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    hf_config = SimpleNamespace(
        hidden_size=32,
        hc_mult=4,
        hc_eps=2e-5,
        rms_norm_eps=3e-5,
        num_hidden_layers=1,
        dspark_target_layer_ids=[0],
        n_mtp_layers=1,
        vocab_size=64,
        index_topk=1,
        dspark_markov_rank=4,
        enable_confidence_head=False,
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        kernel_config=SimpleNamespace(enable_jit_warmup=enabled),
        quant_config=None,
    )
    for name in ("RMSNorm", "ReplicatedLinear", "DeepseekV4DecoderLayer"):
        monkeypatch.setattr(module, name, DummyModule)
    if model_kind == "mtp":
        monkeypatch.setattr(module, "SharedHead", DummyModule)
        for name in (
            "_FUSED_MTP_INPUT_RMSNORM_KERNEL",
            "_MTP_SHARED_HEAD_RMSNORM_KERNEL",
        ):
            monkeypatch.setattr(getattr(module, name), "register_warmup", lambda: None)

        def construct():
            return module.DeepSeekV4MultiTokenPredictorLayer(
                config, torch.empty(1, 1, dtype=torch.int32), "model.layers.1"
            )
    else:
        for name in ("VocabParallelEmbedding", "DSparkMarkovHead"):
            monkeypatch.setattr(module, name, DummyModule)
        monkeypatch.setattr(module, "_use_sequence_parallel", lambda _: False)
        monkeypatch.setattr(module, "get_current_vllm_config", lambda: config)

        def construct():
            return module.DSparkDeepseekV4Model(vllm_config=config)

    compiled = []
    monkeypatch.setattr(_HC_HEAD_FUSED_TILELANG_KERNEL, "compile", compiled.append)
    monkeypatch.setattr("vllm.distributed.is_global_first_rank", lambda: False)
    registry = JitWarmupRegistry(config)
    with registry.activate():
        construct()
        construct()
    registry.warmup()

    expected = _HC_HEAD_FUSED_TILELANG_KERNEL.dispatch(
        hidden_size=32, hc_mult=4, rms_eps=3e-5, hc_eps=2e-5
    )
    assert compiled == ([expected] if enabled else [])


@pytest.mark.parametrize("hidden_size", [1024, 7168, 768])
@pytest.mark.parametrize("max_tokens", [0, 17, 127, 128, 1023, 1024, 2048])
def test_upstream_prenorm_warmup_covers_all_runtime_shapes(
    hidden_size: int, max_tokens: int
) -> None:
    """Keep the fallback boundary guarantee when adopting upstream providers."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        HcPrenormGemmTileLangKernel,
    )

    kernel = HcPrenormGemmTileLangKernel()
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens)
    )
    fields = dict(hidden_size=hidden_size, hc_mult=2, n_out=8)
    warmed = set(kernel.get_warmup_keys(config, **fields))
    runtime = {
        kernel.dispatch(num_tokens=tokens, hc_hidden_size=hidden_size * 2, **fields)
        for tokens in range(1, max_tokens + 1)
    }
    assert warmed == runtime


def test_custom_op_warmup_skips_nvidia_layer() -> None:
    layer = SimpleNamespace(
        hc_attn_fn=SimpleNamespace(device=SimpleNamespace(type="cuda")),
    )
    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))

    with (
        mock.patch(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup._find_first_mhc_layer",
            return_value=layer,
        ),
        mock.patch(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup."
            "_select_custom_op_warmup_token_sizes"
        ) as select_sizes,
    ):
        deepseek_v4_mhc_custom_op_warmup(
            cast(torch.nn.Module, model),
            max_tokens=1024,
        )

    select_sizes.assert_not_called()


def test_custom_op_warmup_preserves_existing_amd_path() -> None:
    layer = SimpleNamespace(
        hc_pre=object(),
        hc_post=object(),
        hc_attn_fn=SimpleNamespace(device=SimpleNamespace(type="cuda")),
    )
    head = SimpleNamespace()
    model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))

    with (
        mock.patch(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup._find_first_mhc_layer",
            return_value=layer,
        ),
        mock.patch(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup._find_mhc_head_module",
            return_value=head,
        ),
        mock.patch(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup."
            "_select_custom_op_warmup_token_sizes",
            return_value=[1, 16],
        ),
        mock.patch(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup."
            "_warmup_custom_op_mhc_layer"
        ) as warm_layer,
        mock.patch(
            "vllm.model_executor.warmup.deepseek_v4_mhc_warmup."
            "_warmup_custom_op_hc_head"
        ) as warm_head,
        mock.patch.object(torch.accelerator, "synchronize") as synchronize,
    ):
        deepseek_v4_mhc_custom_op_warmup(
            cast(torch.nn.Module, model),
            max_tokens=1024,
        )

    warm_layer.assert_called_once_with(layer, [1, 16])
    warm_head.assert_called_once_with(head, [1, 16])
    synchronize.assert_called_once_with()
