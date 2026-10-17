# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.parametrize("bf16_activations", [False, True])
def test_nemotron_h_lm_head_receives_quant_config(bf16_activations):
    """The lm_head gets the quant config, whose FP8 linears run as W8A16 only
    in the bf16_activations mode."""
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    mock_quant_config = ModelOptFp8Config(
        quant_method="FP8",
        is_checkpoint_fp8_serialized=True,
        kv_cache_quant_method=None,
        exclude_modules=[],
    )

    mock_hf_config = Mock()
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64
    mock_hf_config.bf16_activations = bf16_activations

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_config = mock_hf_config
    mock_vllm_config.model_config.dtype = None
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config

    with (
        patch("vllm.model_executor.models.nemotron_h.NemotronHModel") as MockModel,
        patch("vllm.model_executor.models.nemotron_h.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.nemotron_h.LogitsProcessor"),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()
        MockModel.return_value.has_moe = False

        NemotronHForCausalLM(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
    assert mock_quant_config.w8a16 is bf16_activations


def test_relu2_fp8_fusion_uses_registry():
    from vllm.model_executor.models.nemotron_h import NemotronHMLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = NemotronHMLP.__new__(NemotronHMLP)
    torch.nn.Module.__init__(mlp)
    mlp.up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.nemotron_h.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused


@pytest.mark.parametrize(
    "custom_ops,fuse_norm_quant,expected_ops,error",
    [
        (["none"], None, ["none", "+quant_fp8"], None),
        (["none", "+quant_fp8"], False, ["none", "+quant_fp8"], None),
        (["none"], True, None, "fuse_norm_quant=True"),
        (["-quant_fp8"], None, None, "-quant_fp8"),
    ],
)
def test_nemotron_h_batch_invariant_uses_cuda_fp8_quant(
    custom_ops, fuse_norm_quant, expected_ops, error
):
    """Batch-invariant Nemotron-H quantizes FP8 activations with the standalone
    CUDA kernel that training replays and rejects a quant fusion or the native
    op instead of silently rounding differently."""
    from vllm.model_executor.models.config import NemotronHForCausalLMConfig

    pass_config = SimpleNamespace(fuse_norm_quant=fuse_norm_quant, fuse_act_quant=None)
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            custom_ops=list(custom_ops), pass_config=pass_config
        )
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            NemotronHForCausalLMConfig.use_cuda_fp8_quant(vllm_config)
        return
    NemotronHForCausalLMConfig.use_cuda_fp8_quant(vllm_config)
    assert vllm_config.compilation_config.custom_ops == expected_ops
    assert pass_config.fuse_norm_quant is False
    assert pass_config.fuse_act_quant is False


@pytest.fixture
def on_sm10x(monkeypatch):
    """The batch-invariant config checks below assume a supported GPU."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 100
    )


def test_nemotron_h_batch_invariant_requires_sm10x(monkeypatch):
    """Off SM10x, batch-invariant Nemotron-H fails at config time with the
    reason, not later in a kernel check inside the engine core."""
    from vllm.model_executor.models import config
    from vllm.platforms import current_platform

    monkeypatch.setattr(config.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: False
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="float32"),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    with pytest.raises(ValueError, match="SM10x"):
        config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)


@pytest.mark.usefixtures("on_sm10x")
@pytest.mark.parametrize("batch_invariant", [False, True])
def test_nemotron_h_batch_invariant_requires_tp1(monkeypatch, batch_invariant):
    """Only TP=1 matches the training replay, so batch-invariant Nemotron-H
    rejects TP>1 at config time instead of in every worker."""
    from vllm.model_executor.models import config

    monkeypatch.setattr(config.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="float32"),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    if batch_invariant:
        with pytest.raises(ValueError, match="TP=1"):
            config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)
    else:
        config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)


@pytest.mark.usefixtures("on_sm10x")
@pytest.mark.parametrize(
    "kv_cache_dtype,error",
    [("auto", False), ("bfloat16", True), ("fp8", False), ("fp8_e5m2", True)],
)
def test_nemotron_h_batch_invariant_kv_cache_dtype(monkeypatch, kv_cache_dtype, error):
    """The fixed split-KV FA4 backend serves BF16 and FP8 E4M3 KV caches; any
    other dtype is rejected at config time instead of when building layers.
    Without bf16_activations the training replay runs the FP8 KV cache, so a
    BF16 one would silently diverge."""
    from vllm.model_executor.models import config

    monkeypatch.setattr(config.envs, "VLLM_BATCH_INVARIANT", True)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype="float32", cache_dtype=kv_cache_dtype
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        compilation_config=SimpleNamespace(
            custom_ops=[],
            pass_config=SimpleNamespace(fuse_norm_quant=None, fuse_act_quant=None),
        ),
    )
    if error:
        with pytest.raises(ValueError, match="KV cache"):
            config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)
    else:
        config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)


_MIXED_FP8_W4A16 = {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "quantized_layers": {
        "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
        "backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "W4A16_NVFP4"},
    },
}


@pytest.mark.usefixtures("on_sm10x")
@pytest.mark.parametrize(
    "dtype,kv_cache_dtype,quantization,hf_quant_config,error",
    [
        (torch.bfloat16, "bfloat16", "modelopt_mixed", _MIXED_FP8_W4A16, None),
        (torch.bfloat16, "auto", None, None, None),
        (torch.bfloat16, "fp8", "modelopt_mixed", _MIXED_FP8_W4A16, "BF16 KV"),
        (torch.float16, "bfloat16", "modelopt_mixed", _MIXED_FP8_W4A16, "--dtype"),
        (
            torch.bfloat16,
            "bfloat16",
            "modelopt",
            {"quantization": {"quant_algo": "FP8_PB_WO"}},
            "FP8_PB_WO",
        ),
        (torch.bfloat16, "bfloat16", "fp8", None, "no ModelOpt algo"),
    ],
)
def test_nemotron_h_bf16_activations(
    monkeypatch, dtype, kv_cache_dtype, quantization, hf_quant_config, error
):
    """bf16_activations serves FP8 linears as W8A16, so it rejects anything
    that would still quantize activations (an FP8 KV cache, other quantized
    layers) and, under batch invariance, leaves the FP8 quant op alone."""
    from vllm.model_executor.models import config

    monkeypatch.setattr(config.envs, "VLLM_BATCH_INVARIANT", True)
    pass_config = SimpleNamespace(fuse_norm_quant=True, fuse_act_quant=None)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype="float32", cache_dtype=kv_cache_dtype
        ),
        model_config=SimpleNamespace(
            dtype=dtype,
            quantization=quantization,
            hf_config=SimpleNamespace(
                bf16_activations=True, quantization_config=hf_quant_config
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        compilation_config=SimpleNamespace(
            custom_ops=["-quant_fp8"], pass_config=pass_config
        ),
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)
        return
    config.NemotronHForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.compilation_config.custom_ops == ["-quant_fp8"]
    assert pass_config.fuse_norm_quant is True


@pytest.mark.parametrize("tp_size", [1, 2])
def test_grouped_gated_norm_only_at_tp1(monkeypatch, tp_size):
    """The grouped gated-norm kernel normalizes whole groups on one rank; with
    TP the default (batch-invariant, all-reducing) path is used instead of
    failing at runtime."""
    from vllm.model_executor.layers.mamba import mamba_mixer2

    monkeypatch.setattr(mamba_mixer2.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        torch.ops.vllm, "grouped_gated_rms_norm", lambda *args: "grouped"
    )
    monkeypatch.setattr(
        mamba_mixer2.Mixer2RMSNormGated, "forward", lambda self, x, gate: "default"
    )
    cls = mamba_mixer2.GroupedMixer2RMSNormGated
    norm = cls.__new__(cls)
    torch.nn.Module.__init__(norm)
    norm.tp_size = tp_size
    norm.use_rms_norm = True
    norm.weight = torch.nn.Parameter(torch.ones(4))
    norm.group_size = 4
    norm.variance_epsilon = 1e-5
    out = cls.forward(norm, torch.ones(2, 4), None)
    assert out == ("grouped" if tp_size == 1 else "default")
