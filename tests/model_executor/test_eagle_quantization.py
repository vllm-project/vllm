# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import (
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type
DEVICES = (
    [f"{DEVICE_TYPE}:{i}" for i in range(min(torch.accelerator.device_count(), 2))]
    if not current_platform.is_cpu()
    else ["cpu"]
)


# A quantized draft checkpoint's compressed-tensors config (W4A16, group 64): as
# in real checkpoints, its targets name the unfused projections.
_PROJ = r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
W4A16_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "format": "pack-quantized",
            "targets": [rf"re:.*layers\.\d+\.{_PROJ}$"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 64,
                "dynamic": False,
            },
            "input_activations": None,
            "output_activations": None,
        }
    },
    "ignore": [],
}
TINY_QWEN3 = {
    "model_type": "qwen3",
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 1024,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
}


def _model_dir(path: Path, **fields) -> str:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(TINY_QWEN3 | fields))
    return str(path)


def _dflash_vllm_config(
    tmp_path: Path,
    draft_architecture: str,
    draft_quant: dict | None,
) -> VllmConfig:
    """A target and a DFlash draft built from config files only (no weights)."""
    target = ModelConfig(
        model=_model_dir(tmp_path / "target", architectures=["Qwen3ForCausalLM"]),
        skip_tokenizer_init=True,
    )
    draft_fields = {
        "architectures": [draft_architecture],
        "dflash_config": {"mask_token_id": 0, "target_layer_ids": [0, 1]},
        "block_size": 4,
    }
    if draft_quant is not None:
        draft_fields["quantization_config"] = draft_quant
    speculative_config = SpeculativeConfig(
        model=_model_dir(tmp_path / "draft", **draft_fields),
        method="dflash",
        num_speculative_tokens=3,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )
    return VllmConfig(
        model_config=target,
        speculative_config=speculative_config,
        load_config=LoadConfig(),
    )


def _weights(quant_config: CompressedTensorsConfig, layer_name: str):
    scheme = quant_config.get_scheme_dict(torch.nn.Module(), layer_name=layer_name)
    return scheme["weights"] if scheme is not None else None


@pytest.mark.parametrize("architecture", ["DFlashDraftModel", "DFlash2DraftModel"])
def test_get_draft_quant_config_quantizes_fused_layers(tmp_path, architecture):
    """A quantized draft's fused layers (qkv_proj, gate_up_proj) get the
    checkpoint's scheme, like the unfused ones.

    Regression test for #53116: the draft's quant config had no
    packed_modules_mapping, so fused layers matched no target and were
    built unquantized."""
    vllm_config = _dflash_vllm_config(tmp_path, architecture, W4A16_QUANT_CONFIG)

    quant_config = get_draft_quant_config(vllm_config)

    assert isinstance(quant_config, CompressedTensorsConfig)
    for layer in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj"):
        weights = _weights(quant_config, f"model.layers.0.{layer}")
        assert weights is not None, layer
        assert weights.num_bits == 4, layer
        assert weights.group_size == 64, layer


def test_get_draft_quant_config_keeps_ignored_fused_layers_unquantized(tmp_path):
    """A fused layer whose projections the checkpoint ignores stays unquantized."""
    quant = copy.deepcopy(W4A16_QUANT_CONFIG)
    quant["ignore"] = [f"model.layers.1.self_attn.{p}_proj" for p in "qkv"]
    vllm_config = _dflash_vllm_config(tmp_path, "DFlashDraftModel", quant)

    quant_config = get_draft_quant_config(vllm_config)

    assert isinstance(quant_config, CompressedTensorsConfig)
    assert _weights(quant_config, "model.layers.0.self_attn.qkv_proj") is not None
    assert _weights(quant_config, "model.layers.1.self_attn.qkv_proj") is None


def test_get_draft_quant_config_unquantized_draft(tmp_path):
    """An unquantized draft has no quant config."""
    vllm_config = _dflash_vllm_config(tmp_path, "DFlashDraftModel", None)

    assert get_draft_quant_config(vllm_config) is None


def test_get_draft_quant_config_without_draft_model():
    mock_speculative_config = Mock(spec=SpeculativeConfig)
    mock_speculative_config.draft_model_config = None

    mock_vllm_config = Mock(spec=VllmConfig)
    mock_vllm_config.speculative_config = mock_speculative_config
    mock_vllm_config.load_config = Mock(spec=LoadConfig)

    result = get_draft_quant_config(mock_vllm_config)

    assert result is None


@pytest.fixture
def default_device(device):
    original = torch.get_default_device()
    torch.set_default_device(device)
    yield
    torch.set_default_device(original)


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
def test_fc_layer_quant_config_usage(
    default_vllm_config, dist_init, device, default_device
) -> None:
    import torch

    from vllm.model_executor.layers.linear import ReplicatedLinear

    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)

    input_size = 256
    output_size = 128

    fc_no_quant = ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        params_dtype=torch.float16,
        quant_config=None,
        prefix="fc",
    )

    assert fc_no_quant.quant_config is None
    assert fc_no_quant.input_size == input_size
    assert fc_no_quant.output_size == output_size

    mock_quant_config = Mock()
    mock_quant_config.online_quantization_config = None
    fc_with_quant = ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        params_dtype=torch.float16,
        quant_config=mock_quant_config,
        prefix="fc",
    )

    assert fc_with_quant.quant_config == mock_quant_config

    # Check forward pass
    x = torch.randn(2, input_size, dtype=torch.float16)
    output, _ = fc_no_quant(x)
    assert output.shape == (2, output_size)


def test_maybe_remap_kv_scale_name():
    from vllm.model_executor.model_loader.weight_utils import maybe_remap_kv_scale_name

    params_dict = {
        "layers.0.self_attn.kv_scale": Mock(),
        "layers.1.self_attn.kv_scale": Mock(),
    }

    name = "layers.0.self_attn.some_scale"
    remapped = maybe_remap_kv_scale_name(name, params_dict)

    assert remapped in params_dict or remapped == name or remapped is None


def test_eagle3_lm_head_receives_quant_config():
    """Eagle3LlamaForCausalLM must pass quant_config to ParallelLMHead.

    Without quant_config, quantized lm_head weights (e.g. INT8 per-channel)
    in Eagle3 drafter checkpoints fail to load because ParallelLMHead doesn't
    expect weight_packed tensors.
    """
    from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.draft_vocab_size = 1000
    mock_hf_config.hidden_size = 256
    mock_hf_config.vocab_size = 32000
    mock_hf_config.logit_scale = 1.0

    mock_vllm_config = Mock()
    mock_vllm_config.speculative_config.draft_model_config.hf_config = mock_hf_config
    mock_vllm_config.model_config.get_num_layers.return_value = 32

    with (
        patch("vllm.model_executor.models.llama_eagle3.LlamaModel") as MockModel,
        patch("vllm.model_executor.models.llama_eagle3.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.llama_eagle3.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.llama_eagle3.get_draft_quant_config",
            return_value=mock_quant_config,
        ),
    ):
        MockModel.return_value.use_aux_hidden_state = True

        Eagle3LlamaForCausalLM(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert "quant_config" in call_kwargs, (
            "ParallelLMHead must receive quant_config for quantized lm_head weights"
        )
        assert call_kwargs["quant_config"] is mock_quant_config, (
            "ParallelLMHead must receive the draft model's quant_config"
        )
