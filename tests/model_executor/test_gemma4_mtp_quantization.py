# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import torch.nn as nn

from vllm.config import CacheConfig, ModelConfig, SpeculativeConfig, VllmConfig


def _nn_identity(*args, **kwargs):
    return nn.Identity()


def _make_text_config():
    cfg = Mock()
    cfg.attention_bias = False
    cfg.attention_k_eq_v = False
    cfg.global_head_dim = 16
    cfg.head_dim = 8
    cfg.hidden_activation = "gelu_pytorch_tanh"
    cfg.hidden_size = 64
    cfg.intermediate_size = 256
    cfg.layer_types = ["sliding_attention", "full_attention"]
    cfg.max_position_embeddings = 1024
    cfg.num_attention_heads = 8
    cfg.num_hidden_layers = 2
    cfg.num_key_value_heads = 2
    cfg.rms_norm_eps = 1e-6
    cfg.rope_parameters = {
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        "full_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    }
    cfg.sliding_window = 1024
    cfg.vocab_size = 128
    return cfg


def _make_vllm_config():
    text_config = _make_text_config()

    hf_config = Mock()
    hf_config.backbone_hidden_size = 256
    hf_config.text_config = text_config
    hf_config.tie_word_embeddings = False
    hf_config.use_ordered_embeddings = False

    draft_model_config = Mock(spec=ModelConfig)
    draft_model_config.hf_config = hf_config
    draft_model_config.try_get_generation_config.return_value = None

    speculative_config = Mock(spec=SpeculativeConfig)
    speculative_config.draft_model_config = draft_model_config

    vllm_config = Mock(spec=VllmConfig)
    vllm_config.cache_config = Mock(spec=CacheConfig)
    vllm_config.speculative_config = speculative_config
    return vllm_config


def test_gemma4_mtp_attention_projections_receive_quant_config():
    """Attention q_proj/o_proj/attn must receive quant_config."""
    from vllm.model_executor.models.gemma4_mtp import Gemma4MTPAttention

    mock_quant_config = Mock()
    vllm_config = _make_vllm_config()
    config = vllm_config.speculative_config.draft_model_config.hf_config.text_config

    with (
        patch(
            "vllm.model_executor.models.gemma4_mtp."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.models.gemma4_mtp.ColumnParallelLinear",
            side_effect=_nn_identity,
        ) as mock_column,
        patch(
            "vllm.model_executor.models.gemma4_mtp.RowParallelLinear",
            side_effect=_nn_identity,
        ) as mock_row,
        patch(
            "vllm.model_executor.models.gemma4_mtp.RMSNorm", side_effect=_nn_identity
        ),
        patch(
            "vllm.model_executor.models.gemma4_mtp.get_rope", side_effect=_nn_identity
        ),
        patch(
            "vllm.model_executor.models.gemma4_mtp.Attention",
            side_effect=_nn_identity,
        ) as mock_attention,
    ):
        Gemma4MTPAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=mock_quant_config,
            prefix="layers.0.self_attn",
        )

    assert mock_column.call_args.kwargs["quant_config"] is mock_quant_config, (
        "q_proj must receive the draft model's quant_config"
    )
    assert mock_row.call_args.kwargs["quant_config"] is mock_quant_config, (
        "o_proj must receive the draft model's quant_config"
    )
    assert mock_attention.call_args.kwargs["quant_config"] is mock_quant_config, (
        "Attention must receive the draft model's quant_config"
    )


def test_gemma4_mtp_decoder_layer_passes_quant_config():
    """DecoderLayer must forward quant_config to its attention and MLP."""
    from vllm.model_executor.models.gemma4_mtp import Gemma4MTPDecoderLayer

    mock_quant_config = Mock()
    vllm_config = _make_vllm_config()
    config = vllm_config.speculative_config.draft_model_config.hf_config.text_config

    with (
        patch(
            "vllm.model_executor.models.gemma4_mtp.Gemma4MTPAttention",
            side_effect=_nn_identity,
        ) as mock_attention,
        patch(
            "vllm.model_executor.models.gemma4_mtp.Gemma4MLP",
            side_effect=_nn_identity,
        ) as mock_mlp,
        patch(
            "vllm.model_executor.models.gemma4_mtp.RMSNorm", side_effect=_nn_identity
        ),
    ):
        Gemma4MTPDecoderLayer(
            config,
            cache_config=Mock(spec=CacheConfig),
            quant_config=mock_quant_config,
            prefix="layers.0",
        )

    assert mock_attention.call_args.kwargs["quant_config"] is mock_quant_config, (
        "Gemma4MTPAttention must receive the draft model's quant_config"
    )
    assert mock_mlp.call_args.kwargs["quant_config"] is mock_quant_config, (
        "Gemma4MLP must receive the draft model's quant_config"
    )


def test_gemma4_mtp_predictor_uses_draft_quant_config():
    """Gemma4MultiTokenPredictor must source quant_config from the draft model."""
    from vllm.model_executor.models.gemma4_mtp import Gemma4MultiTokenPredictor

    mock_quant_config = Mock()
    vllm_config = _make_vllm_config()

    with (
        patch(
            "vllm.model_executor.models.gemma4_mtp.get_draft_quant_config",
            return_value=mock_quant_config,
        ),
        patch(
            "vllm.model_executor.models.gemma4_mtp.VocabParallelEmbedding",
            side_effect=_nn_identity,
        ) as mock_embedding,
        patch(
            "vllm.model_executor.models.gemma4_mtp.ColumnParallelLinear",
            side_effect=_nn_identity,
        ) as mock_column,
        patch(
            "vllm.model_executor.models.gemma4_mtp.RowParallelLinear",
            side_effect=_nn_identity,
        ) as mock_row,
        patch(
            "vllm.model_executor.models.gemma4_mtp.Gemma4MTPDecoderLayer",
            side_effect=_nn_identity,
        ) as mock_layer,
        patch(
            "vllm.model_executor.models.gemma4_mtp.RMSNorm", side_effect=_nn_identity
        ),
    ):
        Gemma4MultiTokenPredictor(vllm_config=vllm_config, prefix="draft_model")

    assert mock_embedding.call_args.kwargs["quant_config"] is mock_quant_config, (
        "VocabParallelEmbedding must receive the draft model's quant_config"
    )
    assert mock_column.call_args.kwargs["quant_config"] is mock_quant_config, (
        "pre_projection must receive the draft model's quant_config"
    )
    assert mock_row.call_args.kwargs["quant_config"] is mock_quant_config, (
        "post_projection must receive the draft model's quant_config"
    )
    assert all(
        call.kwargs["quant_config"] is mock_quant_config
        for call in mock_layer.call_args_list
    ), "all Gemma4MTPDecoderLayer instances must receive the draft model's quant_config"


def test_gemma4_mtp_lm_head_receives_draft_quant_config():
    """Gemma4MTP lm_head must receive the draft model's quant_config."""
    from vllm.model_executor.models.gemma4_mtp import Gemma4MTP

    mock_quant_config = Mock()
    vllm_config = _make_vllm_config()

    with (
        patch(
            "vllm.compilation.decorators.TorchCompileWithNoGuardsWrapper.__init__",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.gemma4_mtp.get_draft_quant_config",
            return_value=mock_quant_config,
        ),
        patch(
            "vllm.model_executor.models.gemma4_mtp.Gemma4MultiTokenPredictor",
            side_effect=_nn_identity,
        ),
        patch(
            "vllm.model_executor.models.gemma4_mtp.ParallelLMHead",
            side_effect=_nn_identity,
        ) as mock_lm_head,
        patch("vllm.model_executor.models.gemma4_mtp.LogitsProcessor"),
    ):
        Gemma4MTP(vllm_config=vllm_config)

    assert mock_lm_head.call_args.kwargs["quant_config"] is mock_quant_config, (
        "ParallelLMHead must receive the draft model's quant_config"
    )
