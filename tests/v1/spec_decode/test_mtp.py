# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.load import LoadConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.eagle import EagleProposer

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"
DEVICE_TYPE = current_platform.device_type


def test_glm_mtp_defers_lm_head(default_vllm_config):
    from vllm.model_executor.models import deepseek_mtp
    from vllm.models.glm5next.common import mtp

    config = mock.MagicMock(
        hidden_size=16,
        rms_norm_eps=1e-5,
        index_topk=8,
        index_kpool=4,
    )
    vllm_config = mock.MagicMock()
    vllm_config.speculative_config.draft_model_config.hf_text_config = config
    vllm_config.scheduler_config.max_num_batched_tokens = 4

    with (
        mock.patch.object(mtp, "Glm5NextDecoderLayer", return_value=nn.Identity()),
        mock.patch.object(deepseek_mtp, "ParallelLMHead") as parallel_lm_head,
        mock.patch.object(mtp.current_platform, "device_type", "cpu"),
    ):
        layer = mtp.Glm5NextMultiTokenPredictorLayer(vllm_config, "model.layers.1")

    parallel_lm_head.assert_not_called()
    assert layer.shared_head.head is None


def _tiny_glm_mtp_config(model: str) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model)
    shrink = dict(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=512,
    )
    moe = dict(
        moe_intermediate_size=128,
        first_k_dense_replace=0,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
    )
    if model == "zai-org/GLM-OCR":
        config.text_config.update({**shrink, "num_key_value_heads": 1})
    elif model == "zai-org/GLM-4.5-Air":
        config.update({**shrink, **moe, "num_key_value_heads": 1, "head_dim": 128})
    else:
        config.update({**shrink, **moe, "num_key_value_heads": 2})
    return config


@pytest.mark.parametrize(
    "model", ["zai-org/GLM-4.5-Air", "zai-org/GLM-4.7-Flash", "zai-org/GLM-OCR"]
)
def test_glm_mtp_feeds_back_final_normed_hidden(
    dist_init, workspace_init, tmp_path, model
):
    """GLM MTP feeds the final-normed hidden into the next draft step (as
    DeepSeek and GLM-5 Next do) while keeping the logits input unchanged."""
    _tiny_glm_mtp_config(model).save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path), tokenizer_mode="skip", max_model_len=256, dtype="bfloat16"
    )
    speculative_config = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=1,
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        speculative_config=speculative_config,
        load_config=LoadConfig(load_format="dummy"),
    )
    with set_current_vllm_config(vllm_config):
        drafter = get_model(
            vllm_config=vllm_config, model_config=speculative_config.draft_model_config
        )
    (layer,) = drafter.model.layers.values()
    block_outputs = []
    layer.mtp_block.register_forward_hook(
        lambda module, args, out: block_outputs.append(out[0] + out[1])
    )

    num_tokens = 6
    positions = torch.arange(1, num_tokens + 1, device=DEVICE_TYPE)
    if model_config.uses_mrope:
        positions = positions.expand(3, num_tokens).contiguous()
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        out = drafter(
            input_ids=torch.randint(0, 512, (num_tokens,), device=DEVICE_TYPE),
            positions=positions,
            hidden_states=torch.randn(
                num_tokens, 256, device=DEVICE_TYPE, dtype=torch.bfloat16
            ),
        )
    # Proposers use a single returned tensor for both logits and feedback.
    logits_hidden, feedback_hidden = out if isinstance(out, tuple) else (out, out)

    (pre_norm,) = block_outputs
    final_normed = layer.shared_head.norm(pre_norm)
    torch.testing.assert_close(feedback_hidden, final_normed)
    # compute_logits used to apply shared_head to the pre-norm sum itself.
    torch.testing.assert_close(logits_hidden, final_normed)


def _create_mtp_proposer(num_speculative_tokens: int) -> EagleProposer:
    """Create an MTP proposer with unified model configuration."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100, trust_remote_code=True
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE_TYPE),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    return EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)


@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_pp_group")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_model")
def test_mtp_load_model_unified(mock_get_model, mock_get_layers, mock_get_pp_group):
    """Test MTP-specific model loading with unified model approach."""
    # Setup mocks
    mock_model = mock.MagicMock()
    mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    # MTP does not have its own embed_tokens or lm_head
    # so it should share them with the target model
    mock_model.has_own_embed_tokens = False
    mock_model.has_own_lm_head = False

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_attn_1": mock.MagicMock()}

    mock_get_layers.side_effect = [target_attn_layers, all_attn_layers]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    # Create target model
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    target_model.lm_head = mock.MagicMock()

    # Create MTP proposer
    proposer = _create_mtp_proposer(num_speculative_tokens=4)
    proposer.load_model(target_model)

    # Verify MTP-specific behavior:
    # Model is loaded
    mock_get_model.assert_called_once()
    # MTP shares lm_head with target model
    assert proposer.model.lm_head == target_model.lm_head
    # MTP shares embed_tokens with target model
    assert proposer.model.model.embed_tokens == target_model.model.embed_tokens
    assert proposer._draft_attn_layer_names == {"draft_attn_1"}
    assert all_attn_layers["draft_attn_1"].is_draft_layer is True


@pytest.mark.parametrize("num_speculative_tokens", [1])
def test_mtp_propose(num_speculative_tokens, monkeypatch):
    """Test that MTP's forward method returns hidden states directly."""
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(h_states)
        model_mock.side_effect = forward_returns

    # Mock compute_logits
    def create_deterministic_logits(batch_size, vocab_size, token_offset):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_offset] = 100.0
        return logits

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = create_deterministic_logits(
            batch_size, vocab_size, 42
        )
    else:
        logits_returns = [
            create_deterministic_logits(batch_size, vocab_size, 42 + i)
            for i in range(num_speculative_tokens)
        ]
        model_mock.compute_logits.side_effect = logits_returns

    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    # Prepare inputs
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    # Setup attention metadata
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )

    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    # Run propose
    result = proposer.propose(
        num_speculative_tokens=num_speculative_tokens,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    # Verify the model was called correctly
    assert model_mock.called
    # Verify output shape
    assert result.shape == (batch_size, num_speculative_tokens)
