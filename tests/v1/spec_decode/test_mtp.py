# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch
import torch.nn as nn

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
)
from vllm.config.load import LoadConfig
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


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    [
        (
            "vllm.model_executor.models.glm4_moe_lite_mtp",
            "Glm4MoeLiteMultiTokenPredictorLayer",
        ),
        (
            "vllm.model_executor.models.glm4_moe_mtp",
            "Glm4MoeMultiTokenPredictorLayer",
        ),
        (
            "vllm.model_executor.models.glm_ocr_mtp",
            "GlmOcrMultiTokenPredictorLayer",
        ),
    ],
)
def test_glm_mtp_layer_recycles_post_final_norm_hidden(
    default_vllm_config, module_path, class_name
):
    # GLM MTP is a DeepSeek-NextN clone. The layer forward must apply the final
    # norm exactly once (shared_head.norm, fused add+norm over the mtp_block
    # output + residual) and return that POST-final-norm hidden for BOTH tuple
    # positions, so the speculator recycles post-norm into the next step's hnorm
    # and compute_logits consumes it directly. This is the contract fixed in
    # #47448 and vllm/models/glm5next/common/mtp.py. Returning the single
    # pre-norm tensor recycled a hidden one RMSNorm short on chained draft steps
    # (k>=2) — lossless, but lowers draft acceptance.
    #
    # A real RMSNorm with NON-UNIFORM weights is used so that a missing, extra,
    # or wrong normalization is actually caught (a scalar stub would not).
    import importlib
    from types import SimpleNamespace

    from vllm.model_executor.layers.layernorm import RMSNorm

    cls = getattr(importlib.import_module(module_path), class_name)

    hidden_size = 8
    torch.manual_seed(0)
    final_norm = RMSNorm(hidden_size, eps=1e-6)
    with torch.no_grad():
        final_norm.weight.copy_(torch.linspace(0.5, 1.5, hidden_size))

    # Deterministic, distinct block output and residual.
    block_hidden = torch.randn(3, hidden_size)
    block_residual = torch.randn(3, hidden_size)

    # Independently compute the expected post-final-norm hidden BEFORE calling
    # the layer (the fused norm may mutate its inputs in place).
    expected_post, _ = final_norm.forward_native(
        block_hidden.clone(), residual=block_residual.clone()
    )
    pre_norm = block_hidden.clone() + block_residual.clone()

    layer = object.__new__(cls)
    layer.enorm = lambda x: x
    layer.hnorm = lambda x: x
    layer.eh_proj = lambda x: x[..., :hidden_size]
    layer.mtp_block = lambda positions, hidden_states, residual: (
        block_hidden,
        block_residual,
    )
    layer.shared_head = SimpleNamespace(norm=final_norm)

    out = layer.forward(
        input_ids=None,
        positions=torch.tensor([1, 2, 3]),
        previous_hidden_states=torch.randn(3, hidden_size),
        inputs_embeds=torch.randn(3, hidden_size),
    )

    assert isinstance(out, tuple) and len(out) == 2
    logits_hidden, recycle_hidden = out
    # Both tuple positions carry the same post-final-norm hidden.
    assert torch.allclose(logits_hidden, recycle_hidden)
    # It equals the real fused final norm of (block_hidden + residual) — i.e. the
    # correct normalization is recycled (and, since compute_logits now consumes
    # this directly, the logits path is unchanged from before the fix).
    assert torch.allclose(recycle_hidden, expected_post, atol=1e-5)
    # And it is NOT the pre-norm hidden (the old, buggy recycle value).
    assert not torch.allclose(recycle_hidden, pre_norm)
    # Logits-equivalence: the fused add+norm equals the previous eager
    # `shared_head(residual + hidden)` path, so the draft-logits input is
    # preserved (compute_logits now consumes this post-norm hidden directly).
    # Any runtime difference is floating-point from fusing the add into the norm.
    eager_post = final_norm(pre_norm.clone())
    assert torch.allclose(expected_post, eager_post, atol=1e-5)


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
    target_indexer_layers: dict = {}
    all_indexer_layers: dict = {}

    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indexer_layers,
        all_attn_layers,
        all_indexer_layers,
    ]

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


@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
@pytest.mark.parametrize("returns_tuple", [False, True])
def test_mtp_propose(num_speculative_tokens, returns_tuple, monkeypatch):
    """Validate the MTP proposer consumes the draft model output, including the
    (logits_hidden, feedback_hidden) tuple returned by GLM/DeepSeek-style MTP.

    returns_tuple=True exercises the tuple path (model_returns_tuple() True): the
    proposer must unpack it at every draft step (num_speculative_tokens 1 and 2)
    rather than index the tuple as a single tensor.
    """
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    def _ret(x):
        # GLM/DeepSeek MTP return (logits_hidden, feedback_hidden); for the GLM
        # post-final-norm contract both positions are the same tensor.
        return (x, x) if returns_tuple else x

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = _ret(
            torch.zeros(total_tokens, hidden_size, device=device)
        )
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(_ret(h_states))
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
    # Attention-backend compatibility (the num_speculative_tokens > 1 type check)
    # is out of scope here: the model is mocked, so the per-group attn metadata is
    # itself a mock. Disable the check to exercise the multi-step proposer logic.
    proposer.allowed_attn_types = None
    # Multi-step drafting (num_speculative_tokens > 1) reads block_size to build
    # per-step slot mappings; match the mocked attention metadata's block size.
    proposer.block_size = 16
    if returns_tuple:
        # GLM/DeepSeek-style MTP architectures report a tuple return; force it
        # here so the proposer takes the unpacking path for the mocked model.
        monkeypatch.setattr(proposer, "model_returns_tuple", lambda: True)

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
