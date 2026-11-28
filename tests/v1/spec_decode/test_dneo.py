# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DNeo drafter: feature encoder, causal head and speculators config translation."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dneo import DNeoCausalHead, DNeoFeatureEncoder
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.speculators.algos import update_dneo
from vllm.v1.worker.gpu.spec_decode.dneo.config import DNeoSpeculatorConfig


def _speculators_config(**overrides) -> dict:
    config = {
        "speculators_model_type": "dneo",
        "aux_hidden_state_layer_ids": [0, 17, 21, 24, 28, 31, 35],
        "block_size": 16,
        "mask_token_id": 151669,
        "draft_vocab_size": 151936,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "causal_head_rank": 256,
        "causal_head_folded_rank": 128,
        "feature_encoder": "mofa",
        "shared_feature_encoder": True,
        "feature_source": "attention_input",
        "prefix_only_attention": True,
        "rejection_memory": True,
        "sample_from_anchor": False,
    }
    config.update(overrides)
    return config


def test_feature_encoder_normalizes_each_layer_then_mixes():
    torch.manual_seed(0)
    layers, hidden, eps = 3, 8, 1e-6
    encoder = DNeoFeatureEncoder(layers, hidden, eps, params_dtype=torch.float32)
    with torch.no_grad():
        encoder.layer_logits.normal_()
        encoder.input_norm_weight.normal_()
    x = torch.randn(5, layers, hidden) * 40.0  # residual-stream scale
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    normed = normed * encoder.input_norm_weight
    want = (encoder.layer_logits.softmax(-1)[None, :, None] * normed).sum(1)
    torch.testing.assert_close(encoder(x.view(5, layers * hidden)), want)
    assert encoder.input_size == layers * hidden


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires a GPU")
def test_causal_head_step_logits(dist_init):
    torch.manual_seed(0)
    vocab, hidden, rank, folded, block = 64, 16, 8, 6, 4
    device = current_platform.device_type
    head = DNeoCausalHead(
        vocab_size=vocab,
        draft_vocab_size=vocab,
        hidden_size=hidden,
        rank=rank,
        folded_rank=folded,
        block_size=block,
        params_dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_()
    processor = LogitsProcessor(vocab)
    prev = torch.tensor([3, 7], device=device)
    base = torch.randn(2, vocab, device=device)
    hidden_states = torch.randn(2, hidden, device=device)
    projected = head.hidden_projection(hidden_states)
    slot = 2
    a, b = head.markov_scale[slot], head.context_scale[slot]
    latent = a * head.markov_w1.weight[prev] + b * (
        (head.predecessor_codebook.weight[prev] * projected)
        @ head.latent_projection.weight.T
    )
    want = (a + b) * base + latent @ head.markov_w2.weight.T
    got = head.step_logits(base, projected, prev, slot, processor)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_update_dneo_translates_a_speculators_config():
    hf: dict = {}
    update_dneo(_speculators_config(), hf)
    assert hf["architectures"] == ["DNeoDraftModel"]
    # the layer ids are the target's attention inputs, used as is
    assert hf["eagle_aux_hidden_state_layer_ids"] == [0, 17, 21, 24, 28, 31, 35]
    assert hf["dflash_config"]["causal"] is False
    assert hf["dflash_config"]["conv_kernel_size"] == 2
    assert hf["dflash_config"]["sample_from_anchor"] is False
    assert hf["dneo_config"] == {
        "prefix_only_attention": True,
        "rejection_memory": True,
        "causal_head_rank": 256,
        "causal_head_folded_rank": 128,
    }
    assert hf["block_size"] == 16


@pytest.mark.parametrize(
    "overrides",
    [
        {"feature_encoder": "diagonal"},
        {"shared_feature_encoder": False},
        {"feature_source": "residual"},
        {"causal_head_rank": 0},
    ],
)
def test_update_dneo_rejects_unsupported_variants(overrides):
    with pytest.raises(ValueError):
        update_dneo(_speculators_config(**overrides), {})


@pytest.mark.parametrize(
    ("requested", "expected"), [(None, 15), (7, 7), (16, ValueError)]
)
def test_num_speculative_tokens_at_most_the_block_minus_the_anchor(requested, expected):
    hf_config = SimpleNamespace(
        block_size=16, dflash_config={"sample_from_anchor": False}
    )
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=hf_config),
        num_speculative_tokens=requested,
    )
    if expected is ValueError:
        with pytest.raises(ValueError):
            DNeoSpeculatorConfig.resolve_num_speculative_tokens(spec)
        return
    DNeoSpeculatorConfig.resolve_num_speculative_tokens(spec)
    assert spec.num_speculative_tokens == expected
