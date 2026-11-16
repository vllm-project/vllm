# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedConfig

from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.load import LoadConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.models.deepseek_v4.common.spec_decode import needs_mtp_hidden_buffer
from vllm.platforms import current_platform

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


def _tiny_glm_mtp_config(model: str) -> PreTrainedConfig:
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


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Tests the CUDA deepseek_v32 MTP layer"
)
def test_glm52_mtp_eh_proj_gets_low_latency_plan(default_vllm_config):
    from vllm.models.deepseek_v32.nvidia import mtp

    # SpeculativeConfig rewrites the draft model_type; the target keeps GLM's.
    draft_config = mock.MagicMock(
        hidden_size=16, rms_norm_eps=1e-5, index_topk=8, model_type="deepseek_mtp"
    )
    vllm_config = mock.MagicMock()
    vllm_config.speculative_config.draft_model_config.hf_config = draft_config
    vllm_config.model_config.hf_config.model_type = "glm_moe_dsa"
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.scheduler_config.max_num_batched_tokens = 4
    plan = {1: ("cute", None)}

    with (
        mock.patch.object(mtp, "build_glm52_plan", return_value=plan) as build_plan,
        mock.patch.object(mtp, "DeepseekV32DecoderLayer", return_value=nn.Identity()),
        mock.patch.object(mtp, "SharedHead", return_value=nn.Identity()),
        mock.patch.object(mtp, "SparseMLAIndexGroupBuilder"),
        mock.patch.object(mtp, "get_sparse_mla_index_group_max_rows", return_value=4),
        mock.patch.object(mtp.current_platform, "device_type", "cpu"),
    ):
        layer = mtp.DeepseekV32MultiTokenPredictorLayer(vllm_config, "model.layers.78")

    build_plan.assert_called_once_with(layer.eh_proj.weight, torch.bfloat16)
    assert layer._eh_plan is plan


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (None, False),
        ("ngram", False),
        ("mtp", True),
        ("eagle3", True),
        ("draft_model", True),
        ("dspark", False),
    ],
)
def test_deepseek_v4_mtp_hidden_buffer_skips_dspark(method, expected):
    """Only drafters that read DeepSeek V4's pre-hc_head residual get the
    buffer; DSpark drafts from the target's aux hidden states."""
    spec_config = None
    if method is not None:
        spec_config = SpeculativeConfig.__new__(SpeculativeConfig)
        object.__setattr__(spec_config, "method", method)
    assert needs_mtp_hidden_buffer(spec_config) is expected
