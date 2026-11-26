# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone DFly2 configuration and sequential attentive correction."""

import json
from copy import copy, deepcopy

import pytest
import torch
import torch.nn.functional as F

from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="DeepSeek-V4.1 DFly2 on Hopper",
)


@pytest.fixture
def checkpoint_configs(tmp_path):
    text = dict(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=40,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=32,
        o_groups=1,
        moe_intermediate_size=256,
        n_routed_experts=384,
        num_experts_per_tok=6,
        n_shared_experts=1,
        hidden_act="silu",
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        topk_method="noaux_tc",
        routed_scaling_factor=1.5,
        rms_norm_eps=1e-20,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        sliding_window=128,
        index_topk=512,
        max_position_embeddings=1048576,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_parameters=dict(
            rope_type="yarn",
            rope_theta=10000,
            factor=16,
            beta_fast=32,
            beta_slow=1,
            original_max_position_embeddings=65536,
        ),
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    )
    target = dict(
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        dtype="bfloat16",
        text_config=text,
    )
    draft = deepcopy(target)
    draft["architectures"] = ["DFlyDeepseekV41DraftModel"]
    draft["dfly_config"] = dict(
        block_size=6,
        causal_block=False,
        target_kv_layers=[28, 32, 36],
        target_index_layers=[28, 32, 36],
        index_topk=512,
        mask_token_id=255,
        mtp_fusion="slot0",
        dense_dtype="bfloat16",
        rope=dict(
            theta=160000, original_seq_len=65536, factor=16, beta_fast=32, beta_slow=1
        ),
        hidden_correction=dict(intermediate_size=128),
        attentive_correction=dict(
            source_layer=-1,
            block_keys=True,
            block_mode="gated",
            block_kv_in="hidden_embed",
            max_block_slots=16,
        ),
    )
    target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target_path.mkdir()
    draft_path.mkdir()
    (target_path / "config.json").write_text(json.dumps(target))
    (draft_path / "config.json").write_text(json.dumps(draft))
    return target_path, draft_path


def _engine_config(paths, **spec_options):
    target, draft = paths
    return EngineArgs(
        model=str(target),
        dtype="bfloat16",
        max_model_len=1024,
        skip_tokenizer_init=True,
        load_format="dummy",
        max_num_seqs=4,
        max_num_batched_tokens=128,
        enforce_eager=True,
        speculative_config={"method": "dfly", "model": str(draft), **spec_options},
    ).create_engine_config()


def test_standalone_checkpoint_uses_draft_experts_and_normalized_target_hidden(
    checkpoint_configs,
):
    from vllm.models.deepseek_v4.common.spec_decode import needs_mtp_hidden_buffer

    config = _engine_config(checkpoint_configs)
    spec = config.speculative_config
    draft = spec.draft_model_config.hf_config
    assert spec.num_speculative_tokens == 5
    assert (
        draft.num_hidden_layers,
        draft.n_routed_experts,
        draft.num_experts_per_tok,
    ) == (3, 128, 3)
    assert draft.architectures == ["DFlyDeepseekV41DraftModel"]
    assert spec.draft_model_config.get_num_experts() == 128
    assert spec.draft_model_config.get_num_experts_per_tok() == 3
    assert spec.draft_model_config.get_num_layers(spec.draft_parallel_config) == 3
    assert not needs_mtp_hidden_buffer(spec)
    assert config.model_config.hf_config.n_routed_experts == 384


@pytest.mark.parametrize(
    "change,match",
    [
        ({"block_size": 17}, "block_size"),
        ({"target_index_layers": [28, 32, 35]}, "index-source"),
        ({"target_kv_layers": [14, 32, 36]}, "ratio-1"),
        ({"causal_block": True}, "noncausal"),
    ],
)
def test_incompatible_checkpoint_is_rejected_before_loading(
    checkpoint_configs, change, match
):
    path = checkpoint_configs[1] / "config.json"
    data = json.loads(path.read_text())
    data["dfly_config"].update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        _engine_config(checkpoint_configs)


def test_b6_requires_five_proposals(checkpoint_configs):
    with pytest.raises(ValueError, match="block_size - 1"):
        _engine_config(checkpoint_configs, num_speculative_tokens=6)


def _norm(x, weight, eps):
    x = x.float()
    return (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight
    ).bfloat16()


def _rotate(x, positions, cache, inverse=False):
    cos, sin = cache[positions].chunk(2, dim=-1)
    if inverse:
        sin = -sin
    pairs = x[..., -64:].float().reshape(*x.shape[:-1], 32, 2)
    real, imag = pairs.unbind(-1)
    tail = torch.stack(
        (
            real * cos[:, None] - imag * sin[:, None],
            real * sin[:, None] + imag * cos[:, None],
        ),
        dim=-1,
    )
    return torch.cat((x[..., :-64], tail.flatten(-2).bfloat16()), -1)


@torch.inference_mode()
def test_attentive_correction_matches_causal_block_reference(
    checkpoint_configs, dist_init
):
    """The block read includes slot zero, itself, and the sampled previous token.

    Compare each sequential correction with the paper's dense attention formula,
    including separate context/block sinks, relative bias, gate and inverse RoPE.
    """
    from vllm.models.deepseek_v41.nvidia.dfly import DFlyContext, DFlyHiddenCorrection

    config = _engine_config(checkpoint_configs)
    config = copy(config)
    config.model_config = config.speculative_config.draft_model_config
    torch.manual_seed(17)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        correction = DFlyHiddenCorrection(config, "correction")
        context = DFlyContext(config)
        correction.attention.context = context
        for name, param in correction.named_parameters():
            if name.endswith("norm.weight"):
                param.fill_(1)
            else:
                param.normal_(0, 0.05)
        batch, width, heads, hidden_size = 2, 6, 8, 128
        hidden = torch.randn(batch, width, hidden_size)
        embeddings = torch.randn_like(hidden)
        positions = torch.arange(width)[None] + torch.tensor([[1000], [7]])
        context.kv.normal_()
        # An empty target context still has a valid block correction.
        context.indices[0, :4] = torch.tensor([0, 2, 7, 15], dtype=torch.int32)
        block = torch.empty(batch, width, 512)
        expected_keys = []
        attn = correction.attention
        cache = attn.rotary_emb.cos_sin_cache
        eps = config.model_config.hf_config.rms_norm_eps
        for slot in range(width):
            h = hidden[:, slot]
            e = embeddings[:, slot]
            combined = torch.cat(
                (
                    _norm(h, correction.hidden_norm.weight, eps),
                    _norm(e, correction.embed_norm.weight, eps),
                ),
                -1,
            )
            key = _norm(F.linear(combined, attn.kv_in.weight), attn.kv_norm.weight, eps)
            key = _rotate(key[:, None], positions[:, slot], cache).squeeze(1)
            expected_keys.append(key)
            if slot == 0:
                block[:, 0] = attn.block_kv(correction.combine(h, e), positions[:, 0])
                continue
            actual = correction(h, e, positions[:, slot], block, slot)
            q = _norm(F.linear(combined, attn.q_in.weight), attn.q_norm.weight, eps)
            q = F.linear(q, attn.wq_b.weight).view(batch, heads, 512)
            q = _rotate(q, positions[:, slot], cache)
            ctx = torch.zeros_like(q)
            keys = context.kv[0, [0, 2, 7, 15]].float()
            scores = q[0].float() @ keys.T / 512**0.5
            probs = torch.softmax(torch.cat((scores, attn.attn_sink[:, None]), -1), -1)[
                :, :-1
            ]
            ctx[0] = (probs @ keys).bfloat16()
            keys = torch.stack(expected_keys, dim=1).float()
            scores = torch.einsum("bhd,bkd->bhk", q.float(), keys) / 512**0.5
            scores += attn.block_rel_bias[:, torch.arange(slot, -1, -1)]
            sink = attn.block_sink[None, :, None].expand(batch, -1, -1)
            probs = torch.softmax(torch.cat((scores, sink), -1), -1)[..., :-1]
            read = torch.einsum("bhk,bkd->bhd", probs, keys)
            gate = torch.tanh(
                F.linear(combined, attn.block_gate.weight).float()
                + attn.block_gate_bias[:, slot]
            )
            read = (ctx.float() + read * gate[:, :, None]).bfloat16()
            read = _rotate(read, positions[:, slot], cache, inverse=True)
            attentive = F.linear(
                F.linear(read.flatten(1), attn.wo_a.weight), attn.wo_b.weight
            )
            gate, up = F.linear(combined, correction.gate_up_proj.weight).chunk(2, -1)
            delta = F.linear(
                (F.silu(gate.float()) * up.float()).bfloat16(),
                correction.down_proj.weight,
            )
            expected = h + (delta + attentive)
            torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
            torch.testing.assert_close(
                block[:, slot], expected_keys[-1], atol=0.016, rtol=0.016
            )
