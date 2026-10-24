# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.funasr import (
    FunASRModel,
    MultiHeadedAttentionSANM,
    Transformer,
    _load_flash_varlen,
    clip_fake_token_prefixes,
    pack_olens,
    sanm_masked_context,
    unpack_olens,
)
from vllm.model_executor.models.granite_speech import (
    GraniteSpeechConformerAttention,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def tp1_cpu(tmp_path, default_vllm_config):
    """Initialize a temporary single-rank CPU tensor-parallel group."""
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path / 'distributed'}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.fixture
def flash_varlen_spy():
    previous = getattr(_load_flash_varlen, "fn", None)
    missing = not hasattr(_load_flash_varlen, "fn")
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return None

    _load_flash_varlen.fn = spy
    try:
        yield calls
    finally:
        if missing:
            del _load_flash_varlen.fn
        else:
            _load_flash_varlen.fn = previous


def test_clip_fake_token_prefixes_clamps_each_clip():
    hidden = torch.arange(2 * 80 * 2, dtype=torch.float32).reshape(2, 80, 2)
    olens = [80, 50]
    clips = clip_fake_token_prefixes(hidden, olens, [10, 100])
    assert isinstance(clips, list)
    assert [clip.shape[0] for clip in clips] == [10, 50]
    torch.testing.assert_close(clips[0], hidden[0, :10])
    torch.testing.assert_close(clips[1], hidden[1, :50])

    unclipped = clip_fake_token_prefixes(hidden, olens, None)
    assert [clip.shape[0] for clip in unclipped] == [80, 50]
    torch.testing.assert_close(unclipped[1], hidden[1, :50])


def test_pack_olens_drops_empty_rows_from_cu_seqlens():
    hidden = torch.randn(3, 10, 4)
    olens = torch.tensor([10, 0, 4])
    packed, cu_seqlens, max_seqlen, lengths = pack_olens(hidden, olens)
    assert lengths == [10, 0, 4]
    assert cu_seqlens.tolist() == [0, 10, 14]
    assert int(max_seqlen) == 10
    assert packed.shape[:2] == (1, 14)
    restored = unpack_olens(packed, lengths, hidden.size(1), hidden)
    torch.testing.assert_close(restored[0, :10], hidden[0, :10])
    torch.testing.assert_close(restored[2, :4], hidden[2, :4])
    assert torch.count_nonzero(restored[1]) == 0
    assert torch.count_nonzero(restored[2, 4:]) == 0


def test_get_encoder_outputs_merges_fake_token_lengths():
    model = FunASRModel.__new__(FunASRModel)

    def audio_encoder(speech, speech_lengths):
        return speech, speech_lengths

    def audio_adaptor(hidden, ilens):
        return hidden, ilens

    model.encoder = SimpleNamespace(
        audio_encoder=audio_encoder,
        audio_adaptor=audio_adaptor,
    )
    speech = torch.randn(2, 80, 4)
    lengths = torch.tensor([80, 50])
    clips = model.get_encoder_outputs(speech, lengths, torch.tensor([10, 6]))
    assert isinstance(clips, list)
    assert [clip.shape[0] for clip in clips] == [10, 6]
    torch.testing.assert_close(clips[0], speech[0, :10])
    torch.testing.assert_close(clips[1], speech[1, :6])

    olens_only = model.get_encoder_outputs(speech, lengths, None)
    assert [clip.shape[0] for clip in olens_only] == [80, 50]


def test_funasr_adaptor_attends_on_olens_and_merges_fake_lengths(tp1_cpu):
    torch.manual_seed(0)
    adaptor = Transformer(
        downsample_rate=1,
        encoder_dim=8,
        llm_dim=16,
        ffn_dim=32,
        prefix="adaptor",
        n_layer=1,
        attention_heads=4,
    )
    assert adaptor.k == 1
    hidden = torch.randn(2, 80, 8)
    ilens = torch.tensor([80, 50])
    out, olens = adaptor(hidden, ilens)
    assert olens.tolist() == [80, 50]

    clips = clip_fake_token_prefixes(out, olens, torch.tensor([10, 6]))
    assert [clip.shape[0] for clip in clips] == [10, 6]

    for index, length in enumerate(ilens.tolist()):
        alone, alone_olens = adaptor(
            hidden[index : index + 1, :length], ilens[index : index + 1]
        )
        assert int(alone_olens[0]) == length
        torch.testing.assert_close(out[index, :length], alone[0, :length])

    short, _ = adaptor(hidden[:1, :10], torch.tensor([10]))
    # Attention ran on all 80 frames. Truncating to 10 first is a different result.
    assert (out[0, :10] - short[0]).abs().max() > 1e-3


def _monolithic_sanm(attn, hidden, mask, mask_att=None):
    q_h, k_h, v_h, value = attn.forward_qkv(hidden)
    fsmn = attn.forward_fsmn(value, mask, None)
    scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * (attn.d_k**-0.5)
    return attn.forward_attention(v_h, scores, mask, mask_att) + fsmn


def test_sanm_scales_once_and_keeps_length_zero_finite(tp1_cpu, flash_varlen_spy):
    torch.manual_seed(4)
    attn = MultiHeadedAttentionSANM(
        n_head=4, in_feat=16, n_feat=16, kernel_size=3, sanm_shift=0
    )
    value = torch.randn(1, 2, 5, 4)
    scores = torch.randn(1, 2, 3, 5)
    context = sanm_masked_context(value, scores, torch.zeros(1, 3, 5))
    assert torch.isfinite(context).all()
    assert torch.count_nonzero(context) == 0

    hidden = torch.randn(3, 8, 16)
    mask = torch.zeros(3, 1, 8)
    lengths = [4, 0, 7]
    for index, length in enumerate(lengths):
        mask[index, :, :length] = 1
    flash_varlen_spy["n"] = 0
    output = attn(hidden, mask)
    expected = _monolithic_sanm(attn, hidden, mask)
    assert flash_varlen_spy["n"] == 1
    torch.testing.assert_close(output, expected)
    for index, length in enumerate(lengths):
        assert torch.isfinite(output[index]).all()
        if length:
            torch.testing.assert_close(output[index, :length], expected[index, :length])

    long_hidden = torch.randn(2, 130, 16)
    long_mask = torch.ones(2, 1, 130)
    long_mask[:, :, 100:] = 0
    flash_varlen_spy["n"] = 0
    long_out = attn(long_hidden, long_mask)
    long_expected = _monolithic_sanm(attn, long_hidden, long_mask)
    assert flash_varlen_spy["n"] == 1
    torch.testing.assert_close(long_out[:, :100], long_expected[:, :100])


def test_sanm_full_pattern_does_not_call_varlen_flash(tp1_cpu, flash_varlen_spy):
    torch.manual_seed(6)
    attn = MultiHeadedAttentionSANM(
        n_head=4, in_feat=16, n_feat=16, kernel_size=3, sanm_shift=0
    )
    hidden = torch.randn(2, 12, 16)
    mask = torch.ones(2, 1, 12)
    pattern = torch.tril(torch.ones(2, 12, 12))
    flash_varlen_spy["n"] = 0
    output = attn(hidden, mask, mask_att_chunk_encoder=pattern)
    expected = _monolithic_sanm(attn, hidden, mask, pattern)
    assert flash_varlen_spy["n"] == 0
    torch.testing.assert_close(output, expected)


def _legacy_blocked_shaw(attn, hidden_states, attention_dists):
    """Today's one blocked SDPA, including the finfo-max remainder fill."""
    hidden_states = attn.pre_norm(hidden_states)
    batch, num_features, _ = hidden_states.shape
    num_blocks = math.ceil(num_features / attn.context_size)
    remainder = num_features % attn.context_size
    if remainder > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, attn.context_size - remainder))
    query_states = attn.to_q(hidden_states)
    key_states, value_states = attn.to_kv(hidden_states).chunk(2, dim=-1)
    query_states = query_states.reshape(
        batch, num_blocks, attn.context_size, attn.num_heads, -1
    ).transpose(2, 3)
    key_states = key_states.reshape(
        batch, num_blocks, attn.context_size, attn.num_heads, -1
    ).transpose(2, 3)
    value_states = value_states.reshape(
        batch, num_blocks, attn.context_size, attn.num_heads, -1
    ).transpose(2, 3)
    rel_pos_emb = attn.rel_pos_emb(attention_dists.to(hidden_states.device))
    pos_attn = torch.einsum("bnhid,ijd->bnhij", query_states, rel_pos_emb) * attn.scale
    if remainder > 0:
        invalid = torch.ones(
            attn.context_size,
            attn.context_size,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        invalid[:remainder, :remainder] = False
        pos_attn[:, -1].masked_fill_(invalid, -torch.finfo(pos_attn.dtype).max)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=pos_attn,
            scale=attn.scale,
        )
    out = out.transpose(2, 3).reshape(batch, hidden_states.shape[1], -1)
    return attn.to_out(out[:, :num_features])


@pytest.mark.parametrize(
    ("num_features", "dtype"),
    [
        (16, torch.float32),
        (11, torch.float32),
        (16, torch.bfloat16),
        (11, torch.bfloat16),
    ],
)
def test_granite_per_block_shaw_matches_blocked_sdpa(monkeypatch, num_features, dtype):
    torch.manual_seed(9)
    context_size = 8
    config = SimpleNamespace(
        dim_head=8,
        num_heads=2,
        hidden_dim=16,
        max_pos_emb=16,
        context_size=context_size,
    )
    attn = GraniteSpeechConformerAttention(config).to(dtype)
    seq = torch.arange(context_size)
    dists = (
        torch.clamp(seq.view(-1, 1) - seq.view(1, -1), -context_size, context_size)
        + config.max_pos_emb
    )
    hidden = torch.randn(2, num_features, config.hidden_dim, dtype=dtype)
    expected = _legacy_blocked_shaw(attn, hidden, dists)

    masks: list[torch.Tensor] = []
    backends: list[object] = []
    original_sdpa = F.scaled_dot_product_attention
    original_kernel = torch.nn.attention.sdpa_kernel

    def tracked_kernel(*args, **kwargs):
        backend = args[0] if args else kwargs.get("backends")
        backends.append(backend)
        assert "FLASH" not in str(backend)
        assert backend == torch.nn.attention.SDPBackend.MATH
        return original_kernel(*args, **kwargs)

    def tracked_sdpa(query, key, value, attn_mask=None, **kwargs):
        assert query.shape[-2] == context_size
        if attn_mask is not None:
            masks.append(attn_mask.detach().clone())
        return original_sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", tracked_kernel)
    monkeypatch.setattr(F, "scaled_dot_product_attention", tracked_sdpa)
    actual = attn(hidden, dists)

    torch.testing.assert_close(actual, expected)
    num_blocks = math.ceil(num_features / context_size)
    assert len(masks) == num_blocks
    assert len(backends) == num_blocks
    remainder = num_features % context_size
    if remainder:
        last = masks[-1]
        fill = -torch.finfo(last.dtype).max
        assert torch.equal(
            last[..., remainder:, :],
            torch.full_like(last[..., remainder:, :], fill),
        )
        assert torch.equal(
            last[..., :remainder, remainder:],
            torch.full_like(last[..., :remainder, remainder:], fill),
        )
        inside = last[..., :remainder, :remainder]
        assert not torch.equal(inside, torch.full_like(inside, fill))
