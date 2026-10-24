# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from contextlib import contextmanager

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    bucket_flashinfer_max_seqlen,
)
from vllm.model_executor.models import ultravox as ultravox_mod
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.model_executor.models.ultravox import (
    StackAudioFrames,
    UltravoxFeedForwardProjector,
    UltravoxModel,
    UltravoxTransformerProjector,
    UltravoxWhisperEncoder,
    _build_cpu_pack_plan,
    _build_packed_attn_metadata,
    _counts_from_audio_lens,
    _feat_len,
    _prepare_flashinfer_cu_seqlens,
    _stack_packed_frames,
    _upper_bound_mm_lora_counts,
    _zero_padded_conv1_tail,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.core_model


@contextmanager
def _forbid_host_readback():
    """Raise if a tensor is copied back with ``.cpu()`` or ``.numpy()``."""
    original_cpu = torch.Tensor.cpu
    original_numpy = torch.Tensor.numpy

    def _cpu(self, *args, **kwargs):
        raise RuntimeError("cpu readback")

    def _numpy(self, *args, **kwargs):
        raise RuntimeError("numpy readback")

    torch.Tensor.cpu = _cpu
    torch.Tensor.numpy = _numpy
    try:
        yield
    finally:
        torch.Tensor.cpu = original_cpu
        torch.Tensor.numpy = original_numpy


class _AudioConfig:
    max_source_positions = 1500


class _Config:
    def __init__(self, max_source_positions=1500, stack_factor=8):
        self.audio_config = _AudioConfig()
        self.audio_config.max_source_positions = max_source_positions
        self.stack_factor = stack_factor


class _FieldElem:
    def __init__(self, data):
        self.field = object()
        self.data = data


class _Attn:
    def __init__(self, backend):
        self.attn_backend = backend
        self.fp8_enabled = False
        self.num_heads = 4
        self.head_size = 64


class _RecLayer(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.self_attn = type("_Self", (), {"attn": _Attn(backend)})()
        self.meta = None

    def forward(
        self,
        hidden,
        cu_seqlens=None,
        max_seqlen=None,
        sequence_lengths=None,
    ):
        self.meta = {
            "cu": cu_seqlens,
            "max": max_seqlen,
            "seq": sequence_lengths,
            "rows": hidden.shape[0],
        }
        return hidden


def _count_model(max_source_positions=1500, stack_factor=8):
    model = UltravoxModel.__new__(UltravoxModel)
    model.config = _Config(max_source_positions, stack_factor)
    return model


def _make_convs(n_mels, channels, generator):
    conv1 = nn.Conv1d(n_mels, channels, kernel_size=3, padding=1)
    conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
    with torch.no_grad():
        conv1.weight.copy_(torch.randn(conv1.weight.shape, generator=generator))
        conv1.bias.fill_(0.4)
        conv2.weight.copy_(torch.randn(conv2.weight.shape, generator=generator))
        conv2.bias.fill_(-0.3)
    return conv1, conv2


def _conv_out(conv1, conv2, mel, lens, mode):
    hidden = F.gelu(conv1(mel))
    if mode == "ge":
        hidden = _zero_padded_conv1_tail(hidden, lens)
        return F.gelu(conv2(hidden))
    if mode == "gt":
        positions = torch.arange(hidden.shape[-1])
        mask = positions.unsqueeze(0) > lens.reshape(-1).to(torch.int64).unsqueeze(1)
        hidden = hidden.masked_fill(mask.unsqueeze(1), 0)
        return F.gelu(conv2(hidden))
    if mode == "none":
        return F.gelu(conv2(hidden))
    if mode == "conv2":
        hidden = F.gelu(conv2(hidden))
        feat = _feat_len_tensor(lens)
        positions = torch.arange(hidden.shape[-1])
        mask = positions.unsqueeze(0) >= feat.unsqueeze(1)
        return hidden.masked_fill(mask.unsqueeze(1), 0)
    raise AssertionError(mode)


def _feat_len_tensor(lens):
    lens = lens.reshape(-1).to(torch.int64)
    return (lens - 1) // 2 + 1


def _unpadded_conv(conv1, conv2, row, length):
    clip = row[:, :length].unsqueeze(0)
    return F.gelu(conv2(F.gelu(conv1(clip))))


def _make_encoder(n_mels=8, channels=16, max_pos=1500, backend=None):
    if backend is None:
        backend = AttentionBackendEnum.TORCH_SDPA
    encoder = UltravoxWhisperEncoder.__new__(UltravoxWhisperEncoder)
    nn.Module.__init__(encoder)
    generator = torch.Generator().manual_seed(0)
    conv1, conv2 = _make_convs(n_mels, channels, generator)
    encoder.conv1 = conv1
    encoder.conv2 = conv2
    encoder.total_stride = 2
    encoder.max_source_positions = max_pos
    encoder.embed_positions = nn.Embedding(max_pos, channels)
    encoder.layer_norm = nn.Identity()
    layer = _RecLayer(backend)
    encoder.layers = nn.ModuleList([layer])
    return encoder, layer


def _reference_packed(encoder, mel, lengths):
    lens = torch.tensor(lengths)
    hidden = _conv_out(encoder.conv1, encoder.conv2, mel, lens, "ge")
    hidden = hidden.permute(0, 2, 1)
    t_conv = hidden.shape[-2]
    hidden = hidden + encoder.embed_positions.weight[:t_conv]
    plan = _build_cpu_pack_plan(lens, mel.shape[-1])
    flat = hidden.reshape(-1, hidden.shape[-1])
    return flat.index_select(0, plan.pack_index), plan


class _Tower:
    dtype = torch.float32

    def __init__(self):
        self.calls = []

    def __call__(self, features, lens, pack):
        self.calls.append((features.shape, None if pack is None else pack))
        if pack is None:
            return features.new_zeros((features.shape[0], features.shape[-1], 4))
        return features.new_zeros((pack.pack_index.numel(), 4))


class _Projector:
    def __init__(self, stack_factor=8):
        self.stack_factor = stack_factor
        self.seen = []

    def __call__(self, hidden, token_len, feat_lens=None):
        self.seen.append((tuple(hidden.shape), feat_lens, token_len))
        if feat_lens is None:
            return hidden
        rows = sum(
            (int(feat) + self.stack_factor - 1) // self.stack_factor
            for feat in feat_lens
        )
        return hidden.new_zeros((rows, 3))


def _wire_tower(model, stack_factor=8):
    model.tower_connector_lora = True
    model.audio_tower = _Tower()
    model.multi_modal_projector = _Projector(stack_factor)
    return model


def test_init_and_runtime_lora_counts_use_three_formulas():
    # ceil(3008/188)=16, max(16*1500, 3008*8)=24064, connector 16*188=3008.
    # 24064 is 64 above 16*1500 because 188*8=1504 > 1500.
    # ceil(8192/188)=44, max(44*1500, 8192*8)=66000, connector 44*188=8272.
    assert _upper_bound_mm_lora_counts(3008, 1500, 8) == (24064, 3008)
    assert _upper_bound_mm_lora_counts(8192, 1500, 8) == (66000, 8272)
    # Both tower terms: short budget uses num_mm_embeds * stack_factor,
    # long budget uses num_chunks * max_source_positions.
    assert _upper_bound_mm_lora_counts(5, 10, 3) == (20, 8)

    model = _count_model()
    assert model.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=None, num_mm_embeds=3008
    ) == (24064, 3008)
    assert model.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=None, num_mm_embeds=8192
    ) == (66000, 8272)

    odd = {"audio_lens": torch.tensor([17])}
    assert model.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=odd, num_mm_embeds=16
    ) == (9, 2)
    assert model.get_mm_lora_token_counts(
        modality="audio",
        mm_kwargs={"audio_lens": _FieldElem(torch.tensor([17]))},
        num_mm_embeds=3008,
    ) == (9, 2)
    assert _counts_from_audio_lens(torch.tensor([1, 1]), 8) == (2, 2)
    assert (
        model.get_mm_lora_token_counts(
            modality="audio",
            mm_kwargs={"audio_lens": torch.tensor([17])},
            num_mm_embeds=16,
        )[1]
        == 2
    )

    other = _count_model(max_source_positions=10, stack_factor=4)
    assert other.get_mm_lora_token_counts(
        modality="audio",
        mm_kwargs={"audio_lens": torch.tensor([17])},
        num_mm_embeds=99,
    ) == (9, 3)

    missing = model.get_mm_lora_token_counts(
        modality="audio", mm_kwargs={}, num_mm_embeds=16
    )
    assert missing == (0, 0)
    assert missing[1] is not None

    with pytest.raises(RuntimeError, match="host audio_lens"):
        model.get_mm_lora_token_counts(
            modality="audio",
            mm_kwargs={"audio_lens": torch.tensor([17], device="meta")},
            num_mm_embeds=16,
        )


def test_embed_only_keeps_a_later_connector_map():
    model = _count_model()
    items = [
        (4, {}, 8),
        (9, {"audio_lens": torch.tensor([17])}, 16),
    ]
    connector_ids = []
    for lora_id, kwargs, num_embeds in items:
        _tower, connector = model.get_mm_lora_token_counts(
            modality="audio", mm_kwargs=kwargs, num_mm_embeds=num_embeds
        )
        assert connector is not None
        connector_ids.extend([lora_id] * connector)
    assert connector_ids == [9, 9]


def test_lora_ids_follow_item_order_and_one_tower_pass():
    model = _wire_tower(_count_model())
    model.tower_connector_lora = True
    items = [(3, torch.tensor([17])), (7, torch.tensor([3000]))]
    tower_ids = []
    connector_ids = []
    for lora_id, lens in items:
        tower, connector = model.get_mm_lora_token_counts(
            modality="audio",
            mm_kwargs={"audio_lens": lens},
            num_mm_embeds=16,
        )
        tower_ids.extend([lora_id] * tower)
        connector_ids.extend([lora_id] * connector)
    assert tower_ids == [3] * 9 + [7] * 1500
    assert connector_ids == [3] * 2 + [7] * 188

    features = torch.zeros(2, 80, 3000)
    token_len = torch.tensor([2, 188])
    hidden = model._audio_features_to_embeddings(
        features, torch.tensor([17, 3000]), token_len
    )
    assert len(model.audio_tower.calls) == 1
    shape, pack = model.audio_tower.calls[0]
    assert shape[-1] == 3000
    assert pack.positive_feat_lens == [9, 1500]
    assert pack.pack_index.numel() == 1509
    assert torch.equal(pack.pack_index[:9], torch.arange(9))
    assert torch.equal(pack.pack_index[9:], torch.arange(1500, 3000))
    assert model.multi_modal_projector.seen[0][0] == (1509, 4)
    assert model.multi_modal_projector.seen[0][1] == [9, 1500]
    assert hidden.shape == (190, 3)

    split = model._process_audio_input(
        {
            "type": "audio_features",
            "data": [torch.zeros(1, 80, 17), torch.zeros(1, 80, 3000)],
            "lens": torch.tensor([17, 3000]),
            "token_len": token_len,
            "num_chunks": torch.tensor([1, 1]),
        }
    )
    assert [int(part.shape[0]) for part in split] == [2, 188]

    model.audio_tower.calls.clear()
    short = model._audio_features_to_embeddings(
        torch.zeros(2, 80, 31), torch.tensor([17, 31]), torch.tensor([2, 2])
    )
    assert len(model.audio_tower.calls) == 1
    assert model.audio_tower.calls[0][0][-1] == 31
    assert model.audio_tower.calls[0][1].positive_feat_lens == [9, 16]
    assert short.shape[0] == 2 + 2

    wide = _wire_tower(_count_model())
    wide.tower_connector_lora = False
    wide._audio_features_to_embeddings(
        torch.zeros(17, 4, 2), torch.ones(17, dtype=torch.long), torch.ones(17)
    )
    assert [call[0][0] for call in wide.audio_tower.calls] == [16, 1]
    wide.tower_connector_lora = True
    wide.audio_tower.calls.clear()
    wide._audio_features_to_embeddings(
        torch.zeros(17, 4, 2), torch.ones(17, dtype=torch.long), torch.ones(17)
    )
    assert [call[0][0] for call in wide.audio_tower.calls] == [17]


def test_conv1_zero_matches_unpadded_for_any_right_pad():
    generator = torch.Generator().manual_seed(0)
    conv1, conv2 = _make_convs(4, 8, generator)
    lengths = (1, 2, 16, 17, 31, 3000)
    for length in lengths:
        row = torch.randn(4, 3000, generator=generator)
        for pad in (length, min(length + 1, 3000), 3000):
            mel = torch.zeros(1, 4, pad)
            mel[0, :, :length] = row[:, :length]
            lens = torch.tensor([length])
            feat = _feat_len(length)
            reference = _unpadded_conv(conv1, conv2, row, length)
            got = _conv_out(conv1, conv2, mel, lens, "ge")
            # Long convolutions are not bit-exact against a shorter clip.
            torch.testing.assert_close(got[..., :feat], reference, rtol=1e-4, atol=1e-4)
            if length % 2 == 0:
                torch.testing.assert_close(
                    _conv_out(conv1, conv2, mel, lens, "none")[..., :feat],
                    reference,
                    rtol=1e-4,
                    atol=1e-4,
                )
            if length % 2 == 1 and pad > length:
                assert not torch.allclose(
                    _conv_out(conv1, conv2, mel, lens, "gt")[..., :feat],
                    reference,
                    rtol=1e-4,
                    atol=1e-4,
                )
                assert not torch.allclose(
                    _conv_out(conv1, conv2, mel, lens, "conv2")[..., :feat],
                    reference,
                    rtol=1e-4,
                    atol=1e-4,
                )


def test_packed_attention_matches_each_unpadded_chunk():
    generator = torch.Generator().manual_seed(1)
    conv1, conv2 = _make_convs(4, 8, generator)
    lengths = [17, 64]
    width = 3000
    mel = torch.zeros(2, 4, width)
    rows = []
    for index, length in enumerate(lengths):
        row = torch.randn(4, length, generator=generator)
        rows.append(row)
        mel[index, :, :length] = row
    lens = torch.tensor(lengths)
    plan = _build_cpu_pack_plan(lens, width)
    packed_conv = _conv_out(conv1, conv2, mel, lens, "ge").permute(0, 2, 1)
    flat = packed_conv.reshape(-1, packed_conv.shape[-1])
    packed = flat.index_select(0, plan.pack_index)

    def attend(frames):
        return F.scaled_dot_product_attention(
            frames.unsqueeze(0), frames.unsqueeze(0), frames.unsqueeze(0)
        ).squeeze(0)

    pieces = packed.split(plan.positive_feat_lens, dim=0)
    batched = torch.cat([attend(piece) for piece in pieces], dim=0)
    offset = 0
    for index, length in enumerate(lengths):
        feat = _feat_len(length)
        alone_frames = _unpadded_conv(conv1, conv2, rows[index], length)
        alone_frames = alone_frames.permute(0, 2, 1).squeeze(0)
        alone = attend(alone_frames)
        torch.testing.assert_close(pieces[index], alone_frames, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            batched[offset : offset + feat], alone, rtol=1e-4, atol=1e-4
        )
        offset += feat

    today = _conv_out(conv1, conv2, mel, lens, "none").permute(0, 2, 1)
    today_odd = today[0, : _feat_len(lengths[0])]
    assert not torch.allclose(
        attend(today_odd),
        batched[: _feat_len(lengths[0])],
        rtol=1e-4,
        atol=1e-4,
    )


def test_partial_stack_zero_fills_and_does_not_mix_the_next_frames():
    stack = StackAudioFrames(8)
    cases = ((1, 1), (9, 1), (1500, 4))
    for feat_len, real_tail in cases:
        frames = torch.arange(feat_len * 3, dtype=torch.float32).reshape(feat_len, 3)
        tokens, conn_lens = _stack_packed_frames(stack, frames, [feat_len])
        assert conn_lens == [(feat_len + 7) // 8]
        last = tokens[-1].reshape(8, 3)
        assert torch.equal(last[:real_tail], frames[-real_tail:])
        assert torch.equal(last[real_tail:], torch.zeros(8 - real_tail, 3))
        garbage = torch.full((8, 3), 9.0)
        wide = stack(torch.cat([frames, garbage], dim=0).unsqueeze(0)).squeeze(0)
        assert not torch.equal(wide[conn_lens[0] - 1], tokens[-1])


def test_encoder_packs_on_cpu_without_readback():
    encoder, layer = _make_encoder()
    mel = torch.randn(2, 8, 64)
    lengths = [17, 31]
    lens = torch.tensor(lengths)
    with _forbid_host_readback():
        packed = encoder(mel, lens)
    reference, plan = _reference_packed(encoder, mel, lengths)
    torch.testing.assert_close(packed, reference)
    assert layer.meta["rows"] == plan.pack_index.numel()
    assert torch.equal(layer.meta["cu"], plan.cu_seqlens)
    assert int(layer.meta["max"]) == plan.max_seqlen == 16
    assert layer.meta["cu"].device.type == "cpu"
    assert layer.meta["max"].device.type == "cpu"
    assert layer.meta["max"].ndim == 0
    assert layer.meta["seq"] is None
    assert ".cpu()" not in inspect.getsource(encoder.forward)
    assert ".numpy()" not in inspect.getsource(encoder.forward)
    assert "gpu_sync_allowed" not in inspect.getsource(encoder.forward)

    with pytest.raises(ValueError, match="or less"):
        encoder(torch.zeros(1, 8, encoder.max_context_length + 1), torch.tensor([1]))
    with pytest.raises(ValueError, match="conv frames"):
        _build_cpu_pack_plan(torch.tensor([10]), 2)


def _identity_pair():
    def _linear(hidden):
        return hidden, None

    return _linear


def _stub_feedforward():
    projector = UltravoxFeedForwardProjector.__new__(UltravoxFeedForwardProjector)
    nn.Module.__init__(projector)
    projector._pad_and_stack = StackAudioFrames(8)
    projector.ln_pre = nn.Identity()
    projector.linear_1 = _identity_pair()
    projector.act = nn.Identity()
    projector.ln_mid = nn.Identity()
    projector.linear_2 = _identity_pair()
    projector.ln_post = nn.Identity()
    return projector


def _stub_transformer(channels=4, d_model=1):
    projector = UltravoxTransformerProjector.__new__(UltravoxTransformerProjector)
    nn.Module.__init__(projector)
    projector._pad_and_stack = StackAudioFrames(8)
    projector.ln_pre = nn.Identity()

    def linear_in(hidden):
        zeros = hidden.new_zeros(*hidden.shape[:-1], d_model)
        return zeros, None

    projector.linear_in = linear_in
    projector.embed_positions = nn.Embedding(32, d_model)
    with torch.no_grad():
        projector.embed_positions.weight.copy_(
            torch.arange(32, dtype=torch.float32).view(32, 1).expand(32, d_model)
        )
    projector.layers = nn.ModuleList([_RecLayer(AttentionBackendEnum.TORCH_SDPA)])
    projector.ln_post = nn.Identity()

    def linear_out(hidden):
        return hidden, None

    projector.linear_out = linear_out
    projector._channels = channels
    return projector


def test_projectors_pack_per_chunk_and_ignore_device_token_len():
    frames = torch.arange(9 * 4, dtype=torch.float32).reshape(9, 4)
    token_len = torch.tensor([2])
    feedforward = _stub_feedforward()
    with _forbid_host_readback():
        projected = feedforward(frames, token_len, feat_lens=[9])
    assert projected.shape == (2, 32)
    last = projected[1].reshape(8, 4)
    assert torch.equal(last[0], frames[8])
    assert torch.equal(last[1:], torch.zeros(7, 4))

    transformer = _stub_transformer()
    packed = torch.randn(1 + 9, 4)
    with _forbid_host_readback():
        encoded = transformer(packed, token_len, feat_lens=[1, 9])
    assert encoded.shape == (3, 1)
    assert torch.equal(encoded.view(-1), torch.tensor([0.0, 0.0, 1.0]))
    meta = transformer.layers[0].meta
    assert torch.equal(meta["cu"], torch.tensor([0, 1, 3], dtype=torch.int32))
    assert int(meta["max"]) == 2
    assert meta["max"].device.type == "cpu"
    assert meta["seq"] is None

    rectangle = torch.randn(2, 9, 4)
    with _forbid_host_readback():
        rectangular = transformer(rectangle, token_len)
    # Stack factor 8 pads 9 frames to 16, so each row has 2 tokens.
    assert transformer.layers[0].meta["rows"] == 4
    assert int(transformer.layers[0].meta["max"]) == 2
    assert rectangular.shape[0] == 2
    assert rectangular.shape[1] == 2


def test_sdpa_metadata_stays_on_cpu_and_flash_attn_is_unscaled():
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    sdpa = _Attn(AttentionBackendEnum.TORCH_SDPA)
    meta = _build_packed_attn_metadata(sdpa, cu, 5, 16, torch.device("cpu"))
    assert meta["sequence_lengths"] is None
    assert meta["cu_seqlens"].device.type == "cpu"
    assert meta["max_seqlen"].device.type == "cpu"
    assert int(meta["max_seqlen"]) == 5
    assert torch.equal(meta["cu_seqlens"], cu)

    flash = _Attn(AttentionBackendEnum.FLASH_ATTN)
    flashed = _build_packed_attn_metadata(flash, cu, 5, 16, torch.device("cpu"))
    assert flashed["sequence_lengths"] is None
    assert int(flashed["max_seqlen"]) == 5
    assert torch.equal(flashed["cu_seqlens"], cu)


def _expected_flashinfer_cu(lengths, scale, *, times_three):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    batch = len(lengths)
    bucket = next(b for b in (8, 16, 32, 64) if b >= batch)
    scaled = [value * scale for value in cu]
    extra = bucket - batch

    def pad(values):
        return values + [values[-1]] * extra

    qk = pad(scaled)
    values = pad([value * 3 for value in scaled]) if times_three else list(qk)
    seq = list(lengths) + [0] * extra
    return qk + values, seq


def test_flashinfer_padding_matches_the_numpy_helper_contract():
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    got, seq = _prepare_flashinfer_cu_seqlens(
        cu,
        hidden_size=16,
        tp_size=1,
        fp8_padded_hidden_size=None,
        device=torch.device("cpu"),
    )
    expect_cu, expect_seq = _expected_flashinfer_cu([3, 5], 16, times_three=True)
    assert got.dtype == torch.int32
    assert seq.dtype == torch.int32
    assert got.tolist() == expect_cu
    assert seq.tolist() == expect_seq
    assert len(seq) == 8
    assert len(got) == 18

    fp8, fp8_seq = _prepare_flashinfer_cu_seqlens(
        cu,
        hidden_size=16,
        tp_size=2,
        fp8_padded_hidden_size=32,
        device=torch.device("cpu"),
    )
    expect_fp8, expect_fp8_seq = _expected_flashinfer_cu([3, 5], 16, times_three=False)
    assert fp8.tolist() == expect_fp8
    assert fp8_seq.tolist() == expect_fp8_seq
    half = len(fp8) // 2
    assert fp8[:half].tolist() == fp8[half:].tolist()

    full = torch.arange(9, dtype=torch.int32)
    full_cu, full_seq = _prepare_flashinfer_cu_seqlens(
        full,
        hidden_size=16,
        tp_size=1,
        fp8_padded_hidden_size=None,
        device=torch.device("cpu"),
    )
    assert len(full_seq) == 8
    assert len(full_cu) == 18
    assert full_seq.tolist() == [1] * 8


def test_flashinfer_metadata_buckets_max_seqlen_on_the_host(monkeypatch):
    monkeypatch.setattr(ultravox_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    seen = {}

    def _fp8(num_heads, head_size):
        seen["num_heads"] = num_heads
        seen["head_size"] = head_size
        return 32

    monkeypatch.setattr(ultravox_mod, "get_fp8_padded_hidden_size", _fp8)
    attn = _Attn(AttentionBackendEnum.FLASHINFER)
    attn.fp8_enabled = True
    attn.num_heads = 4
    attn.head_size = 64
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    meta = _build_packed_attn_metadata(attn, cu, 5, 16, torch.device("cpu"))
    assert seen == {"num_heads": 8, "head_size": 64}
    assert meta["max_seqlen"].device.type == "cpu"
    assert meta["max_seqlen"].ndim == 0
    assert int(meta["max_seqlen"]) == bucket_flashinfer_max_seqlen(5) == 1024
    expect, expect_seq = _expected_flashinfer_cu([3, 5], 16, times_three=False)
    assert meta["cu_seqlens"].tolist() == expect
    assert meta["sequence_lengths"].tolist() == expect_seq


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA length fallback")
def test_cuda_lengths_return_without_readback():
    encoder, layer = _make_encoder()
    encoder.conv1 = encoder.conv1.cuda()
    encoder.conv2 = encoder.conv2.cuda()
    encoder.embed_positions = encoder.embed_positions.cuda()
    mel = torch.randn(2, 8, 64, device="cuda")
    lens = torch.tensor([17, 31], device="cuda")
    with _forbid_host_readback():
        hidden = encoder(mel, lens)
    t_conv = (64 - 1) // 2 + 1
    assert hidden.shape == (2, t_conv, 16)
    assert int(layer.meta["max"]) == t_conv
    assert layer.meta["rows"] == 2 * t_conv
    assert layer.meta["max"].device.type == "cpu"

    feedforward = _stub_feedforward()
    token_len = torch.tensor([2, 2], device="cuda")
    features = torch.randn(2, 9, 4, device="cuda")
    with _forbid_host_readback():
        projected = feedforward(features, token_len)
    assert projected.shape[0] == 2

    transformer = _stub_transformer()
    transformer.embed_positions = transformer.embed_positions.cuda()
    with _forbid_host_readback():
        rectangular = transformer(features, token_len)
    assert rectangular.shape[0] == 2
    assert int(transformer.layers[0].meta["max"]) == rectangular.shape[1]
    assert transformer.layers[0].meta["max"].device.type == "cpu"


def test_ultravox_does_not_opt_into_whisper_encoder_cudagraph():
    assert not issubclass(UltravoxModel, SupportsEncoderCudaGraph)
    assert not issubclass(UltravoxWhisperEncoder, SupportsEncoderCudaGraph)
    assert not hasattr(UltravoxModel, "get_encoder_cudagraph_config")
    assert not hasattr(UltravoxWhisperEncoder, "get_encoder_cudagraph_config")
    assert ".cpu()" not in inspect.getsource(_build_packed_attn_metadata)
    assert ".numpy()" not in inspect.getsource(_prepare_flashinfer_cu_seqlens)
    assert ".cpu()" not in inspect.getsource(_counts_from_audio_lens)
