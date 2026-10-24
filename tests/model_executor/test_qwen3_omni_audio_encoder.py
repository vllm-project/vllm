# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.models import qwen3_asr, qwen3_omni_moe_thinker
from vllm.v1.attention.backends.registry import AttentionBackendEnum

WINDOW = 100


def _patch_runtime(monkeypatch, tp_size: int) -> None:
    import vllm.model_executor.layers.attention.mm_encoder_attention as mm_attn
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(
        qwen3_omni_moe_thinker,
        "get_tensor_model_parallel_world_size",
        lambda tp=tp_size: tp,
    )
    for mod in (linear, parameter):
        monkeypatch.setattr(
            mod, "get_tensor_model_parallel_world_size", lambda tp=tp_size: tp
        )
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)

    def _backend(*args, **kwargs):
        return AttentionBackendEnum.TORCH_SDPA

    monkeypatch.setattr(qwen3_omni_moe_thinker, "get_vit_attn_backend", _backend)
    monkeypatch.setattr(mm_attn, "get_vit_attn_backend", _backend)


def _audio_config(**overrides) -> Qwen3OmniMoeAudioEncoderConfig:
    fields = dict(
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=20,
        encoder_ffn_dim=40,
        d_model=160,
        activation_function="gelu",
        max_source_positions=64,
        n_window=50,
        n_window_infer=800,
        output_dim=16,
        conv_chunksize=500,
        downsample_hidden_size=4,
    )
    fields.update(overrides)
    return Qwen3OmniMoeAudioEncoderConfig(**fields)


def _build_encoder(monkeypatch, tp_size: int, quant_config=None, **overrides):
    _patch_runtime(monkeypatch, tp_size)
    return qwen3_omni_moe_thinker.Qwen3OmniMoeAudioEncoder(
        _audio_config(**overrides),
        prefix="audio_tower",
        quant_config=quant_config,
    )


class _RecordingQuantConfig(QuantizationConfig):
    def __init__(self) -> None:
        super().__init__()
        self.prefixes: list[str] = []

    def get_name(self) -> str:
        return "fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "_RecordingQuantConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        self.prefixes.append(prefix)
        return UnquantizedLinearMethod()


def _reference_chunk_lengths(lengths: list[int], window: int) -> list[int]:
    """A zero length adds no chunk. A zero remainder is one full window."""
    chunks: list[int] = []
    for length in lengths:
        full, remainder = divmod(length, window)
        chunks.extend([window] * full)
        if remainder:
            chunks.append(remainder)
    return chunks


def _reference_cu(aftercnn: list[int], window_aftercnn: int) -> list[int]:
    segments = [0]
    for cnn_len in aftercnn:
        full, remainder = divmod(cnn_len, window_aftercnn)
        segments.extend([window_aftercnn] * full)
        if remainder:
            segments.append(remainder)
    total = 0
    cu: list[int] = []
    for value in segments:
        total += value
        cu.append(total)
    return cu


def test_tp8_rejects_20_heads_before_qkv(monkeypatch):
    _patch_runtime(monkeypatch, 8)
    constructed: list[str] = []
    real = qwen3_omni_moe_thinker.QKVParallelLinear

    def _wrapped(*args, **kwargs):
        constructed.append(kwargs.get("prefix", ""))
        return real(*args, **kwargs)

    monkeypatch.setattr(qwen3_omni_moe_thinker, "QKVParallelLinear", _wrapped)
    config = SimpleNamespace(encoder_attention_heads=20)
    with pytest.raises(
        ValueError,
        match=("encoder_attention_heads=20 is not divisible by tensor_parallel_size=8"),
    ):
        qwen3_omni_moe_thinker.Qwen3OmniMoeAudioEncoder(config)
    assert constructed == []


def test_tp_shards_heads_when_divisible(monkeypatch):
    tp2 = _build_encoder(monkeypatch, 2)
    attn2 = tp2.layers[0].self_attn
    assert attn2.num_heads == 20
    assert attn2.num_local_heads == 10

    tp4 = _build_encoder(monkeypatch, 4)
    attn4 = tp4.layers[0].self_attn
    assert attn4.num_heads == 20
    assert attn4.num_local_heads == 5
    assert tp4.layers[0].fc1.tp_size == 4


def test_ffn_indivisible_still_asserts(monkeypatch):
    with pytest.raises(AssertionError, match="6 is not divisible by 4"):
        _build_encoder(
            monkeypatch,
            4,
            encoder_attention_heads=4,
            d_model=32,
            encoder_ffn_dim=6,
        )


def test_quant_config_reaches_linears_not_conv_or_norm(monkeypatch):
    quant_config = _RecordingQuantConfig()
    encoder = _build_encoder(monkeypatch, 1, quant_config=quant_config)
    expected = {
        "audio_tower.conv_out",
        "audio_tower.layers.0.self_attn.qkv_proj",
        "audio_tower.layers.0.self_attn.out_proj",
        "audio_tower.layers.0.fc1",
        "audio_tower.layers.0.fc2",
        "audio_tower.proj1",
        "audio_tower.proj2",
    }
    assert set(quant_config.prefixes) == expected
    layer = encoder.layers[0]
    for linear in (
        encoder.conv_out,
        layer.self_attn.qkv_proj,
        layer.self_attn.out_proj,
        layer.fc1,
        layer.fc2,
        encoder.proj1,
        encoder.proj2,
    ):
        assert linear.quant_config is quant_config
    for conv in (encoder.conv2d1, encoder.conv2d2, encoder.conv2d3):
        assert isinstance(conv, nn.Conv2d)
        assert not hasattr(conv, "quant_config")
    for norm in (layer.self_attn_layer_norm, layer.final_layer_norm, encoder.ln_post):
        assert isinstance(norm, nn.LayerNorm)
        assert not hasattr(norm, "quant_config")


def _ignore_config(ignored: list[str]) -> SimpleNamespace:
    quant_config = SimpleNamespace(
        ignored_layers=list(ignored), packed_modules_mapping={}
    )

    def apply_vllm_mapper(mapper):
        quant_config.ignored_layers = mapper.apply_list(quant_config.ignored_layers)

    quant_config.apply_vllm_mapper = apply_vllm_mapper
    thinker_config = SimpleNamespace(audio_config=SimpleNamespace())
    hf_config = SimpleNamespace(
        thinker_config=thinker_config,
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
    )
    model_config = SimpleNamespace(hf_config=hf_config, multimodal_config=None)
    return SimpleNamespace(quant_config=quant_config, model_config=model_config)


@contextmanager
def _null_mark(self, *args, **kwargs):
    yield


def _stop_encoder(seen: dict):
    def _boom(config, prefix: str = "", quant_config=None):
        seen["quant_config"] = quant_config
        raise RuntimeError("stop")

    return _boom


def test_thinker_renames_ignore_list_before_tower(monkeypatch):
    ignored = [
        "thinker.audio_tower.layers.0.self_attn.q_proj",
        "thinker.audio_tower.layers.0.self_attn.k_proj",
        "thinker.audio_tower.layers.0.self_attn.v_proj",
        "thinker.visual.blocks.0.attn.qkv",
        "thinker.model.layers.0.mlp.gate_proj",
    ]
    vllm_config = _ignore_config(ignored)
    quant_config = vllm_config.quant_config
    seen: dict = {}
    monkeypatch.setattr(
        qwen3_omni_moe_thinker, "Qwen3OmniMoeAudioEncoder", _stop_encoder(seen)
    )
    cls = qwen3_omni_moe_thinker.Qwen3OmniMoeThinkerForConditionalGeneration
    monkeypatch.setattr(cls, "_mark_tower_model", _null_mark)
    with pytest.raises(RuntimeError, match="stop"):
        cls(vllm_config=vllm_config)

    assert seen["quant_config"] is quant_config
    assert quant_config.ignored_layers == [
        "audio_tower.layers.0.self_attn.q_proj",
        "audio_tower.layers.0.self_attn.k_proj",
        "audio_tower.layers.0.self_attn.v_proj",
        "visual.blocks.0.attn.qkv",
        "language_model.model.layers.0.mlp.gate_proj",
    ]
    assert quant_config.packed_modules_mapping["qkv_proj"] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    prefix = "audio_tower.layers.0.self_attn.qkv_proj"
    assert is_layer_skipped(
        prefix,
        quant_config.ignored_layers,
        quant_config.packed_modules_mapping,
        match_mode="exact",
    )
    with pytest.raises(ValueError, match="some but not all shards"):
        is_layer_skipped(
            prefix,
            ["audio_tower.layers.0.self_attn.q_proj"],
            quant_config.packed_modules_mapping,
            match_mode="exact",
        )


def test_asr_aliases_undotted_projectors_before_tower(monkeypatch):
    mapper = qwen3_asr.Qwen3ASRForConditionalGeneration.hf_to_vllm_mapper
    assert (
        mapper.map_name("model.multi_modal_projector.linear_1.weight")
        == "audio_tower.proj1.weight"
    )
    assert (
        mapper.map_name("model.multi_modal_projector.linear_2.weight")
        == "audio_tower.proj2.weight"
    )
    rename = mapper.get_rename_mapper()
    assert rename.map_name("model.multi_modal_projector.linear_1") == (
        "audio_tower.proj1"
    )
    assert rename.map_name("model.multi_modal_projector.linear_2") == (
        "audio_tower.proj2"
    )

    ignored = [
        "model.multi_modal_projector.linear_1",
        "model.multi_modal_projector.linear_2",
        "model.audio_tower.layers.0.self_attn.q_proj",
        "model.audio_tower.layers.0.self_attn.k_proj",
        "model.audio_tower.layers.0.self_attn.v_proj",
    ]
    vllm_config = _ignore_config(ignored)
    quant_config = vllm_config.quant_config
    seen: dict = {}
    monkeypatch.setattr(qwen3_asr, "Qwen3OmniMoeAudioEncoder", _stop_encoder(seen))
    cls = qwen3_asr.Qwen3ASRForConditionalGeneration
    monkeypatch.setattr(cls, "_mark_tower_model", _null_mark)
    with pytest.raises(RuntimeError, match="stop"):
        cls(vllm_config=vllm_config)

    assert seen["quant_config"] is quant_config
    assert quant_config.ignored_layers == [
        "audio_tower.proj1",
        "audio_tower.proj2",
        "audio_tower.layers.0.self_attn.q_proj",
        "audio_tower.layers.0.self_attn.k_proj",
        "audio_tower.layers.0.self_attn.v_proj",
    ]
    assert is_layer_skipped(
        "audio_tower.layers.0.self_attn.qkv_proj",
        quant_config.ignored_layers,
        quant_config.packed_modules_mapping,
        match_mode="exact",
    )
    assert is_layer_skipped(
        "audio_tower.proj1",
        quant_config.ignored_layers,
        quant_config.packed_modules_mapping,
        match_mode="exact",
    )


def test_realtime_and_clef_call_super_init(monkeypatch):
    from vllm.model_executor.models.clef_omni import ClefOmniForJointSchema
    from vllm.model_executor.models.qwen3_asr_realtime import (
        Qwen3ASRRealtimeGeneration,
    )

    vllm_config = _ignore_config(["thinker.audio_tower.layers.0.self_attn.q_proj"])
    seen: dict = {}
    monkeypatch.setattr(
        qwen3_omni_moe_thinker, "Qwen3OmniMoeAudioEncoder", _stop_encoder(seen)
    )
    monkeypatch.setattr(
        qwen3_omni_moe_thinker.Qwen3OmniMoeThinkerForConditionalGeneration,
        "_mark_tower_model",
        _null_mark,
    )
    with pytest.raises(RuntimeError, match="stop"):
        ClefOmniForJointSchema(vllm_config=vllm_config)
    assert seen["quant_config"] is vllm_config.quant_config
    assert vllm_config.quant_config.ignored_layers == [
        "audio_tower.layers.0.self_attn.q_proj"
    ]

    asr_config = _ignore_config(["model.multi_modal_projector.linear_1"])
    seen.clear()
    monkeypatch.setattr(qwen3_asr, "Qwen3OmniMoeAudioEncoder", _stop_encoder(seen))
    monkeypatch.setattr(
        qwen3_asr.Qwen3ASRForConditionalGeneration, "_mark_tower_model", _null_mark
    )
    with pytest.raises(RuntimeError, match="stop"):
        Qwen3ASRRealtimeGeneration(vllm_config=asr_config)
    assert seen["quant_config"] is asr_config.quant_config
    assert asr_config.quant_config.ignored_layers == ["audio_tower.proj1"]


def test_audio_cu_seqlens_int32_starts_at_zero():
    lens = torch.tensor([13, 4, 8, 0], dtype=torch.long)
    cu = qwen3_omni_moe_thinker._audio_cu_seqlens(lens, 8)
    assert cu.dtype == torch.int32
    assert cu.device == lens.device
    assert int(cu[0]) == 0
    assert torch.equal(cu, torch.tensor([0, 8, 13, 17, 25], dtype=torch.int32))

    empty = qwen3_omni_moe_thinker._audio_cu_seqlens(lens[:0], 8)
    assert empty.dtype == torch.int32
    assert int(empty[0]) == 0
    assert empty.shape == (1,)

    if torch.cuda.is_available():
        cuda_lens = lens.cuda()
        cuda_cu = qwen3_omni_moe_thinker._audio_cu_seqlens(cuda_lens, 8)
        assert cuda_cu.dtype == torch.int32
        assert cuda_cu.device == cuda_lens.device
        assert int(cuda_cu[0]) == 0


def test_forward_pads_to_chunk_max_and_cu_seqlens(monkeypatch):
    encoder = _build_encoder(monkeypatch, 1)
    recorded: dict = {"syncs": 0}
    real_gather = qwen3_omni_moe_thinker._gather_audio_chunks
    real_sync = qwen3_omni_moe_thinker.gpu_sync_allowed

    def _gather(features, chunk_lengths, pad_width):
        recorded["lengths"] = chunk_lengths.detach().cpu().tolist()
        recorded["pad_width"] = pad_width
        gathered = real_gather(features, chunk_lengths, pad_width)
        recorded["gathered"] = gathered.detach().cpu()
        return gathered

    @contextmanager
    def _count_sync(*args, **kwargs):
        recorded["syncs"] += 1
        with real_sync(*args, **kwargs):
            yield

    monkeypatch.setattr(qwen3_omni_moe_thinker, "_gather_audio_chunks", _gather)
    monkeypatch.setattr(qwen3_omni_moe_thinker, "gpu_sync_allowed", _count_sync)

    def _capture(hidden_states, cu_seqlens, max_seqlen):
        recorded["cu"] = cu_seqlens
        return hidden_states

    for layer in encoder.layers:
        layer.forward = _capture

    recorded["syncs"] = 0
    empty = encoder(
        torch.zeros(encoder.num_mel_bins, 0),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    assert empty.shape == (0, encoder.proj2.output_size)
    assert recorded["syncs"] == 0

    cases = [
        ([1, 2, 5], [13, 4, 8]),
        ([100], [13]),
        ([101], [13, 1]),
        ([200], [26]),
        ([1, 2, 5, 50, 100, 101], [1, 2, 3, 7, 13, 14, 1]),
        # Zero-length rows must not overwrite another row's tail.
        ([50, 0], [13]),
        ([150, 0], [13, 4]),
        ([50, 0, 30], [13, 4]),
        ([0, 50], [13]),
        ([100, 0], [13]),
        ([0, 200], [26]),
        ([0, 150, 0], [13, 1]),
    ]
    for feature_lens, aftercnn in cases:
        expect_lengths = _reference_chunk_lengths(feature_lens, WINDOW)
        assert expect_lengths
        expect_pad = max(expect_lengths)
        assert expect_pad == max(min(length, WINDOW) for length in feature_lens)
        total = sum(feature_lens)
        features = torch.arange(encoder.num_mel_bins * total, dtype=torch.float32)
        features = features.reshape(encoder.num_mel_bins, total)
        after = torch.tensor(aftercnn, dtype=torch.long)
        recorded["syncs"] = 0
        encoder(features, torch.tensor(feature_lens, dtype=torch.long), after)

        assert recorded["lengths"] == expect_lengths
        assert recorded["pad_width"] == expect_pad
        assert recorded["pad_width"] == max(recorded["lengths"])
        chunks = list(features.T.split(expect_lengths, dim=0))
        reference = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)
        assert torch.equal(recorded["gathered"], reference)

        cnn_len = expect_pad
        for _ in range(3):
            cnn_len = (cnn_len - 1) // 2 + 1
        window_aftercnn = cnn_len * (encoder.n_window_infer // WINDOW)
        cu = recorded["cu"]
        assert cu.dtype == torch.int32
        assert cu.device == after.device
        assert int(cu[0]) == 0
        expect_cu = torch.tensor(
            _reference_cu(aftercnn, window_aftercnn), dtype=torch.int32
        )
        assert torch.equal(cu.cpu(), expect_cu)
        assert recorded["syncs"] == 2

    assert _reference_chunk_lengths([100], WINDOW) == [100]
    assert _reference_chunk_lengths([200], WINDOW) == [100, 100]
    assert _reference_chunk_lengths([101], WINDOW) == [100, 1]
    short = _reference_chunk_lengths([1, 2, 5], WINDOW)
    assert max(short) == 5


def _conv_stack(generator: torch.Generator) -> list[nn.Conv2d]:
    layers: list[nn.Conv2d] = []
    in_channels = 1
    for _ in range(3):
        layer = nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, padding=1, bias=True)
        with torch.no_grad():
            layer.weight.copy_(
                torch.empty(layer.weight.shape).normal_(0, 0.05, generator=generator)
            )
            layer.bias.copy_(
                torch.empty(layer.bias.shape).normal_(0, 0.05, generator=generator)
            )
        layers.append(layer)
        in_channels = 8
    return layers


def _apply_convs(layers: list[nn.Conv2d], features: torch.Tensor) -> torch.Tensor:
    hidden = features
    for layer in layers:
        hidden = F.gelu(layer(hidden))
    return hidden


def test_conv_bias_pad_to_chunk_max_matches_narrow():
    generator = torch.Generator().manual_seed(0)
    layers = _conv_stack(generator)
    mel = 8

    def _one(length: int) -> torch.Tensor:
        values = torch.arange(mel * length, dtype=torch.float32)
        values = values.reshape(1, 1, mel, length)
        return values / max(length, 1)

    errors = []
    for length in (1, 2, 5, 50, 100):
        narrow = _apply_convs(layers, _one(length))
        wide = F.pad(_one(length), (0, WINDOW - length))
        wide_out = _apply_convs(layers, wide)
        valid = wide_out[..., : narrow.shape[-1]]
        error = (valid - narrow).abs().max().item()
        if length == WINDOW:
            assert error == 0
        else:
            assert not torch.equal(valid, narrow)
            errors.append(error)
    # Window padding changes the biased conv. Own-length padding matches.
    assert 1e-3 < max(errors) < 1e-1

    # 101 is chunks [100, 1]. Both policies pad that tail to 100.
    tail = _one(1)
    padded_tail = F.pad(tail, (0, WINDOW - 1))
    narrow_tail = _apply_convs(layers, tail)
    tail_error = (
        (_apply_convs(layers, padded_tail)[..., : narrow_tail.shape[-1]] - narrow_tail)
        .abs()
        .max()
        .item()
    )
    assert 1e-4 < tail_error < 1e-1

    batch = torch.cat([F.pad(_one(length), (0, 5 - length)) for length in (1, 2, 5)], 0)
    window_batch = torch.cat(
        [F.pad(_one(length), (0, WINDOW - length)) for length in (1, 2, 5)], 0
    )
    batch_out = _apply_convs(layers, batch)
    window_out = _apply_convs(layers, window_batch)
    mixed_error = (window_out[..., : batch_out.shape[-1]] - batch_out).abs().max()
    assert 1e-5 < mixed_error.item() < 1e-1


def _capture_length_math(monkeypatch, module):
    order: list[tuple[str, str]] = []
    real_lengths = module._get_feat_extract_output_lengths
    real_h2d = module.async_tensor_h2d

    def _lengths(values):
        order.append(("lengths", values.device.type))
        return real_lengths(values)

    def _h2d(data, device=None, dtype=None, out=None):
        if isinstance(data, torch.Tensor):
            kind = data.device.type
        else:
            kind = type(data).__name__
        order.append(("h2d", kind))
        return real_h2d(data, device, dtype=dtype, out=out)

    monkeypatch.setattr(module, "_get_feat_extract_output_lengths", _lengths)
    monkeypatch.setattr(module, "async_tensor_h2d", _h2d)
    return order


def test_process_audio_input_keeps_lengths_on_cpu(monkeypatch):
    lengths = torch.tensor([100, 30], dtype=torch.long)
    features = torch.zeros(4, 130)

    class _Tower:
        dtype = torch.float32

        def __call__(self, inputs, feature_lens, aftercnn_lens):
            assert feature_lens.device.type == "cpu"
            assert aftercnn_lens.device.type == "cpu"
            return torch.zeros(int(aftercnn_lens.sum()), 2)

    thinker_order = _capture_length_math(monkeypatch, qwen3_omni_moe_thinker)
    thinker_cls = qwen3_omni_moe_thinker.Qwen3OmniMoeConditionalGenerationMixin
    thinker_out = thinker_cls._process_audio_input(
        SimpleNamespace(audio_tower=_Tower()),
        {"input_features": features, "audio_feature_lengths": lengths},
    )
    assert thinker_order[0] == ("lengths", "cpu")
    assert thinker_order.count(("h2d", "cpu")) == 2
    assert len(tuple(thinker_out)) == 2

    asr_order = _capture_length_math(monkeypatch, qwen3_asr)
    asr_out = qwen3_asr.Qwen3ASRForConditionalGeneration._process_audio_input(
        SimpleNamespace(audio_tower=_Tower()),
        {"input_features": features, "audio_feature_lengths": lengths},
    )
    assert asr_order[0] == ("lengths", "cpu")
    assert asr_order.count(("h2d", "cpu")) == 2
    assert len(tuple(asr_out)) == 2
