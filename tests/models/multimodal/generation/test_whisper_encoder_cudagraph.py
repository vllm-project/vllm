# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whisper conv-stem batching and fixed full-chunk mel CUDA graphs.

Conv-stem checks run on CPU. CUDA graph replay is skipped without a device
graph. No checkpoint is downloaded.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import WhisperConfig

from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.model_executor.models.whisper import (
    WhisperEncoder,
    WhisperForConditionalGeneration,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

HAS_DEVICE_GRAPH = current_platform.is_cuda_alike() or current_platform.is_xpu()

if current_platform.is_xpu():
    from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper

    # Route torch.cuda graph APIs to torch.xpu, as XPUModelRunner does.
    with _torch_cuda_wrapper():
        pass

N_MELS = 8
D_MODEL = 32
POSITIONS = 8
STATIC_MEL = 16


def _hf_config(**overrides: object) -> WhisperConfig:
    params: dict[str, object] = dict(
        vocab_size=64,
        num_mel_bins=N_MELS,
        d_model=D_MODEL,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_layers=0,
        decoder_layers=0,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=POSITIONS,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    params.update(overrides)
    return WhisperConfig(**params)


def _stem_encoder(**overrides: object) -> WhisperEncoder:
    hf_config = _hf_config(**overrides)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        cache_config=None,
        quant_config=None,
    )
    return WhisperEncoder(vllm_config=vllm_config, enable_pp=False)


class _TinyModelConfig:
    def __init__(self, hf_config: WhisperConfig, max_model_len: int = 4) -> None:
        self.hf_config = hf_config
        self.dtype = torch.float32
        self.max_model_len = max_model_len
        self.head_dtype = torch.float32
        self.is_multimodal_model = True
        self.multimodal_config = MultiModalConfig(
            limit_per_prompt={"audio": 1, "video": 0},
            mm_encoder_attn_backend=AttentionBackendEnum.TORCH_SDPA,
        )

    def get_multimodal_config(self) -> MultiModalConfig:
        return self.multimodal_config


def _vllm_config(hf_config: WhisperConfig) -> VllmConfig:
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            encoder_cudagraph_token_budgets=[POSITIONS, POSITIONS * 2],
        )
    )
    # Smaller than max_source_positions. The budget must ignore this.
    vllm_config.model_config = _TinyModelConfig(hf_config, max_model_len=4)
    vllm_config.scheduler_config.max_num_batched_tokens = POSITIONS * 2
    return vllm_config


def _manager(
    vllm_config: VllmConfig,
    model: WhisperForConditionalGeneration,
    device: torch.device | None = None,
) -> EncoderCudaGraphManager:
    return EncoderCudaGraphManager(
        vllm_config,
        torch.device("cpu") if device is None else device,
        torch.float32,
        model,
    )


@pytest.fixture
def whisper_bundle(dist_init):
    del dist_init
    hf_config = _hf_config(encoder_layers=1)
    vllm_config = _vllm_config(hf_config)
    with set_current_vllm_config(vllm_config):
        model = WhisperForConditionalGeneration(vllm_config=vllm_config)
        model.eval()
        yield vllm_config, model


def _seed_conv(encoder: WhisperEncoder) -> None:
    with torch.no_grad():
        encoder.conv1.weight.normal_()
        encoder.conv2.weight.normal_()
        encoder.conv1.bias.zero_()
        encoder.conv2.bias.zero_()


def test_batched_conv_stem_matches_per_item() -> None:
    torch.manual_seed(0)
    encoder = _stem_encoder()
    features = torch.randn(3, N_MELS, STATIC_MEL)

    one = features[:1]
    per_item = encoder._conv_embed(one[0]).unsqueeze(0)
    assert torch.equal(encoder._conv_embed(one), per_item)
    assert torch.equal(encoder.forward(one), encoder.layer_norm(per_item))

    batched = encoder._conv_embed(features)
    itemwise = torch.stack(
        [encoder._conv_embed(features[i]) for i in range(features.shape[0])],
        dim=0,
    )
    assert torch.allclose(batched, itemwise, atol=1e-5, rtol=1e-5)
    # fp32 CPU batched conv is not bit-exact for B>1 (max abs ~2e-7).
    assert not torch.equal(batched, itemwise)
    assert torch.allclose(
        encoder.forward(features),
        encoder.layer_norm(itemwise),
        atol=1e-5,
        rtol=1e-5,
    )


def test_odd_length_pad_changes_last_conv_frame() -> None:
    torch.manual_seed(0)
    encoder = _stem_encoder()
    _seed_conv(encoder)

    odd = torch.randn(N_MELS, 15)
    padded = F.pad(odd, (0, 1))
    odd_stem = encoder._conv_embed(odd)
    padded_stem = encoder._conv_embed(padded)
    assert odd_stem.shape[0] == 8
    assert padded_stem.shape[0] == 8
    assert (odd_stem[-1] - padded_stem[-1]).abs().max() > 1e-3

    batched = encoder.forward(odd.unsqueeze(0))
    assert batched.shape[1] == odd_stem.shape[0]
    assert torch.equal(batched, encoder.layer_norm(odd_stem.unsqueeze(0)))


def test_unequal_lengths_stay_per_item() -> None:
    torch.manual_seed(0)
    encoder = _stem_encoder()
    _seed_conv(encoder)
    short = torch.randn(N_MELS, 11)
    longer = torch.randn(N_MELS, 12)

    output = encoder.forward([short, longer])
    itemwise = torch.stack(
        [encoder._conv_embed(short), encoder._conv_embed(longer)],
        dim=0,
    )
    assert output.shape[1] == 6
    assert torch.equal(output, encoder.layer_norm(itemwise))

    padded = F.pad(short, (0, 1))
    wrong = encoder._conv_embed(torch.stack([padded, longer], dim=0))
    assert (wrong[0, -1] - itemwise[0, -1]).abs().max() > 1e-3


def test_rank3_list_is_concatenated_not_stacked() -> None:
    torch.manual_seed(0)
    encoder = _stem_encoder()
    chunk = torch.randn(2, N_MELS, STATIC_MEL)
    other = torch.randn(2, N_MELS, STATIC_MEL)

    assert torch.equal(encoder.forward([chunk]), encoder.forward(chunk))
    cat = torch.cat([chunk, other], dim=0)
    output = encoder.forward([chunk, other])
    assert output.shape[0] == 4
    assert torch.equal(output, encoder.forward(cat))


def test_equal_rank2_list_and_tuple_are_stacked() -> None:
    torch.manual_seed(0)
    encoder = _stem_encoder()
    items = [torch.randn(N_MELS, STATIC_MEL) for _ in range(3)]
    stacked = torch.stack(items, dim=0)

    assert torch.equal(encoder.forward([items[0]]), encoder.forward(stacked[:1]))
    assert torch.allclose(
        encoder.forward(items), encoder.forward(stacked), atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        encoder.forward(tuple(items)),
        encoder.forward(stacked),
        atol=1e-5,
        rtol=1e-5,
    )
    assert encoder.forward(tuple(items)).shape[0] == 3


def test_bare_rank2_stays_on_the_per_item_loop() -> None:
    encoder = _stem_encoder()
    features = torch.randn(N_MELS, STATIC_MEL)
    # One-shot conv accepts (C, T). The per-item loop iterates rows instead.
    assert encoder._conv_embed(features).shape[0] == POSITIONS
    with pytest.raises(RuntimeError):
        encoder.forward(features)


def test_static_full_chunk_mel_frames() -> None:
    encoder = _stem_encoder()
    assert encoder.static_full_chunk_mel_frames() == STATIC_MEL

    stock = _stem_encoder(max_source_positions=1500)
    assert stock.static_full_chunk_mel_frames() == 3000

    encoder.conv2.stride = (1,)
    assert encoder.static_full_chunk_mel_frames() is None


def test_budget_ignores_max_model_len_and_sizes_width(whisper_bundle) -> None:
    vllm_config, model = whisper_bundle
    assert vllm_config.model_config.max_model_len == 4
    assert model.get_encoder_cudagraph_budget_range(vllm_config) == (
        POSITIONS,
        POSITIONS * 2,
    )
    assert model.model.encoder.static_full_chunk_mel_frames() == STATIC_MEL

    manager = _manager(vllm_config, model)
    assert manager.max_batch_size == 2
    assert manager.token_budgets == [POSITIONS, POSITIONS * 2]
    assert manager._find_smallest_fitting_budget_given_tokens(POSITIONS) == POSITIONS
    two_clip_tokens = POSITIONS * 2
    assert (
        manager._find_smallest_fitting_budget_given_tokens(two_clip_tokens)
        == two_clip_tokens
    )

    width_1 = model.prepare_encoder_cudagraph_capture_inputs(
        POSITIONS,
        manager.max_batch_size,
        manager.max_frames_per_batch,
        torch.device("cpu"),
        torch.float32,
    )
    width_2 = model.prepare_encoder_cudagraph_capture_inputs(
        POSITIONS * 2,
        manager.max_batch_size,
        manager.max_frames_per_batch,
        torch.device("cpu"),
        torch.float32,
    )
    assert width_1.values["input_features"].shape == (1, N_MELS, STATIC_MEL)
    assert width_2.values["input_features"].shape == (2, N_MELS, STATIC_MEL)

    with pytest.raises(ValueError, match="cannot hold one"):
        model.prepare_encoder_cudagraph_capture_inputs(
            4,
            manager.max_batch_size,
            0,
            torch.device("cpu"),
            torch.float32,
        )

    features = torch.randn(3, N_MELS, STATIC_MEL)
    specs = model.get_encoder_cudagraph_item_specs({"input_features": features})
    assert [spec.output_tokens for spec in specs] == [POSITIONS, POSITIONS, POSITIONS]
    assert [spec.input_size for spec in specs] == [STATIC_MEL, STATIC_MEL, STATIC_MEL]
    selected = model.select_encoder_cudagraph_items(
        {"input_features": features}, [0, 2]
    )
    assert torch.equal(selected["input_features"], features[[0, 2]])
    empty = model.select_encoder_cudagraph_items({"input_features": features}, [])
    assert empty["input_features"].shape[0] == 0

    full = features[:1]
    short = features[:, :, : STATIC_MEL - 1]
    assert model.should_use_encoder_cudagraph({"input_features": full}) is True
    assert model.should_use_encoder_cudagraph({"input_features": short}) is False
    assert model.should_use_encoder_cudagraph({"input_features": [full[0]]}) is False
    assert manager.execute({"input_features": full}) is None
    assert manager.is_captured() is False

    saved_budgets = list(vllm_config.compilation_config.encoder_cudagraph_token_budgets)
    vllm_config.compilation_config.encoder_cudagraph_token_budgets = [POSITIONS * 2]
    suffix_manager = _manager(vllm_config, model)
    vllm_config.compilation_config.encoder_cudagraph_token_budgets = saved_budgets
    assert suffix_manager.max_batch_size == 2
    suffix_width = model.prepare_encoder_cudagraph_capture_inputs(
        POSITIONS * 2,
        suffix_manager.max_batch_size,
        suffix_manager.max_frames_per_batch,
        torch.device("cpu"),
        torch.float32,
    )
    assert suffix_width.values["input_features"].shape[0] == 2


def test_capture_stays_off_without_touching_the_pool(whisper_bundle) -> None:
    vllm_config, model = whisper_bundle
    mm_config = vllm_config.model_config.multimodal_config

    vllm_config.lora_config = object()
    assert model.encoder_cudagraph_capture_allowed() is False
    manager = _manager(vllm_config, model)
    manager.capture(graph_pool=object())
    assert manager.is_captured() is False
    vllm_config.lora_config = None

    mm_config.mm_encoder_attn_dtype = "fp8"
    mm_config.mm_encoder_fp8_scale_path = None
    assert model.encoder_cudagraph_capture_allowed() is False
    manager = _manager(vllm_config, model)
    manager.capture(graph_pool=object())
    assert manager.is_captured() is False

    mm_config.mm_encoder_fp8_scale_path = "static-fp8-scales.json"
    assert model.encoder_cudagraph_capture_allowed() is True
    mm_config.mm_encoder_attn_dtype = None
    mm_config.mm_encoder_fp8_scale_path = None

    saved_stride = model.model.encoder.conv2.stride
    model.model.encoder.conv2.stride = (1,)
    assert model.model.encoder.static_full_chunk_mel_frames() is None
    assert model.encoder_cudagraph_capture_allowed() is False
    manager = _manager(vllm_config, model)
    manager.capture(graph_pool=object())
    assert manager.is_captured() is False
    model.model.encoder.conv2.stride = saved_stride
    assert model.encoder_cudagraph_capture_allowed() is True


def test_protocol_is_only_on_whisper_conditional_generation() -> None:
    from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
    from vllm.model_executor.models.kimi_audio import (
        KimiAudioForConditionalGeneration,
        KimiAudioWhisperEncoder,
    )
    from vllm.model_executor.models.minicpmo import MiniCPMO2_6, MiniCPMO4_5
    from vllm.model_executor.models.minicpmv import (
        MiniCPMV2_6,
        MiniCPMV4_5,
        _MiniCPMVEncoderCudaGraphMixin,
    )
    from vllm.model_executor.models.moss_transcribe_diarize import (
        MossTranscribeDiarizeForConditionalGeneration,
        MossTranscribeDiarizeWhisperEncoder,
    )
    from vllm.model_executor.models.ultravox import (
        UltravoxModel,
        UltravoxWhisperEncoder,
    )

    assert SupportsEncoderCudaGraph in WhisperForConditionalGeneration.__mro__
    for cls in (
        WhisperEncoder,
        UltravoxWhisperEncoder,
        UltravoxModel,
        KimiAudioWhisperEncoder,
        KimiAudioForConditionalGeneration,
        MossTranscribeDiarizeWhisperEncoder,
        MossTranscribeDiarizeForConditionalGeneration,
        MiniCPMV4_5,
        MiniCPMO4_5,
    ):
        assert SupportsEncoderCudaGraph not in cls.__mro__

    assert SupportsEncoderCudaGraph in MiniCPMV2_6.__mro__
    assert SupportsEncoderCudaGraph in MiniCPMO2_6.__mro__
    assert SupportsEncoderCudaGraph in Gemma4ForConditionalGeneration.__mro__

    allowed = SupportsEncoderCudaGraph.__dict__["encoder_cudagraph_capture_allowed"]
    should_use = SupportsEncoderCudaGraph.__dict__["should_use_encoder_cudagraph"]
    assert allowed(None) is True
    assert should_use(None, {}) is True
    for cls in (
        Gemma4ForConditionalGeneration,
        _MiniCPMVEncoderCudaGraphMixin,
    ):
        assert "encoder_cudagraph_capture_allowed" not in cls.__dict__
        assert "should_use_encoder_cudagraph" not in cls.__dict__


@pytest.mark.skipif(not HAS_DEVICE_GRAPH, reason="Requires CUDA, ROCm or XPU graph")
def test_fixed_mel_cudagraph_replay(whisper_bundle) -> None:
    vllm_config, model = whisper_bundle
    device = torch.device(current_platform.device_type, 0)
    model.to(device=device)
    manager = _manager(vllm_config, model, device)
    manager.capture(current_platform.graph_pool_handle())
    assert manager.is_captured() is True

    graphs = manager.budget_graphs["default"]
    assert graphs[POSITIONS].input_buffers["input_features"].shape[0] == 1
    assert graphs[POSITIONS * 2].input_buffers["input_features"].shape[0] == 2

    def fill_nan() -> None:
        for meta in graphs.values():
            meta.output_buffer.fill_(float("nan"))

    one = torch.randn(1, N_MELS, STATIC_MEL, device=device)
    fill_nan()
    result = manager.execute({"input_features": one})
    assert result is not None
    assert len(result) == 1
    assert result[0].shape == (POSITIONS, D_MODEL)
    assert torch.isfinite(graphs[POSITIONS].output_buffer).all().item()
    assert torch.isnan(graphs[POSITIONS * 2].output_buffer).all().item()
    eager = model.encoder_eager_forward({"input_features": one})
    torch.testing.assert_close(result[0], eager, atol=1e-4, rtol=1e-4)

    two = torch.randn(2, N_MELS, STATIC_MEL, device=device)
    fill_nan()
    result = manager.execute({"input_features": two})
    assert result is not None
    assert len(result) == 2
    assert torch.isnan(graphs[POSITIONS].output_buffer).all().item()
    assert torch.isfinite(graphs[POSITIONS * 2].output_buffer).all().item()
    eager = model.encoder_eager_forward({"input_features": two})
    torch.testing.assert_close(torch.cat(result, dim=0), eager, atol=1e-4, rtol=1e-4)

    short = one[:, :, : STATIC_MEL - 1]
    assert manager.execute({"input_features": short}) is None

    saved_budgets = list(vllm_config.compilation_config.encoder_cudagraph_token_budgets)
    vllm_config.compilation_config.encoder_cudagraph_token_budgets = [POSITIONS * 2]
    suffix_manager = _manager(vllm_config, model, device)
    suffix_manager.capture(current_platform.graph_pool_handle())
    vllm_config.compilation_config.encoder_cudagraph_token_budgets = saved_budgets

    clip = torch.randn(1, N_MELS, STATIC_MEL, device=device)
    result = suffix_manager.execute({"input_features": clip})
    assert result is not None
    assert len(result) == 1
    replay = suffix_manager.budget_graphs["default"][POSITIONS * 2]
    assert replay.input_buffers["input_features"].shape[0] == 2
    assert replay.output_buffer.shape[0] == POSITIONS * 2
    assert torch.equal(replay.input_buffers["input_features"][0], clip[0])
    assert torch.equal(
        replay.input_buffers["input_features"][1],
        torch.zeros_like(replay.input_buffers["input_features"][1]),
    )
    eager = model.encoder_eager_forward({"input_features": clip})
    torch.testing.assert_close(result[0], eager, atol=1e-4, rtol=1e-4)
