# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from transformers import Qwen3Config

from vllm.model_executor.models.interfaces import supports_multimodal, supports_pp
from vllm.model_executor.models.interfaces_base import is_text_generation_model
from vllm.model_executor.models.moss_audio import (
    DEFAULT_MAX_AUDIO_SECONDS,
    MOSS_AUDIO_BOS_TOKEN,
    MOSS_AUDIO_BOS_TOKEN_ID,
    MOSS_AUDIO_EOS_TOKEN,
    MOSS_AUDIO_EOS_TOKEN_ID,
    MOSS_AUDIO_TOKEN,
    MOSS_AUDIO_TOKEN_ID,
    MossAudioConfig,
    MossAudioEncoder,
    MossAudioEncoderConfig,
    MossAudioModel,
    MossAudioMultiModalProcessor,
    MossAudioProcessingInfo,
    MossAudioProcessor,
    _moss_audio_field_config,
)
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.model_executor.models.utils import AutoWeightsLoader

from ...registry import _MULTIMODAL_EXAMPLE_MODELS


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)

    def batch_decode(
        self, batch_token_ids: list[list[int]], **kwargs: object
    ) -> list[str]:
        return [self.decode(token_ids, **kwargs) for token_ids in batch_token_ids]


class _Info:
    def __init__(self, processor: MossAudioProcessor) -> None:
        self.processor = processor

    def get_hf_processor(self, **kwargs: object) -> MossAudioProcessor:
        del kwargs
        return self.processor


class _ProcessingContext:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def get_tokenizer(self) -> _Tokenizer:
        return self.tokenizer


class _OutKwargs:
    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        self.data = data

    def get_data(self) -> dict[str, torch.Tensor]:
        return self.data


class _ModelConfig:
    def __init__(self) -> None:
        self.hf_config = MossAudioConfig(language_config=Qwen3Config())
        self.multimodal_config = None


class _ParallelConfig:
    def __init__(
        self,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size


class _VllmConfig:
    def __init__(self, parallel_config: _ParallelConfig) -> None:
        self.model_config = _ModelConfig()
        self.quant_config = None
        self.parallel_config = parallel_config


def _patch_tensor_parallel_for_linear_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.linear as linear_layers
    import vllm.model_executor.models.moss_audio as moss_audio_module
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(
        moss_audio_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        linear_layers,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(linear_layers, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)


def test_moss_audio_processor_expands_audio_placeholder() -> None:
    processor = MossAudioProcessor(_Tokenizer())
    prompt = (
        f"before {MOSS_AUDIO_BOS_TOKEN}{MOSS_AUDIO_TOKEN}{MOSS_AUDIO_EOS_TOKEN} after"
    )
    raw_mel_len = 17
    audio = torch.zeros(160 * raw_mel_len)

    processed = processor(text=prompt, audio=[audio])
    input_ids = processed["input_ids"][0].tolist()
    expected_audio_tokens = MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)

    assert input_ids.count(MOSS_AUDIO_BOS_TOKEN_ID) == 1
    assert input_ids.count(MOSS_AUDIO_EOS_TOKEN_ID) == 1
    assert input_ids.count(MOSS_AUDIO_TOKEN_ID) == expected_audio_tokens
    assert processed["audio_data"].shape == (1, 128, raw_mel_len)
    assert processed["audio_data_seqlens"].tolist() == [raw_mel_len]


def test_moss_audio_time_markers_are_not_embedding_targets() -> None:
    processor = MossAudioProcessor(_Tokenizer(), enable_time_marker=True)
    mm_processor = object.__new__(MossAudioMultiModalProcessor)
    mm_processor.info = _Info(processor)

    audio_len = 320
    audio_token_len = MossAudioEncoder.compute_num_audio_tokens(audio_len)
    update = mm_processor._get_prompt_updates(
        mm_items=None,
        hf_processor_mm_kwargs={},
        out_mm_kwargs=_OutKwargs(
            {"audio_data_seqlens": torch.tensor([audio_len], dtype=torch.long)}
        ),
    )[0].resolve(0)

    full = update.content.full
    is_embed = update.content.is_embed(None, full)

    assert full[0] == MOSS_AUDIO_BOS_TOKEN_ID
    assert full[-1] == MOSS_AUDIO_EOS_TOKEN_ID
    assert is_embed.sum().item() == audio_token_len
    assert len(full) > audio_token_len + 2


def test_moss_audio_processing_info_caches_default_processor() -> None:
    info = MossAudioProcessingInfo(_ProcessingContext())

    processor = info.get_hf_processor()
    same_processor = info.get_hf_processor()
    explicit_default_processor = info.get_hf_processor(enable_time_marker=False)
    time_marker_processor = info.get_hf_processor(enable_time_marker=True)

    assert same_processor is processor
    assert explicit_default_processor is processor
    assert time_marker_processor is not processor
    assert not processor.enable_time_marker
    assert time_marker_processor.enable_time_marker


def test_moss_audio_encoder_vectorized_mask_matches_token_count() -> None:
    config = MossAudioEncoderConfig(
        output_dim=8,
        num_mel_bins=8,
        n_window=4,
        conv_chunksize=2,
    )
    encoder = object.__new__(MossAudioEncoder)
    encoder.config = config
    encoder.deepstack_encoder_layer_indexes = []
    encoder.chunk_frames = config.n_window * 2
    encoder.conv_chunksize = config.conv_chunksize

    def _encode_chunk_batch(
        input_features: torch.Tensor,
        seq_lengths: torch.Tensor,
        output_deepstack_hidden_states: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert not output_deepstack_hidden_states
        downsampled_lengths = MossAudioEncoder._compute_downsampled_length(seq_lengths)
        return (
            input_features.new_ones(
                input_features.shape[0],
                int(downsampled_lengths.max().item()),
                config.output_dim,
            ),
            [],
        )

    encoder._encode_chunk_batch = _encode_chunk_batch
    feature_lens = torch.tensor([5, 13], dtype=torch.long)
    input_features = torch.randn(2, config.num_mel_bins, 13)

    hidden_states, deepstack = MossAudioEncoder.forward(
        encoder,
        input_features,
        feature_lens=feature_lens,
        output_deepstack_hidden_states=False,
    )

    expected_tokens = sum(
        MossAudioEncoder.compute_num_audio_tokens(int(length))
        for length in feature_lens
    )
    assert hidden_states.shape == (1, expected_tokens, config.output_dim)
    assert deepstack is None


def test_moss_audio_field_config_and_interfaces() -> None:
    fields = _moss_audio_field_config({})

    assert set(fields) == {"audio_data", "audio_data_seqlens"}
    assert (
        MossAudioModel.get_placeholder_str("audio", 0)
        == f"{MOSS_AUDIO_BOS_TOKEN}{MOSS_AUDIO_TOKEN}{MOSS_AUDIO_EOS_TOKEN}"
    )
    assert supports_multimodal(MossAudioModel)
    assert is_text_generation_model(MossAudioModel)
    assert not supports_pp(MossAudioModel)


def test_moss_audio_registry_entries() -> None:
    assert _MULTIMODAL_MODELS["MossAudioModel"] == ("moss_audio", "MossAudioModel")
    assert (
        _MULTIMODAL_EXAMPLE_MODELS["MossAudioModel"].default
        == "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
    )


@pytest.mark.parametrize(
    ("parallel_config", "message"),
    [
        (_ParallelConfig(tensor_parallel_size=2), "tensor_parallel_size=1"),
        (_ParallelConfig(pipeline_parallel_size=2), "pipeline_parallel_size=1"),
    ],
)
def test_moss_audio_rejects_unsupported_parallelism(
    parallel_config: _ParallelConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MossAudioModel(vllm_config=_VllmConfig(parallel_config))


def test_moss_audio_deepstack_requires_matching_audio_token_count() -> None:
    model = object.__new__(MossAudioModel)
    inputs_embeds = torch.zeros(4, 8)
    is_multimodal = torch.tensor([False, True, True, False])

    with pytest.raises(ValueError, match="DeepStack audio token count mismatch"):
        model._cache_deepstack_input_embeds(
            inputs_embeds=inputs_embeds,
            deepstack_embeddings=((torch.ones(1, 8),),),
            is_multimodal=is_multimodal,
        )


def test_moss_audio_deepstack_disabled_skips_cache() -> None:
    class _FakeAudioEncoder:
        dtype = torch.float32

        def __init__(self) -> None:
            self.output_deepstack_hidden_states: bool | None = None

        def __call__(
            self,
            audio_data: torch.Tensor,
            *,
            feature_lens: torch.Tensor,
            output_deepstack_hidden_states: bool,
        ) -> tuple[torch.Tensor, None]:
            del audio_data
            self.output_deepstack_hidden_states = output_deepstack_hidden_states
            audio_lengths = MossAudioEncoder._compute_downsampled_length(feature_lens)
            total_tokens = int(audio_lengths.sum().item())
            return torch.ones(1, total_tokens, 8), None

    class _IdentityAdapter:
        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    model = object.__new__(MossAudioModel)
    audio_encoder = _FakeAudioEncoder()
    model.audio_encoder = audio_encoder
    model.audio_adapter = _IdentityAdapter()
    model.deepstack_audio_merger_list = []
    model.deepstack_input_embeds = None

    multimodal_embeddings = model.embed_multimodal(
        audio_data=torch.zeros(2, 128, 9),
        audio_data_seqlens=torch.tensor([8, 9], dtype=torch.long),
    )

    assert audio_encoder.output_deepstack_hidden_states is False
    assert len(multimodal_embeddings) == 1
    assert [embeds.shape for embeds in multimodal_embeddings[0]] == [
        torch.Size([1, 8]),
        torch.Size([2, 8]),
    ]
    assert model.deepstack_input_embeds is None


def test_moss_audio_rejects_too_short_audio() -> None:
    processor = MossAudioProcessor(_Tokenizer())
    prompt = f"{MOSS_AUDIO_BOS_TOKEN}{MOSS_AUDIO_TOKEN}{MOSS_AUDIO_EOS_TOKEN}"

    with pytest.raises(ValueError, match="too short"):
        processor(text=prompt, audio=[torch.empty(0)])

    model = object.__new__(MossAudioModel)
    with pytest.raises(ValueError, match="too short"):
        model._parse_and_validate_audio_input(
            audio_data=torch.zeros(1, 128, 1),
            audio_data_seqlens=torch.tensor([0], dtype=torch.long),
        )


def test_moss_audio_max_tokens_uses_default_30s_limit() -> None:
    info = object.__new__(MossAudioProcessingInfo)

    raw_mel_len = math.ceil((DEFAULT_MAX_AUDIO_SECONDS * 16000) / 160)
    assert info.get_mm_max_tokens_per_item(seq_len=2048, mm_counts={"audio": 1}) == {
        "audio": MossAudioEncoder.compute_num_audio_tokens(raw_mel_len),
    }


def test_moss_audio_encoder_loads_realistic_attention_weight_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig

    _patch_tensor_parallel_for_linear_layers(monkeypatch)
    config = MossAudioEncoderConfig(
        d_model=8,
        output_dim=8,
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        downsample_hidden_size=2,
        deepstack_encoder_layer_indexes=[],
    )
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(vllm_config):
        encoder = MossAudioEncoder(config)

    assert "load_weights" not in MossAudioEncoder.__dict__

    attention = encoder.layers[0].self_attn
    assert hasattr(attention, "q_proj")
    assert hasattr(attention, "k_proj")
    assert hasattr(attention, "v_proj")
    assert hasattr(attention, "out_proj")
    assert not hasattr(attention, "qkv")
    assert attention.k_proj.bias is None

    attn_output = attention(
        torch.randn(1, 3, config.d_model),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert attn_output.shape == (1, 3, config.d_model)

    weight_names = [
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_proj.bias",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.v_proj.bias",
        "layers.0.self_attn.out_proj.weight",
        "layers.0.self_attn.out_proj.bias",
        "conv1.weight",
        "conv1.bias",
    ]
    params = dict(encoder.named_parameters(remove_duplicate=False))
    assert "layers.0.self_attn.k_proj.bias" not in params

    expected_weights = {
        name: torch.full_like(params[name], fill_value=float(i + 1))
        for i, name in enumerate(weight_names)
    }
    loaded = AutoWeightsLoader(encoder).load_weights(expected_weights.items())

    assert loaded == set(weight_names)
    assert not any(".qkv." in name for name in loaded)
    for name, expected_weight in expected_weights.items():
        assert torch.equal(params[name], expected_weight)


def test_moss_audio_weight_mapper_preserves_language_model_prefix() -> None:
    mapped = MossAudioModel.hf_to_vllm_mapper.apply_list(
        [
            "language_model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
            "audio_encoder.layers.0.self_attn.q_proj.weight",
        ]
    )

    assert mapped == [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.lm_head.weight",
        "audio_encoder.layers.0.self_attn.q_proj.weight",
    ]


def test_moss_audio_weight_mapper_matches_hf_index_names() -> None:
    mapped = MossAudioModel.hf_to_vllm_mapper.apply_list(
        [
            "audio_encoder.layers.0.self_attn.q_proj.weight",
            "audio_encoder.layers.0.self_attn.q_proj.bias",
            "audio_encoder.layers.0.self_attn.k_proj.weight",
            "audio_encoder.layers.0.self_attn.v_proj.weight",
            "audio_encoder.layers.0.self_attn.v_proj.bias",
            "audio_encoder.conv1.weight",
            "audio_adapter.gate_proj.weight",
            "deepstack_audio_merger_list.0.down_proj.weight",
            "language_model.layers.0.self_attn.q_proj.weight",
            "language_model.norm.weight",
            "lm_head.weight",
        ]
    )

    assert mapped == [
        "audio_encoder.layers.0.self_attn.q_proj.weight",
        "audio_encoder.layers.0.self_attn.q_proj.bias",
        "audio_encoder.layers.0.self_attn.k_proj.weight",
        "audio_encoder.layers.0.self_attn.v_proj.weight",
        "audio_encoder.layers.0.self_attn.v_proj.bias",
        "audio_encoder.conv1.weight",
        "audio_adapter.gate_proj.weight",
        "deepstack_audio_merger_list.0.down_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]
