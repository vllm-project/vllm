# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config

from vllm.config.multimodal import MultiModalConfig
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
    MOSS_AUDIO_PLACEHOLDER,
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

    def get_tokenizer(self) -> object:
        return self.processor.tokenizer


class _ProcessingContext:
    def __init__(
        self,
        *,
        model: str = ".",
        revision: str | None = None,
        mm_processor_kwargs: dict[str, object] | None = None,
    ) -> None:
        self.tokenizer = _Tokenizer()
        self.model_config = _ProcessingModelConfig(
            model=model,
            revision=revision,
            multimodal_config=MultiModalConfig(
                mm_processor_kwargs=mm_processor_kwargs
            ),
            hf_config=MossAudioConfig(language_config=Qwen3Config()),
        )

    def get_tokenizer(self) -> _Tokenizer:
        return self.tokenizer

    def get_hf_config(self) -> MossAudioConfig:
        return self.model_config.hf_config

    def get_merged_mm_kwargs(
        self,
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        return self.model_config.multimodal_config.merge_mm_processor_kwargs(kwargs)


class _ProcessingModelConfig:
    def __init__(
        self,
        *,
        model: str,
        revision: str | None,
        multimodal_config: MultiModalConfig,
        hf_config: MossAudioConfig,
    ) -> None:
        self.model = model
        self.revision = revision
        self.multimodal_config = multimodal_config
        self.hf_config = hf_config

    def get_multimodal_config(self) -> MultiModalConfig:
        return self.multimodal_config


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
    prompt = f"before {MOSS_AUDIO_PLACEHOLDER} after"
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


def test_moss_audio_processor_expands_multi_token_audio_span() -> None:
    processor = MossAudioProcessor(_Tokenizer())
    prompt = (
        f"before {MOSS_AUDIO_BOS_TOKEN}{MOSS_AUDIO_TOKEN}"
        f"{MOSS_AUDIO_TOKEN}{MOSS_AUDIO_EOS_TOKEN} after"
    )
    raw_mel_len = 17
    audio = torch.zeros(160 * raw_mel_len)

    processed = processor(text=prompt, audio=[audio])
    input_ids = processed["input_ids"][0].tolist()
    expected_audio_tokens = MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)

    assert input_ids.count(MOSS_AUDIO_BOS_TOKEN_ID) == 1
    assert input_ids.count(MOSS_AUDIO_EOS_TOKEN_ID) == 1
    assert input_ids.count(MOSS_AUDIO_TOKEN_ID) == expected_audio_tokens


def test_moss_audio_processor_preserves_placeholder_without_audio() -> None:
    processor = MossAudioProcessor(_Tokenizer())
    prompt = f"before {MOSS_AUDIO_PLACEHOLDER} after"

    processed = processor(text=prompt)
    input_ids = processed["input_ids"][0].tolist()

    assert input_ids == [
        *[ord(char) for char in "before "],
        MOSS_AUDIO_BOS_TOKEN_ID,
        MOSS_AUDIO_TOKEN_ID,
        MOSS_AUDIO_EOS_TOKEN_ID,
        *[ord(char) for char in " after"],
    ]
    assert "audio_data" not in processed
    assert "audio_data_seqlens" not in processed


def test_moss_audio_processor_inserts_default_prompt_without_placeholder() -> None:
    processor = MossAudioProcessor(_Tokenizer())
    raw_mel_len = 17
    audio = torch.zeros(160 * raw_mel_len)

    processed = processor(text="Describe this audio.", audio=[audio])
    input_ids = processed["input_ids"][0].tolist()
    expected_audio_tokens = MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)
    expected_prefix = [
        MOSS_AUDIO_BOS_TOKEN_ID,
        *([MOSS_AUDIO_TOKEN_ID] * expected_audio_tokens),
        MOSS_AUDIO_EOS_TOKEN_ID,
        ord("\n"),
    ]

    assert input_ids[: len(expected_prefix)] == expected_prefix
    assert processed["audio_data"].shape == (1, 128, raw_mel_len)


def test_moss_audio_processor_uses_custom_mel_config() -> None:
    processor = MossAudioProcessor(
        _Tokenizer(),
        mel_config={
            "mel_dim": 80,
            "mel_sr": 8000,
            "mel_hop_length": 80,
            "mel_n_fft": 200,
        },
    )
    raw_mel_len = 17
    audio = torch.zeros(80 * raw_mel_len)

    processed = processor(text=MOSS_AUDIO_PLACEHOLDER, audio=[audio])

    assert processor.mel_config == {
        "mel_dim": 80,
        "mel_sr": 8000,
        "mel_hop_length": 80,
        "mel_n_fft": 200,
    }
    assert processed["audio_data"].shape == (1, 80, raw_mel_len)
    assert processed["audio_data"].dtype == torch.float32


def test_moss_audio_processor_uses_manual_audio_token_ids() -> None:
    processor = MossAudioProcessor(
        _Tokenizer(),
        audio_token_id=901,
        audio_start_id=902,
        audio_end_id=903,
        enable_time_marker=True,
    )
    raw_mel_len = 320
    audio = torch.zeros(160 * raw_mel_len)

    processed = processor(text=MOSS_AUDIO_PLACEHOLDER, audio=[audio])
    input_ids = processed["input_ids"][0].tolist()
    audio_token_len = MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)

    assert input_ids[0] == 902
    assert input_ids[-1] == 903
    assert input_ids.count(901) == audio_token_len
    assert MOSS_AUDIO_BOS_TOKEN_ID not in input_ids
    assert MOSS_AUDIO_EOS_TOKEN_ID not in input_ids


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


def test_moss_audio_prompt_updates_match_chat_template_tokenization() -> None:
    class _ChatTemplateTokenizer(_Tokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            placeholder = MOSS_AUDIO_PLACEHOLDER
            if text == placeholder:
                return [11, 12, 13, 14]
            if text == f"{placeholder}\n":
                return [11, 12, 13, 99]
            if text == "\n":
                return [99]
            return super().encode(text, add_special_tokens=add_special_tokens)

    processor = MossAudioProcessor(_ChatTemplateTokenizer())
    mm_processor = object.__new__(MossAudioMultiModalProcessor)
    mm_processor.info = _Info(processor)

    updates = mm_processor._get_prompt_updates(
        mm_items=None,
        hf_processor_mm_kwargs={},
        out_mm_kwargs=_OutKwargs(
            {"audio_data_seqlens": torch.tensor([17], dtype=torch.long)}
        ),
    )
    resolved_updates = [update.resolve(0) for update in updates]

    assert [update.target for update in resolved_updates] == [
        [MOSS_AUDIO_BOS_TOKEN_ID, MOSS_AUDIO_TOKEN_ID, MOSS_AUDIO_EOS_TOKEN_ID],
        [11, 12, 13, 14],
        [11, 12, 13, 99],
    ]
    assert resolved_updates[1].content.full == resolved_updates[0].content.full
    assert resolved_updates[2].content.full == [
        *resolved_updates[0].content.full,
        99,
    ]

    mm_prompt_updates = mm_processor._bind_and_group_updates(updates, {"audio": 1})
    new_token_ids, placeholders = mm_processor._apply_prompt_updates(
        [1, 11, 12, 13, 99, 2],
        mm_prompt_updates,
    )

    assert new_token_ids == [
        1,
        *resolved_updates[2].content.full,
        2,
    ]
    assert len(placeholders["audio"]) == 1
    assert placeholders["audio"][0].start_idx == 1
    assert placeholders["audio"][0].length == len(resolved_updates[2].content.full)
    assert placeholders["audio"][0].is_embed is not None
    assert not bool(placeholders["audio"][0].is_embed[-1])


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


def test_moss_audio_processing_info_reads_processor_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "processor_config.json").write_text(
        """
        {
          "audio_token_id": 301,
          "audio_start_id": 302,
          "audio_end_id": 303,
          "enable_time_marker": true,
          "mel_config": {
            "mel_dim": 80,
            "mel_sr": 8000,
            "mel_hop_length": 80,
            "mel_n_fft": 200
          },
          "processor_class": "PalomarProcessor"
        }
        """,
    )
    info = MossAudioProcessingInfo(_ProcessingContext(model=str(tmp_path)))

    processor = info.get_hf_processor()

    assert processor.audio_token_id == 301
    assert processor.audio_start_id == 302
    assert processor.audio_end_id == 303
    assert processor.enable_time_marker
    assert processor.mel_config == {
        "mel_dim": 80,
        "mel_sr": 8000,
        "mel_hop_length": 80,
        "mel_n_fft": 200,
    }


def test_moss_audio_processing_info_reads_preprocessor_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "preprocessor_config.json").write_text(
        """
        {
          "audio_token_id": 401,
          "mel_config": {
            "sampling_rate": 12000,
            "feature_size": 64,
            "hop_length": 120,
            "n_fft": 240
          }
        }
        """,
    )
    info = MossAudioProcessingInfo(_ProcessingContext(model=str(tmp_path)))

    processor = info.get_hf_processor()

    assert processor.audio_token_id == 401
    assert processor.audio_start_id == MOSS_AUDIO_BOS_TOKEN_ID
    assert processor.audio_end_id == MOSS_AUDIO_EOS_TOKEN_ID
    assert not processor.enable_time_marker
    assert processor.mel_config == {
        "mel_dim": 64,
        "mel_sr": 12000,
        "mel_hop_length": 120,
        "mel_n_fft": 240,
    }


def test_moss_audio_processing_info_merges_processor_kwargs(
    tmp_path: Path,
) -> None:
    (tmp_path / "processor_config.json").write_text(
        """
        {
          "audio_token_id": 501,
          "audio_start_id": 502,
          "audio_end_id": 503,
          "enable_time_marker": false,
          "mel_config": {
            "mel_dim": 80,
            "mel_sr": 8000,
            "mel_hop_length": 80,
            "mel_n_fft": 200
          }
        }
        """,
    )
    info = MossAudioProcessingInfo(
        _ProcessingContext(
            model=str(tmp_path),
            mm_processor_kwargs={
                "audio_token_id": 601,
                "mel_config": {
                    "mel_dim": 96,
                    "mel_sr": 12000,
                },
            },
        )
    )

    processor = info.get_hf_processor(
        audio_end_id=603,
        enable_time_marker=True,
        mel_config={"mel_hop_length": 120},
    )

    assert processor.audio_token_id == 601
    assert processor.audio_start_id == 502
    assert processor.audio_end_id == 603
    assert processor.enable_time_marker
    assert processor.mel_config == {
        "mel_dim": 96,
        "mel_sr": 12000,
        "mel_hop_length": 120,
        "mel_n_fft": 200,
    }


def test_moss_audio_processing_info_cache_key_includes_processor_config() -> None:
    info = MossAudioProcessingInfo(_ProcessingContext())

    processor = info.get_hf_processor(audio_token_id=701)
    same_processor = info.get_hf_processor(audio_token_id=701)
    different_token_processor = info.get_hf_processor(audio_token_id=702)
    different_time_marker_processor = info.get_hf_processor(
        audio_token_id=701,
        enable_time_marker=True,
    )
    different_mel_processor = info.get_hf_processor(
        audio_token_id=701,
        mel_config={"mel_dim": 80},
    )

    assert same_processor is processor
    assert different_token_processor is not processor
    assert different_time_marker_processor is not processor
    assert different_mel_processor is not processor


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
    assert MossAudioModel.get_placeholder_str("audio", 0) == MOSS_AUDIO_PLACEHOLDER
    assert supports_multimodal(MossAudioModel)
    assert is_text_generation_model(MossAudioModel)
    assert not supports_pp(MossAudioModel)


def test_moss_audio_config_exposes_language_config() -> None:
    config = MossAudioConfig(language_config=Qwen3Config(hidden_size=2560))

    assert config.get_text_config() is config.language_config
    assert config.hidden_size == config.language_config.hidden_size
    assert config.num_attention_heads == config.language_config.num_attention_heads
    assert config.num_key_value_heads == config.language_config.num_key_value_heads
    assert config.head_dim == config.language_config.head_dim


def test_moss_audio_arch_config_uses_language_config() -> None:
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
    )

    language_config = Qwen3Config(
        hidden_size=2560,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=80,
        num_hidden_layers=36,
        vocab_size=151936,
        max_position_embeddings=32768,
    )
    config = MossAudioConfig(language_config=language_config)
    convertor = MODEL_ARCH_CONFIG_CONVERTORS["moss_audio"](config, config)

    arch_config = convertor.convert()

    assert arch_config.hidden_size == 2560
    assert arch_config.total_num_attention_heads == 32
    assert arch_config.total_num_kv_heads == 8
    assert arch_config.head_size == 80
    assert arch_config.total_num_hidden_layers == 36
    assert arch_config.vocab_size == 151936
    assert arch_config.derived_max_model_len_and_key == (
        32768,
        "language_config.max_position_embeddings",
    )


@pytest.mark.parametrize(
    (
        "language_config",
        "expected_hidden_size",
        "expected_num_heads",
        "expected_kv_heads",
        "expected_head_dim",
        "expected_vocab_size",
    ),
    [
        (
            Qwen3Config(
                hidden_size=2560,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=80,
                num_hidden_layers=36,
                vocab_size=151936,
                max_position_embeddings=32768,
            ),
            2560,
            32,
            8,
            80,
            151936,
        ),
        (
            Qwen3Config(
                hidden_size=4096,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                num_hidden_layers=36,
                vocab_size=151936,
                max_position_embeddings=32768,
            ),
            4096,
            32,
            8,
            128,
            151936,
        ),
    ],
)
def test_moss_audio_arch_config_supports_4b_and_8b_language_configs(
    language_config: Qwen3Config,
    expected_hidden_size: int,
    expected_num_heads: int,
    expected_kv_heads: int,
    expected_head_dim: int,
    expected_vocab_size: int,
) -> None:
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
    )

    config = MossAudioConfig(language_config=language_config)
    convertor = MODEL_ARCH_CONFIG_CONVERTORS["moss_audio"](config, config)

    arch_config = convertor.convert()

    assert config.hidden_size == expected_hidden_size
    assert arch_config.hidden_size == expected_hidden_size
    assert arch_config.total_num_attention_heads == expected_num_heads
    assert arch_config.total_num_kv_heads == expected_kv_heads
    assert arch_config.head_size == expected_head_dim
    assert arch_config.vocab_size == expected_vocab_size


def test_moss_audio_registry_entries() -> None:
    assert _MULTIMODAL_MODELS["MossAudioModel"] == ("moss_audio", "MossAudioModel")
    info = _MULTIMODAL_EXAMPLE_MODELS["MossAudioModel"]
    assert info.default == "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
    assert info.extras == {
        "4b-thinking": "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        "8b-instruct": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        "8b-thinking": "OpenMOSS-Team/MOSS-Audio-8B-Thinking",
    }


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
    assert [embeds.shape for embeds in multimodal_embeddings] == [
        torch.Size([1, 8]),
        torch.Size([2, 8]),
    ]
    assert model.deepstack_input_embeds is None


def test_moss_audio_embed_multimodal_packs_deepstack_per_audio() -> None:
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
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            del audio_data
            self.output_deepstack_hidden_states = output_deepstack_hidden_states
            audio_lengths = MossAudioEncoder._compute_downsampled_length(feature_lens)
            total_tokens = int(audio_lengths.sum().item())
            hidden_states = torch.ones(1, total_tokens, 8)
            return hidden_states, [hidden_states * 2, hidden_states * 3]

    class _ScaleAdapter:
        def __init__(self, scale: int) -> None:
            self.scale = scale

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states * self.scale

    model = object.__new__(MossAudioModel)
    audio_encoder = _FakeAudioEncoder()
    model.audio_encoder = audio_encoder
    model.audio_adapter = _ScaleAdapter(5)
    model.deepstack_audio_merger_list = [_ScaleAdapter(7), _ScaleAdapter(11)]
    model.deepstack_input_embeds = None

    multimodal_embeddings = model.embed_multimodal(
        audio_data=torch.zeros(2, 128, 9),
        audio_data_seqlens=torch.tensor([8, 9], dtype=torch.long),
    )

    assert audio_encoder.output_deepstack_hidden_states is True
    assert [embeds.shape for embeds in multimodal_embeddings] == [
        torch.Size([1, 24]),
        torch.Size([2, 24]),
    ]
    main_embeddings, deepstack_embeddings = model._split_multimodal_embeddings(
        multimodal_embeddings,
        hidden_size=8,
    )
    assert [embeds.shape for embeds in main_embeddings] == [
        torch.Size([1, 8]),
        torch.Size([2, 8]),
    ]
    assert [[embeds.shape for embeds in layer] for layer in deepstack_embeddings] == [
        [torch.Size([1, 8]), torch.Size([2, 8])],
        [torch.Size([1, 8]), torch.Size([2, 8])],
    ]
    assert torch.equal(main_embeddings[0], torch.full((1, 8), 5.0))
    assert torch.equal(deepstack_embeddings[0][0], torch.full((1, 8), 14.0))
    assert torch.equal(deepstack_embeddings[1][0], torch.full((1, 8), 33.0))


def test_moss_audio_embed_input_ids_caches_packed_deepstack() -> None:
    class _FakeLanguageModel:
        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros(input_ids.shape[0], 8)

    model = object.__new__(MossAudioModel)
    model.language_model = _FakeLanguageModel()
    model.deepstack_audio_merger_list = [object(), object()]
    model.deepstack_input_embeds = None

    multimodal_embeddings = (
        torch.cat(
            [
                torch.full((1, 8), 5.0),
                torch.full((1, 8), 14.0),
                torch.full((1, 8), 33.0),
            ],
            dim=-1,
        ),
        torch.cat(
            [
                torch.full((2, 8), 7.0),
                torch.full((2, 8), 22.0),
                torch.full((2, 8), 44.0),
            ],
            dim=-1,
        ),
    )
    is_multimodal = torch.tensor([False, True, True, True, False])

    inputs_embeds = model.embed_input_ids(
        input_ids=torch.arange(5),
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )

    assert torch.equal(inputs_embeds[1], torch.full((8,), 5.0))
    assert torch.equal(inputs_embeds[2], torch.full((8,), 7.0))
    assert torch.equal(inputs_embeds[3], torch.full((8,), 7.0))
    assert model.deepstack_input_embeds is not None
    deepstack_tensors = model.deepstack_input_embeds.tensors
    assert set(deepstack_tensors) == {
        "deepstack_input_embeds_0",
        "deepstack_input_embeds_1",
    }
    assert torch.equal(
        deepstack_tensors["deepstack_input_embeds_0"][is_multimodal],
        torch.cat([torch.full((1, 8), 14.0), torch.full((2, 8), 22.0)]),
    )
    assert torch.equal(
        deepstack_tensors["deepstack_input_embeds_1"][is_multimodal],
        torch.cat([torch.full((1, 8), 33.0), torch.full((2, 8), 44.0)]),
    )
    assert torch.equal(
        deepstack_tensors["deepstack_input_embeds_0"][~is_multimodal],
        torch.zeros(2, 8),
    )


def test_moss_audio_model_load_weights_skips_sinusoid_buffer() -> None:
    model = object.__new__(MossAudioModel)
    torch.nn.Module.__init__(model)

    weights = [
        (
            "audio_encoder.embed_positions.inv_timescales",
            torch.empty(1),
        ),
    ]
    loaded = model.load_weights(weights)

    assert loaded == set()


def test_moss_audio_rejects_too_short_audio() -> None:
    processor = MossAudioProcessor(_Tokenizer())
    prompt = MOSS_AUDIO_PLACEHOLDER

    with pytest.raises(ValueError, match="too short"):
        processor(text=prompt, audio=[torch.empty(0)])

    model = object.__new__(MossAudioModel)
    with pytest.raises(ValueError, match="too short"):
        model._parse_and_validate_audio_input(
            audio_data=torch.zeros(1, 128, 1),
            audio_data_seqlens=torch.tensor([0], dtype=torch.long),
        )


def test_moss_audio_max_tokens_uses_default_30s_limit() -> None:
    info = MossAudioProcessingInfo(_ProcessingContext())

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
