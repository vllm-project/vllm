# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from types import SimpleNamespace

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
    MOSS_AUDIO_PLACEHOLDER,
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
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)

    def batch_decode(self, batch_token_ids, **kwargs):
        return [self.decode(token_ids, **kwargs) for token_ids in batch_token_ids]


def _mm_processor(processor):
    mm_processor = object.__new__(MossAudioMultiModalProcessor)
    mm_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: processor,
        get_tokenizer=lambda: processor.tokenizer,
    )
    return mm_processor


def _processing_context(model=".", revision=None, mm_processor_kwargs=None):
    tokenizer = _Tokenizer()
    mm_config = MultiModalConfig(mm_processor_kwargs=mm_processor_kwargs)
    hf_config = MossAudioConfig(language_config=Qwen3Config())
    model_config = SimpleNamespace(
        model=model,
        revision=revision,
        multimodal_config=mm_config,
        hf_config=hf_config,
        get_multimodal_config=lambda: mm_config,
    )
    return SimpleNamespace(
        tokenizer=tokenizer,
        model_config=model_config,
        get_tokenizer=lambda: tokenizer,
        get_hf_config=lambda: hf_config,
        get_merged_mm_kwargs=mm_config.merge_mm_processor_kwargs,
    )


def _vllm_config(tensor_parallel_size=1, pipeline_parallel_size=1):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=MossAudioConfig(language_config=Qwen3Config()),
            multimodal_config=None,
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
    )


class _FakeAudioEncoder:
    dtype = torch.float32

    def __init__(self, deepstack_layers=0):
        self.deepstack_layers = deepstack_layers
        self.output_deepstack_hidden_states = None

    def __call__(self, audio_data, *, feature_lens, output_deepstack_hidden_states):
        del audio_data
        self.output_deepstack_hidden_states = output_deepstack_hidden_states
        lengths = MossAudioEncoder._compute_downsampled_length(feature_lens)
        hidden_states = torch.ones(1, int(lengths.sum().item()), 8)
        if not output_deepstack_hidden_states:
            return hidden_states, None
        return hidden_states, [hidden_states * scale
                               for scale in range(2, 2 + self.deepstack_layers)]


def _mel_config(mel_dim, mel_sr, mel_hop_length, mel_n_fft):
    return {
        "mel_dim": mel_dim,
        "mel_sr": mel_sr,
        "mel_hop_length": mel_hop_length,
        "mel_n_fft": mel_n_fft,
    }


def _patch_tensor_parallel_for_linear_layers(monkeypatch):
    import vllm.model_executor.layers.linear as linear_layers
    import vllm.model_executor.models.moss_audio as moss_audio_module
    import vllm.model_executor.parameter as parameter_module

    for module in (moss_audio_module, linear_layers, parameter_module):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: 1
        )
    monkeypatch.setattr(linear_layers, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)


@pytest.mark.parametrize(
    ("prompt", "prefix"),
    [
        (
            f"before {MOSS_AUDIO_PLACEHOLDER} after",
            [*[ord(char) for char in "before "], MOSS_AUDIO_BOS_TOKEN_ID],
        ),
        (
            f"before {MOSS_AUDIO_BOS_TOKEN}{MOSS_AUDIO_TOKEN}"
            f"{MOSS_AUDIO_TOKEN}{MOSS_AUDIO_EOS_TOKEN} after",
            [*[ord(char) for char in "before "], MOSS_AUDIO_BOS_TOKEN_ID],
        ),
        ("Describe this audio.", [MOSS_AUDIO_BOS_TOKEN_ID]),
    ],
)
def test_moss_audio_processor_expands_audio_placeholders(prompt, prefix):
    raw_mel_len = 17
    processed = MossAudioProcessor(_Tokenizer())(
        text=prompt, audio=[torch.zeros(160 * raw_mel_len)]
    )
    input_ids = processed["input_ids"][0].tolist()

    assert input_ids[: len(prefix)] == prefix
    assert input_ids.count(MOSS_AUDIO_BOS_TOKEN_ID) == 1
    assert input_ids.count(MOSS_AUDIO_EOS_TOKEN_ID) == 1
    assert input_ids.count(MOSS_AUDIO_TOKEN_ID) == (
        MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)
    )
    assert processed["audio_data"].shape == (1, 128, raw_mel_len)
    assert processed["audio_data_seqlens"].tolist() == [raw_mel_len]


def test_moss_audio_processor_preserves_placeholder_without_audio():
    processed = MossAudioProcessor(_Tokenizer())(
        text=f"before {MOSS_AUDIO_PLACEHOLDER} after"
    )

    assert processed["input_ids"][0].tolist() == [
        *[ord(char) for char in "before "],
        MOSS_AUDIO_BOS_TOKEN_ID,
        MOSS_AUDIO_TOKEN_ID,
        MOSS_AUDIO_EOS_TOKEN_ID,
        *[ord(char) for char in " after"],
    ]
    assert "audio_data" not in processed
    assert "audio_data_seqlens" not in processed


def test_moss_audio_processor_uses_config_overrides():
    processor = MossAudioProcessor(
        _Tokenizer(),
        audio_token_id=901,
        audio_start_id=902,
        audio_end_id=903,
        enable_time_marker=True,
        mel_config=_mel_config(80, 8000, 80, 200),
    )
    raw_mel_len = 320
    processed = processor(
        text=MOSS_AUDIO_PLACEHOLDER, audio=[torch.zeros(80 * raw_mel_len)]
    )
    input_ids = processed["input_ids"][0].tolist()

    assert input_ids[0] == 902
    assert input_ids[-1] == 903
    assert input_ids.count(901) == MossAudioEncoder.compute_num_audio_tokens(
        raw_mel_len
    )
    assert MOSS_AUDIO_BOS_TOKEN_ID not in input_ids
    assert MOSS_AUDIO_EOS_TOKEN_ID not in input_ids
    assert processed["audio_data"].shape == (1, 80, raw_mel_len)
    assert processed["audio_data"].dtype == torch.float32
    assert processor.mel_config["mel_n_fft"] == 200


def test_moss_audio_prompt_updates_match_chat_template_tokenization():
    class _ChatTemplateTokenizer(_Tokenizer):
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if text == MOSS_AUDIO_PLACEHOLDER:
                return [11, 12, 13, 14]
            if text == f"{MOSS_AUDIO_PLACEHOLDER}\n":
                return [11, 12, 13, 99]
            if text == "\n":
                return [99]
            return super().encode(text)

    processor = MossAudioProcessor(_ChatTemplateTokenizer(), enable_time_marker=True)
    mm_processor = _mm_processor(processor)
    audio_len = 320
    updates = mm_processor._get_prompt_updates(
        mm_items=None,
        hf_processor_mm_kwargs={},
        out_mm_kwargs=SimpleNamespace(
            get_data=lambda: {
                "audio_data_seqlens": torch.tensor([audio_len], dtype=torch.long)
            }
        ),
    )
    resolved = [update.resolve(0) for update in updates]
    full = resolved[0].content.full
    is_embed = resolved[0].content.is_embed(None, full)

    assert [update.target for update in resolved] == [
        [MOSS_AUDIO_BOS_TOKEN_ID, MOSS_AUDIO_TOKEN_ID, MOSS_AUDIO_EOS_TOKEN_ID],
        [11, 12, 13, 14],
        [11, 12, 13, 99],
    ]
    assert full[0] == MOSS_AUDIO_BOS_TOKEN_ID
    assert full[-1] == MOSS_AUDIO_EOS_TOKEN_ID
    assert is_embed.sum().item() == MossAudioEncoder.compute_num_audio_tokens(
        audio_len
    )
    assert len(full) > is_embed.sum().item() + 2
    assert resolved[1].content.full == full
    assert resolved[2].content.full == [*full, 99]

    groups = mm_processor._bind_and_group_updates(updates, {"audio": 1})
    new_ids, placeholders = mm_processor._apply_prompt_updates(
        [1, 11, 12, 13, 99, 2], groups
    )

    assert new_ids == [1, *resolved[2].content.full, 2]
    assert len(placeholders["audio"]) == 1
    assert placeholders["audio"][0].start_idx == 1
    assert placeholders["audio"][0].length == len(resolved[2].content.full)
    assert not bool(placeholders["audio"][0].is_embed[-1])


@pytest.mark.parametrize(
    ("filename", "config", "expected"),
    [
        ("processor_config.json", {
            "audio_token_id": 301, "audio_start_id": 302, "audio_end_id": 303,
            "enable_time_marker": True, "mel_config": _mel_config(80, 8000, 80, 200),
            "processor_class": "PalomarProcessor",
        }, (301, 302, 303, True, 80, 8000, 80, 200)),
        ("preprocessor_config.json", {
            "audio_token_id": 401,
            "mel_config": {"sampling_rate": 12000, "feature_size": 64,
                           "hop_length": 120, "n_fft": 240},
        }, (401, MOSS_AUDIO_BOS_TOKEN_ID, MOSS_AUDIO_EOS_TOKEN_ID, False, 64,
            12000, 120, 240)),
    ],
)
def test_moss_audio_processing_info_reads_processor_configs(
    tmp_path, filename, config, expected
):
    (tmp_path / filename).write_text(json.dumps(config))
    processor = MossAudioProcessingInfo(
        _processing_context(model=str(tmp_path))
    ).get_hf_processor()

    assert (
        processor.audio_token_id,
        processor.audio_start_id,
        processor.audio_end_id,
        processor.enable_time_marker,
        *processor.mel_config.values(),
    ) == expected


def test_moss_audio_processing_info_merges_processor_kwargs(tmp_path):
    (tmp_path / "processor_config.json").write_text(json.dumps({
        "audio_token_id": 501, "audio_start_id": 502, "audio_end_id": 503,
        "enable_time_marker": False, "mel_config": _mel_config(80, 8000, 80, 200),
    }))
    info = MossAudioProcessingInfo(
        _processing_context(
            model=str(tmp_path),
            mm_processor_kwargs={
                "audio_token_id": 601, "mel_config": {"mel_dim": 96, "mel_sr": 12000}
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
    assert processor.mel_config == _mel_config(96, 12000, 120, 200)


def test_moss_audio_processing_info_cache_key_includes_processor_config():
    info = MossAudioProcessingInfo(_processing_context())
    default_processor = info.get_hf_processor()
    processor = info.get_hf_processor(audio_token_id=701)

    assert info.get_hf_processor(enable_time_marker=False) is default_processor
    assert info.get_hf_processor(audio_token_id=701) is processor
    assert info.get_hf_processor(audio_token_id=702) is not processor
    assert info.get_hf_processor(
        audio_token_id=701, enable_time_marker=True
    ) is not processor
    assert info.get_hf_processor(
        audio_token_id=701, mel_config={"mel_dim": 80}
    ) is not processor


def test_moss_audio_metadata_contracts():
    assert set(_moss_audio_field_config({})) == {"audio_data", "audio_data_seqlens"}
    assert MossAudioModel.get_placeholder_str("audio", 0) == MOSS_AUDIO_PLACEHOLDER
    assert supports_multimodal(MossAudioModel)
    assert is_text_generation_model(MossAudioModel)
    assert not supports_pp(MossAudioModel)
    assert _MULTIMODAL_MODELS["MossAudioModel"] == ("moss_audio", "MossAudioModel")
    info = _MULTIMODAL_EXAMPLE_MODELS["MossAudioModel"]
    assert info.default == "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
    assert set(info.extras) == {"4b-thinking", "8b-instruct", "8b-thinking"}

    raw_mel_len = math.ceil((DEFAULT_MAX_AUDIO_SECONDS * 16000) / 160)
    processing_info = MossAudioProcessingInfo(_processing_context())
    assert processing_info.get_mm_max_tokens_per_item(
        seq_len=2048, mm_counts={"audio": 1}
    ) == {"audio": MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)}


@pytest.mark.parametrize(("hidden_size", "head_dim"), [(2560, 80), (4096, 128)])
def test_moss_audio_arch_config_supports_4b_and_8b_language_configs(
    hidden_size, head_dim
):
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
    )

    language_config = Qwen3Config(
        hidden_size=hidden_size,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=head_dim,
        num_hidden_layers=36,
        vocab_size=151936,
        max_position_embeddings=32768,
    )
    config = MossAudioConfig(language_config=language_config)
    arch_config = MODEL_ARCH_CONFIG_CONVERTORS["moss_audio"](
        config, config
    ).convert()

    assert config.get_text_config() is language_config
    assert config.hidden_size == arch_config.hidden_size == hidden_size
    assert config.num_attention_heads == arch_config.total_num_attention_heads == 32
    assert config.num_key_value_heads == arch_config.total_num_kv_heads == 8
    assert config.head_dim == arch_config.head_size == head_dim
    assert arch_config.total_num_hidden_layers == 36
    assert arch_config.vocab_size == 151936
    assert arch_config.derived_max_model_len_and_key == (
        32768,
        "language_config.max_position_embeddings",
    )


def test_moss_audio_error_paths():
    for kwargs, message in (
        ({"tensor_parallel_size": 2}, "tensor_parallel_size=1"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size=1"),
    ):
        with pytest.raises(ValueError, match=message):
            MossAudioModel(vllm_config=_vllm_config(**kwargs))

    model = object.__new__(MossAudioModel)
    with pytest.raises(ValueError, match="DeepStack audio token count mismatch"):
        model._cache_deepstack_input_embeds(
            inputs_embeds=torch.zeros(4, 8),
            deepstack_embeddings=((torch.ones(1, 8),),),
            is_multimodal=torch.tensor([False, True, True, False]),
        )

    with pytest.raises(ValueError, match="too short"):
        MossAudioProcessor(_Tokenizer())(
            text=MOSS_AUDIO_PLACEHOLDER, audio=[torch.empty(0)]
        )
    with pytest.raises(ValueError, match="too short"):
        model._parse_and_validate_audio_input(
            audio_data=torch.zeros(1, 128, 1),
            audio_data_seqlens=torch.tensor([0], dtype=torch.long),
        )


@pytest.mark.parametrize("deepstack_scales", [(), (7, 11)])
def test_moss_audio_embed_multimodal_packs_by_audio(deepstack_scales):
    model = object.__new__(MossAudioModel)
    model.audio_encoder = _FakeAudioEncoder(len(deepstack_scales))
    model.audio_adapter = lambda hidden_states: hidden_states * 5
    model.deepstack_audio_merger_list = [
        lambda hidden_states, scale=scale: hidden_states * scale
        for scale in deepstack_scales
    ]
    model.deepstack_input_embeds = None

    embeddings = model.embed_multimodal(
        audio_data=torch.zeros(2, 128, 9),
        audio_data_seqlens=torch.tensor([8, 9], dtype=torch.long),
    )

    assert model.audio_encoder.output_deepstack_hidden_states is bool(
        deepstack_scales
    )
    assert [embeds.shape for embeds in embeddings] == [
        torch.Size([1, 8 * (1 + len(deepstack_scales))]),
        torch.Size([2, 8 * (1 + len(deepstack_scales))]),
    ]
    if not deepstack_scales:
        assert model.deepstack_input_embeds is None
        return

    main_embeddings, deepstack_embeddings = model._split_multimodal_embeddings(
        embeddings, hidden_size=8
    )
    assert [embeds.shape for embeds in main_embeddings] == [torch.Size([1, 8]),
                                                            torch.Size([2, 8])]
    assert [[e.shape for e in layer] for layer in deepstack_embeddings] == [
        [torch.Size([1, 8]), torch.Size([2, 8])] for _ in deepstack_scales
    ]
    assert torch.equal(main_embeddings[0], torch.full((1, 8), 5.0))
    for idx, scale in enumerate(deepstack_scales):
        assert torch.equal(
            deepstack_embeddings[idx][0],
            torch.full((1, 8), float((idx + 2) * scale)),
        )


def test_moss_audio_embed_input_ids_caches_packed_deepstack():
    class _FakeLanguageModel:
        def embed_input_ids(self, input_ids):
            return torch.zeros(input_ids.shape[0], 8)

    model = object.__new__(MossAudioModel)
    model.language_model = _FakeLanguageModel()
    model.deepstack_audio_merger_list = [object(), object()]
    model.deepstack_input_embeds = None
    multimodal_embeddings = (
        torch.cat([torch.full((1, 8), x) for x in (5.0, 14.0, 33.0)], dim=-1),
        torch.cat([torch.full((2, 8), x) for x in (7.0, 22.0, 44.0)], dim=-1),
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
    tensors = model.deepstack_input_embeds.tensors
    assert set(tensors) == {"deepstack_input_embeds_0", "deepstack_input_embeds_1"}
    for tensor in tensors.values():
        assert tensor[is_multimodal].abs().sum() > 0
        assert torch.equal(tensor[~is_multimodal], torch.zeros(2, 8))


def test_moss_audio_encoder_loads_realistic_attention_weight_names(monkeypatch):
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
    with set_current_vllm_config(
        VllmConfig(device_config=DeviceConfig(device="cpu"))
    ):
        encoder = MossAudioEncoder(config)

    attention = encoder.layers[0].self_attn
    assert all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj"))
    assert hasattr(attention, "out_proj")
    assert not hasattr(attention, "qkv")
    assert attention.k_proj.bias is None
    assert attention(
        torch.randn(1, 3, config.d_model), torch.ones(1, 3, dtype=torch.bool)
    ).shape == (1, 3, config.d_model)

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
    weights = {
        name: torch.full_like(params[name], fill_value=float(i + 1))
        for i, name in enumerate(weight_names)
    }

    loaded = AutoWeightsLoader(encoder).load_weights(weights.items())

    assert "load_weights" not in MossAudioEncoder.__dict__
    assert loaded == set(weight_names)
    assert not any(".qkv." in name for name in loaded)
    assert torch.equal(params["layers.0.self_attn.q_proj.weight"], weights[
        "layers.0.self_attn.q_proj.weight"
    ])


def test_moss_audio_encoder_out_proj_matches_hf_identity_when_dims_equal(
    monkeypatch,
):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig

    _patch_tensor_parallel_for_linear_layers(monkeypatch)

    def make_config(output_dim):
        return MossAudioEncoderConfig(
            d_model=8,
            output_dim=output_dim,
            num_mel_bins=8,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            downsample_hidden_size=2,
            deepstack_encoder_layer_indexes=[],
        )

    with set_current_vllm_config(
        VllmConfig(device_config=DeviceConfig(device="cpu"))
    ):
        identity_encoder = MossAudioEncoder(make_config(output_dim=8))
        projected_encoder = MossAudioEncoder(make_config(output_dim=4))

    identity_params = dict(identity_encoder.named_parameters())
    projected_params = dict(projected_encoder.named_parameters())

    assert isinstance(identity_encoder.out_proj, torch.nn.Identity)
    assert "out_proj.weight" not in identity_params
    assert not isinstance(projected_encoder.out_proj, torch.nn.Identity)
    assert "out_proj.weight" in projected_params


def test_moss_audio_weight_contracts():
    model = object.__new__(MossAudioModel)
    torch.nn.Module.__init__(model)
    assert model.load_weights(
        [("audio_encoder.embed_positions.inv_timescales", torch.empty(1))]
    ) == set()

    names = [
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

    assert MossAudioModel.hf_to_vllm_mapper.apply_list(names) == [
        *names[:8],
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]
