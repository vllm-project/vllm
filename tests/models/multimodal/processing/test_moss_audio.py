# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config

from vllm.model_executor.models.interfaces import SupportsLoRA, supports_lora
from vllm.model_executor.models.moss_audio import (
    MOSS_AUDIO_BOS_TOKEN,
    MOSS_AUDIO_BOS_TOKEN_ID,
    MOSS_AUDIO_EOS_TOKEN,
    MOSS_AUDIO_EOS_TOKEN_ID,
    MOSS_AUDIO_PLACEHOLDER,
    MOSS_AUDIO_TOKEN,
    MOSS_AUDIO_TOKEN_ID,
    GatedMLP,
    MossAudioConfig,
    MossAudioDummyInputsBuilder,
    MossAudioEncoder,
    MossAudioEncoderConfig,
    MossAudioModel,
    MossAudioMultiModalProcessor,
    MossAudioProcessingInfo,
    MossAudioProcessor,
    MossQwen3ForCausalLM,
    MossQwen3Model,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import batched_tensors_equal
from vllm.sequence import IntermediateTensors


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)

    def batch_decode(self, batch_token_ids, **kwargs):
        return [self.decode(token_ids, **kwargs) for token_ids in batch_token_ids]


class _MMConfig:
    enable_mm_embeds = False
    mm_processor_cache_gb = 1

    def merge_mm_processor_kwargs(self, kwargs):
        return dict(kwargs)

    def get_limit_per_prompt(self, modality):
        del modality
        return 3


class _ModelConfig:
    def __init__(self):
        self.model = "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
        self.revision = None
        self.max_model_len = 4096
        self.encoder_config = {}
        self.dtype = torch.float32
        self.hf_config = MossAudioConfig(language_config=Qwen3Config())
        self.multimodal_config = _MMConfig()

    def get_multimodal_config(self):
        return self.multimodal_config

    def get_inputs_embeds_size(self):
        return None


class _ProcessingContext:
    def __init__(self):
        self.model_config = _ModelConfig()
        self.tokenizer = _Tokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def get_hf_config(self):
        return self.model_config.hf_config

    def get_mm_config(self):
        return self.model_config.get_multimodal_config()

    def get_merged_mm_kwargs(self, kwargs):
        return self.get_mm_config().merge_mm_processor_kwargs(kwargs)

    def call_hf_processor(self, hf_processor, data, kwargs):
        merged_kwargs = self.get_merged_mm_kwargs(kwargs)
        merged_kwargs.setdefault("return_tensors", "pt")
        return hf_processor(**data, **merged_kwargs)


class _TestMossAudioProcessingInfo(MossAudioProcessingInfo):
    def _get_processor_config_defaults(self):
        return {}


def _vllm_config(tensor_parallel_size=1, pipeline_parallel_size=1, hf_config=None):
    if hf_config is None:
        hf_config = MossAudioConfig(language_config=Qwen3Config())
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
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
        self.input_shape = None
        self.feature_lens = None

    def __call__(self, audio_data, *, feature_lens, output_deepstack_hidden_states):
        self.input_shape = tuple(audio_data.shape)
        self.feature_lens = feature_lens.detach().cpu().clone()
        self.output_deepstack_hidden_states = output_deepstack_hidden_states
        lengths = MossAudioEncoder._compute_downsampled_length(feature_lens)
        hidden_states = torch.ones(1, int(lengths.sum().item()), 8)
        if not output_deepstack_hidden_states:
            return hidden_states, None
        return hidden_states, [
            hidden_states * scale for scale in range(2, 2 + self.deepstack_layers)
        ]


def _patch_tensor_parallel_for_linear_layers(monkeypatch, tp_size=1, tp_rank=0):
    import vllm.model_executor.layers.linear as linear_layers
    import vllm.model_executor.models.moss_audio as moss_audio_module
    import vllm.model_executor.parameter as parameter_module

    for module in (moss_audio_module, linear_layers, parameter_module):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: tp_size
        )
    monkeypatch.setattr(
        linear_layers, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(
        linear_layers, "tensor_model_parallel_all_reduce", lambda tensor: tensor
    )


def _build_moss_audio_processor(cache=None):
    ctx = _ProcessingContext()
    info = _TestMossAudioProcessingInfo(ctx)
    return (
        MossAudioMultiModalProcessor(
            info,
            MossAudioDummyInputsBuilder(info),
            cache=cache,
        ),
        ctx,
    )


def _assert_mm_inputs_equal(left, right):
    assert left["prompt_token_ids"] == right["prompt_token_ids"]
    assert left["mm_hashes"] == right["mm_hashes"]

    left_placeholder = left["mm_placeholders"]["audio"][0]
    right_placeholder = right["mm_placeholders"]["audio"][0]
    assert left_placeholder.offset == right_placeholder.offset
    assert left_placeholder.length == right_placeholder.length
    assert left_placeholder.is_embed.tolist() == right_placeholder.is_embed.tolist()

    assert batched_tensors_equal(
        left["mm_kwargs"].get_data(),
        right["mm_kwargs"].get_data(),
    )


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


def test_moss_audio_multimodal_processor_handles_token_and_cache_paths():
    raw_mel_len = 17
    audio = np.zeros(160 * raw_mel_len, dtype=np.float32)
    prompt = f"{MOSS_AUDIO_PLACEHOLDER}\nTranscribe this audio."

    baseline_processor, ctx = _build_moss_audio_processor()
    mm_items = baseline_processor.info.parse_mm_data({"audio": [audio]})
    token_prompt = ctx.get_tokenizer().encode(prompt, add_special_tokens=False)

    baseline_text = baseline_processor(
        prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )
    baseline_token = baseline_processor(
        token_prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )

    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    cached_processor, _ = _build_moss_audio_processor(cache=cache)
    cached_text_miss = cached_processor(
        prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )
    cached_text_hit = cached_processor(
        prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )
    cached_token_hit = cached_processor(
        token_prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )

    expected_audio_tokens = MossAudioEncoder.compute_num_audio_tokens(raw_mel_len)
    prompt_token_ids = baseline_text["prompt_token_ids"]
    assert prompt_token_ids.count(MOSS_AUDIO_TOKEN_ID) == expected_audio_tokens
    assert baseline_text["mm_placeholders"]["audio"][0].length == (
        expected_audio_tokens + 2
    )

    _assert_mm_inputs_equal(baseline_text, baseline_token)
    _assert_mm_inputs_equal(baseline_text, cached_text_miss)
    _assert_mm_inputs_equal(baseline_text, cached_text_hit)
    _assert_mm_inputs_equal(baseline_text, cached_token_hit)


def test_moss_audio_supports_language_model_lora_only():
    assert supports_lora(MossAudioModel)

    model = object.__new__(MossAudioModel)
    assert isinstance(model, SupportsLoRA)

    mapping = model.get_mm_mapping()
    assert mapping.language_model == ["language_model."]
    assert mapping.tower_model == []
    assert mapping.connector == []


def test_moss_audio_error_paths():
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


def test_moss_audio_validates_tp_config():
    vllm_config = _vllm_config(tensor_parallel_size=2)
    vllm_config.model_config.hf_config.adapter_hidden_size = 7

    with pytest.raises(ValueError, match="adapter_hidden_size"):
        MossAudioModel(vllm_config=vllm_config)

    vllm_config = _vllm_config(tensor_parallel_size=2)
    vllm_config.model_config.hf_config.audio_config.d_model = 6
    vllm_config.model_config.hf_config.audio_config.encoder_attention_heads = 3
    with pytest.raises(ValueError, match="encoder_attention_heads"):
        MossAudioModel(vllm_config=vllm_config)


def test_moss_audio_rejects_audio_data_list_seqlen_count_mismatch():
    model = object.__new__(MossAudioModel)

    with pytest.raises(ValueError, match="audio_data batch size"):
        model._parse_and_validate_audio_input(
            audio_data=[torch.zeros(128, 8), torch.zeros(128, 11)],
            audio_data_seqlens=torch.tensor([8], dtype=torch.long),
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

    assert model.audio_encoder.output_deepstack_hidden_states is bool(deepstack_scales)
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
    assert [embeds.shape for embeds in main_embeddings] == [
        torch.Size([1, 8]),
        torch.Size([2, 8]),
    ]
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


def _patch_pp_group(monkeypatch, *, first=True, last=True):
    import vllm.model_executor.models.moss_audio as moss_audio_module

    monkeypatch.setattr(
        moss_audio_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=first, is_last_rank=last),
    )


def test_moss_audio_pp_forward_routes_deepstack(monkeypatch):
    for first in (True, False):
        calls: list[dict[str, object]] = []

        def fake_lm(*args, _calls=calls, **kwargs):
            del args
            _calls.append(kwargs)
            return torch.ones(1, 1)

        _patch_pp_group(monkeypatch, first=first)
        model = object.__new__(MossAudioModel)
        torch.nn.Module.__init__(model)
        model.language_model = fake_lm
        cached = IntermediateTensors({"deepstack_input_embeds_0": torch.ones(3, 8)})
        inter = IntermediateTensors(
            {
                "hidden_states": torch.ones(3, 8),
                "residual": torch.zeros(3, 8),
                "deepstack_input_embeds_0": torch.full((3, 8), 5.0),
            }
        )
        inputs_embeds = torch.full((3, 8), 9.0)
        model.deepstack_input_embeds = cached

        model.forward(
            input_ids=None,
            positions=torch.arange(3),
            intermediate_tensors=None if first else inter,
            inputs_embeds=inputs_embeds if first else None,
        )

        kwargs = calls[0]
        assert kwargs["inputs_embeds"] is (inputs_embeds if first else None)
        assert kwargs["deepstack_input_embeds"] is (cached if first else inter)
        assert model.deepstack_input_embeds is None

    calls = []

    def fake_lm_non_first_rank(*args, **kwargs):
        del args
        calls.append(kwargs)
        return torch.ones(1, 1)

    _patch_pp_group(monkeypatch, first=False)
    model = object.__new__(MossAudioModel)
    torch.nn.Module.__init__(model)
    model.language_model = fake_lm_non_first_rank
    model.deepstack_input_embeds = IntermediateTensors({})
    inter = IntermediateTensors(
        {
            "hidden_states": torch.ones(3, 8),
            "residual": torch.zeros(3, 8),
        }
    )

    model.forward(
        input_ids=None,
        positions=torch.arange(3),
        intermediate_tensors=inter,
        inputs_embeds=torch.ones(3, 8),
    )
    assert calls[0]["inputs_embeds"] is None
    assert calls[0]["deepstack_input_embeds"] is inter


def test_moss_qwen3_deepstack_keys_for_pp(monkeypatch):
    class AddOne(torch.nn.Module):
        def forward(self, positions, hidden_states, residual):
            del positions, residual
            return hidden_states + 1, torch.zeros_like(hidden_states)

    def make_model(num_layers, deepstack_layers=None):
        model = object.__new__(MossQwen3Model)
        torch.nn.Module.__init__(model)
        model.start_layer, model.end_layer = 0, num_layers
        model.layers = torch.nn.ModuleList([AddOne() for _ in range(num_layers)])
        model.norm = lambda hidden_states, residual: (hidden_states, residual)
        model._maybe_add_hidden_state = lambda aux, *args: aux
        model.deepstack_inject_layer_indices = (
            range(0) if deepstack_layers is None else deepstack_layers
        )
        return model

    _patch_pp_group(monkeypatch, first=True, last=True)
    output = make_model(3).forward(
        input_ids=None,
        positions=torch.arange(2),
        inputs_embeds=torch.zeros(2, 4),
        deepstack_input_embeds=IntermediateTensors(
            {
                "deepstack_input_embeds_2": torch.full((2, 4), 5.0),
            }
        ),
    )
    assert torch.equal(output, torch.full((2, 4), 8.0))

    _patch_pp_group(monkeypatch, first=True, last=False)
    deepstack = IntermediateTensors(
        {
            "deepstack_input_embeds_0": torch.full((2, 4), 7.0),
            "deepstack_input_embeds_3": torch.full((2, 4), 11.0),
        }
    )
    output = make_model(2, range(4)).forward(
        input_ids=None,
        positions=torch.arange(2),
        inputs_embeds=torch.zeros(2, 4),
        deepstack_input_embeds=deepstack,
    )
    assert isinstance(output, IntermediateTensors)
    assert set(output.tensors) == {
        "hidden_states",
        "residual",
        "deepstack_input_embeds_2",
        "deepstack_input_embeds_3",
    }
    assert torch.equal(output["hidden_states"], torch.full((2, 4), 9.0))
    assert torch.equal(output["deepstack_input_embeds_2"], torch.zeros(2, 4))
    assert output["deepstack_input_embeds_3"] is deepstack["deepstack_input_embeds_3"]

    inner_model = make_model(0, range(2))
    inner_model.make_empty_intermediate_tensors = lambda batch, dtype, device: (
        IntermediateTensors(
            {
                "hidden_states": torch.zeros(batch, 4, dtype=dtype, device=device),
                "residual": torch.zeros(batch, 4, dtype=dtype, device=device),
            }
        )
    )
    language_model = object.__new__(MossQwen3ForCausalLM)
    torch.nn.Module.__init__(language_model)
    language_model.model = inner_model
    language_model.config = SimpleNamespace(hidden_size=4)
    language_model.deepstack_inject_layer_indices = range(2)

    tensors = MossQwen3ForCausalLM.make_empty_intermediate_tensors(
        language_model,
        batch_size=3,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    assert set(tensors.tensors) == {
        "hidden_states",
        "residual",
        "deepstack_input_embeds_0",
        "deepstack_input_embeds_1",
    }
    assert tensors["deepstack_input_embeds_0"].shape == (3, 4)
    assert tensors["deepstack_input_embeds_0"].dtype == torch.float16

    _patch_pp_group(monkeypatch, first=True, last=False)
    forward_tensors = inner_model.forward(
        input_ids=None,
        positions=torch.arange(3),
        inputs_embeds=torch.ones(3, 4, dtype=torch.float16),
        deepstack_input_embeds=None,
    )
    assert isinstance(forward_tensors, IntermediateTensors)
    assert set(forward_tensors.tensors) == set(tensors.tensors)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_moss_audio_gated_mlp_tp_shapes_and_loading(monkeypatch, tp_size):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig

    _patch_tensor_parallel_for_linear_layers(monkeypatch, tp_size=tp_size)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        mlp = GatedMLP(input_size=4, hidden_size=8, output_size=6)

    params = dict(mlp.named_parameters())
    assert params["gate_up_proj.weight"].shape == torch.Size([16 // tp_size, 4])
    assert params["down_proj.weight"].shape == torch.Size([6, 8 // tp_size])

    gate_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    up_weight = torch.arange(100, 132, dtype=torch.float32).reshape(8, 4)
    down_weight = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    loaded = mlp.load_weights(
        [
            ("gate_proj.weight", gate_weight),
            ("up_proj.weight", up_weight),
            ("down_proj.weight", down_weight),
        ]
    )

    assert loaded == {"gate_up_proj.weight", "down_proj.weight"}
    shard = 8 // tp_size
    assert torch.equal(params["gate_up_proj.weight"][:shard], gate_weight[:shard])
    assert torch.equal(params["gate_up_proj.weight"][shard:], up_weight[:shard])
    assert torch.equal(params["down_proj.weight"], down_weight[:, : 8 // tp_size])

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        packed_mlp = GatedMLP(input_size=4, hidden_size=8, output_size=6)
    packed_params = dict(packed_mlp.named_parameters())
    loaded = packed_mlp.load_weights(
        [("gate_up_proj.weight", torch.cat([gate_weight, up_weight], dim=0))]
    )
    assert loaded == {"gate_up_proj.weight"}
    assert torch.equal(
        packed_params["gate_up_proj.weight"][:shard],
        gate_weight[:shard],
    )
    assert torch.equal(
        packed_params["gate_up_proj.weight"][shard:],
        up_weight[:shard],
    )


def test_moss_audio_encoder_loads_realistic_attention_weight_names(monkeypatch):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig

    _patch_tensor_parallel_for_linear_layers(monkeypatch, tp_size=2)
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
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        encoder = MossAudioEncoder(config)

    attention = encoder.layers[0].self_attn
    assert all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj"))
    assert hasattr(attention, "out_proj")
    assert not hasattr(attention, "qkv")
    assert attention.k_proj.bias is None

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
    assert torch.equal(
        params["layers.0.self_attn.q_proj.weight"],
        weights["layers.0.self_attn.q_proj.weight"],
    )
