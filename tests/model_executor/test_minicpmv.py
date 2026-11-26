# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModel, PreTrainedConfig

from vllm.model_executor.models.minicpmv import (
    MiniCPMV2_0,
    MiniCPMV2_5,
    MiniCPMV2_6,
    _build_llama_text_config,
)
from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    _expand_flags,
    _select_vision_items,
    _vision_downsample_mode,
)

pytestmark = pytest.mark.skip_global_cleanup


def make_model() -> MiniCPMV2_6:
    return object.__new__(MiniCPMV2_6)  # type: ignore[type-abstract]  # protocol attrs unset


def new_model_4_6() -> MiniCPMV4_6ForConditionalGeneration:
    return object.__new__(
        MiniCPMV4_6ForConditionalGeneration  # type: ignore[type-abstract]  # protocol attrs unset
    )


def make_model_4_6(
    monkeypatch: pytest.MonkeyPatch,
) -> MiniCPMV4_6ForConditionalGeneration:
    monkeypatch.setattr(
        MiniCPMV4_6ForConditionalGeneration,
        "_process_vision_input",
        lambda self, vision_input, use_vit_merger=None: [vision_input["image_embeds"]],
    )
    return new_model_4_6()


def test_video_embeds_reach_the_vision_parser():
    embeds = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

    modalities = make_model()._parse_and_validate_multimodal_inputs(video_embeds=embeds)

    video_input = modalities["videos"]
    assert video_input is not None
    assert video_input["type"] == "image_embeds"
    assert torch.equal(video_input["image_embeds"], embeds)


def test_video_embeds_are_embedded_by_4_6(monkeypatch):
    embeds = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)

    embeddings = make_model_4_6(monkeypatch).embed_multimodal(video_embeds=embeds)

    assert len(embeddings) == 1
    assert torch.equal(embeddings[0], embeds)


def test_image_and_video_embeds_stay_in_their_own_modality():
    image_embeds = torch.zeros(1, 4, 8)
    video_embeds = torch.ones(1, 4, 8)

    modalities = make_model()._parse_and_validate_multimodal_inputs(
        image_embeds=image_embeds,
        video_embeds=video_embeds,
    )

    image_input = modalities["images"]
    video_input = modalities["videos"]
    assert image_input is not None
    assert video_input is not None
    assert torch.equal(image_input["image_embeds"], image_embeds)
    assert torch.equal(video_input["image_embeds"], video_embeds)


class TestPerItemMergerFlags:
    """The merger flag is per request but items from several requests can be
    batched together, so it has to be applied per item."""

    def test_expand_flags_broadcasts_a_single_value(self):
        assert _expand_flags(None, 3) == [None, None, None]
        assert _expand_flags(torch.tensor([True]), 3) == [True, True, True]
        assert _expand_flags(True, 2) == [True, True]

    def test_expand_flags_keeps_per_item_values(self):
        flags = [torch.tensor([True]), torch.tensor([False])]

        assert _expand_flags(flags, 2) == [True, False]

    def test_expand_flags_falls_back_on_a_shape_mismatch(self):
        # Too many values to map onto the items: keep one decision for the
        # batch rather than failing the request.
        assert _expand_flags([True, False, True], 2) == [True, True]

    def test_vision_downsample_mode(self):
        assert _vision_downsample_mode(None) is None
        assert _vision_downsample_mode(True) == "16x"
        assert _vision_downsample_mode(False) == "4x"

    def test_select_vision_items_slices_by_item(self):
        # 3 items holding 2, 1 and 2 slices; one channel, one patch, width 4.
        pixel_values = [torch.full((1, 1, 4), float(i)) for i in range(5)]
        tgt_sizes = torch.arange(10, dtype=torch.long).reshape(5, 2)
        image_input = {
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "num_slices": torch.tensor([2, 1, 2]),
        }

        selected = _select_vision_items(image_input, [2], [2, 1, 2])

        assert len(selected["pixel_values"]) == 2
        assert bool(selected["pixel_values"][0][0, 0, 0] == 3.0)
        assert bool(selected["pixel_values"][1][0, 0, 0] == 4.0)
        assert selected["num_slices"].tolist() == [2]
        assert torch.equal(selected["tgt_sizes"], tgt_sizes[3:5])

    def test_select_vision_items_preserves_order(self):
        pixel_values = [torch.full((1, 1, 4), float(i)) for i in range(3)]
        image_input = {
            "pixel_values": pixel_values,
            "tgt_sizes": torch.zeros(3, 2, dtype=torch.long),
            "num_slices": torch.tensor([1, 1, 1]),
        }

        selected = _select_vision_items(image_input, [2, 0], [1, 1, 1])

        assert selected["num_slices"].tolist() == [1, 1]
        assert bool(selected["pixel_values"][0][0, 0, 0] == 2.0)
        assert bool(selected["pixel_values"][1][0, 0, 0] == 0.0)

    def test_process_vision_input_applies_flags_per_item(self, monkeypatch):
        calls: list[str | None] = []

        def fake_get_vision_hidden_states(self, data, downsample_mode=None):
            calls.append(downsample_mode)
            tag = 1.0 if downsample_mode == "16x" else 0.0
            return [torch.full((1,), tag) for _ in data["pixel_values"]]

        monkeypatch.setattr(
            MiniCPMV4_6ForConditionalGeneration,
            "get_vision_hidden_states",
            fake_get_vision_hidden_states,
        )
        model = new_model_4_6()

        image_input = {
            "type": "pixel_values",
            "pixel_values": [torch.zeros(1, 1, 4) for _ in range(3)],
            "tgt_sizes": torch.zeros(3, 2, dtype=torch.long),
            "num_slices": torch.tensor([1, 1, 1]),
        }

        result = model._process_vision_input(
            image_input,
            use_vit_merger=[
                torch.tensor([True]),
                torch.tensor([False]),
                torch.tensor([True]),
            ],
        )

        # Each item keeps its own mode even though they share one call.
        assert [float(r[0]) for r in result] == [1.0, 0.0, 1.0]
        # Mixed flags are encoded as separate groups, not one collapsed batch.
        assert calls == ["16x", "4x"]

    def test_process_vision_input_uses_one_group_for_uniform_flags(self, monkeypatch):
        calls: list[str | None] = []

        def fake_get_vision_hidden_states(self, data, downsample_mode=None):
            calls.append(downsample_mode)
            return [torch.zeros(1) for _ in data["pixel_values"]]

        monkeypatch.setattr(
            MiniCPMV4_6ForConditionalGeneration,
            "get_vision_hidden_states",
            fake_get_vision_hidden_states,
        )
        model = new_model_4_6()

        image_input = {
            "type": "pixel_values",
            "pixel_values": [torch.zeros(1, 1, 4) for _ in range(2)],
            "tgt_sizes": torch.zeros(2, 2, dtype=torch.long),
            "num_slices": torch.tensor([1, 1]),
        }

        model._process_vision_input(image_input, use_vit_merger=torch.tensor([True]))

        # The common case still makes a single encoder call.
        assert calls == ["16x"]


@pytest.mark.parametrize(
    ("extra_fields", "expected_head_dim", "expected_rope_type"),
    [
        (
            {
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 1,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 128256,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0,
            },
            128,
            "default",
        ),
        (
            {
                "hidden_size": 2560,
                "intermediate_size": 10240,
                "num_hidden_layers": 1,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "vocab_size": 73448,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "logit_scale": 0.5,
                "rope_parameters": {
                    "rope_type": "longrope",
                    "rope_theta": 10000.0,
                    "short_factor": [1.0] * 64,
                    "long_factor": [1.0] * 64,
                    "original_max_position_embeddings": 32786,
                    "factor": 1.0,
                },
            },
            128,
            "longrope",
        ),
    ],
)
def test_build_llama_text_config_for_minicpmv_versions(
    extra_fields: dict,
    expected_head_dim: int,
    expected_rope_type: str,
):
    """Ensure 2.5 and 4.0 (head_dim=128 + longrope) build valid LlamaModels."""
    hf_config = PreTrainedConfig(
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        **extra_fields,
    )

    llama_config = _build_llama_text_config(hf_config)

    assert llama_config.tie_word_embeddings is False
    assert llama_config.head_dim == expected_head_dim
    assert getattr(llama_config, "logit_scale", None) == extra_fields.get("logit_scale")
    expected_theta = extra_fields.get(
        "rope_theta", extra_fields.get("rope_parameters", {}).get("rope_theta")
    )
    assert llama_config.rope_parameters["rope_theta"] == expected_theta

    with torch.device("meta"):
        model = AutoModel.from_config(llama_config)

    assert model.rotary_emb.rope_type == expected_rope_type
    assert model.layers[0].self_attn.q_proj.weight.shape == (
        extra_fields["num_attention_heads"] * expected_head_dim,
        extra_fields["hidden_size"],
    )


def test_minicpmv_forward_dispatch():
    """Ensure MiniCPMV2_0 calls llm.model directly while 2.5 calls llm."""
    calls: list[tuple[str, object]] = []

    class DummyLLM:
        def model(self, **kwargs):
            calls.append(("llm.model", kwargs["inputs_embeds"]))
            return torch.zeros(1)

        def __call__(self, **kwargs):
            calls.append(("llm", kwargs["inputs_embeds"]))
            return torch.ones(1)

    for cls, expected_target in ((MiniCPMV2_0, "llm.model"), (MiniCPMV2_5, "llm")):
        model = object.__new__(cls)
        torch.nn.Module.__init__(model)
        model.llm = DummyLLM()  # type: ignore[assignment]

        embeds = torch.zeros(1, 2)
        model.forward(
            input_ids=None,
            positions=torch.zeros(1, dtype=torch.long),
            inputs_embeds=embeds,
        )
        assert calls[-1][0] == expected_target
        assert calls[-1][1] is embeds
        if cls is MiniCPMV2_0:
            # MiniCPMV2_0 clears inputs_embeds when intermediate_tensors is set.
            model.forward(
                input_ids=None,
                positions=torch.zeros(1, dtype=torch.long),
                intermediate_tensors=object(),  # type: ignore[arg-type]
                inputs_embeds=embeds,
            )
            assert calls[-1] == ("llm.model", None)
