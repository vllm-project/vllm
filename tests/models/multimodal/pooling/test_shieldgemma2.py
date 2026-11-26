# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import jinja2
import pytest
import requests
from transformers import ShieldGemma2Config

from tests.utils import RemoteOpenAIServer, large_gpu_test
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.config import ModelConfig
from vllm.entrypoints.pooling.classify.protocol import ClassificationResponse
from vllm.model_executor.models.shieldgemma2 import ShieldGemma2ForImageClassification
from vllm.transformers_utils.chat_templates import (
    get_chat_template_override_path,
)
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

MODEL_NAME = "google/shieldgemma-2-4b-it"
MODEL_TYPE = "shieldgemma2"
TEMPLATE_PATH = get_chat_template_override_path(MODEL_TYPE, MODEL_NAME)
BOS = "<bos>"
POLICY = "No Violence/Gore content: example policy."
IMAGE_URL = f"{VLLM_S3_BUCKET_URL}/multimodal_asset/cat_snow.jpg"


def render_prompt(template_source: str) -> str:
    environment = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = environment.from_string(template_source)
    content = [{"type": "image"}, {"type": "text", "text": POLICY}]
    return template.render(
        bos_token=BOS,
        messages=[{"role": "user", "content": content}],
        raise_exception=pytest.fail,
    )


def load_hub_template() -> str:
    config = get_hf_file_to_dict("chat_template.json", MODEL_NAME)
    assert config is not None, f"Cannot download {MODEL_NAME} chat template"
    return config["chat_template"]


def test_template_is_registered_over_the_hub_template():
    assert TEMPLATE_PATH is not None
    assert TEMPLATE_PATH.name == "template_shieldgemma2.jinja"


def test_template_is_hub_template_plus_one_bos():
    # transformers' ShieldGemma2Processor ends up with two <bos>: the Hub
    # template writes one and the tokenizer adds another.
    assert TEMPLATE_PATH is not None
    ours = render_prompt(TEMPLATE_PATH.read_text())
    hub = render_prompt(load_hub_template())
    assert hub.startswith(BOS + "<start_of_turn>user\n<start_of_image>")
    assert ours == BOS + hub


def test_weight_names_map_to_gemma3_layout():
    mapper = ShieldGemma2ForImageClassification.hf_to_vllm_mapper
    cases = {
        "model.language_model.model.layers.0.mlp.up_proj.weight": (
            "language_model.model.layers.0.mlp.up_proj.weight"
        ),
        "model.language_model.lm_head.weight": "language_model.lm_head.weight",
        "model.vision_tower.vision_model.post_layernorm.weight": (
            "vision_tower.vision_model.post_layernorm.weight"
        ),
        "model.multi_modal_projector.mm_input_projection_weight": (
            "multi_modal_projector.mm_input_projection_weight"
        ),
    }
    for original, expected in cases.items():
        assert mapper.map_name(original) == expected


def test_config_selects_two_way_softmax_over_no_yes():
    model_config = ModelConfig(MODEL_NAME, runner="pooling")
    hf_config = model_config.hf_config
    text_config = hf_config.get_text_config()

    assert text_config.method == "from_2_way_softmax"
    assert text_config.classifier_from_token == ["No", "Yes"]
    assert hf_config.num_labels == text_config.num_labels == 1


def test_config_rejects_logit_softcap():
    hf_overrides = {"text_config": {"final_logit_softcapping": 30.0}}

    with pytest.raises(ValueError, match="final_logit_softcapping"):
        ModelConfig(MODEL_NAME, runner="pooling", hf_overrides=hf_overrides)


def test_image_tokens_attend_bidirectionally():
    # Gemma3 image tokens need the bidirectional (mm prefix) mask; it is keyed
    # on hf_config.model_type, so "shieldgemma2" must be listed too.
    config = ShieldGemma2Config()
    convertor = ModelArchConfigConvertorBase(config, config.get_text_config())
    assert convertor.is_mm_prefix_lm()


@pytest.fixture(scope="module")
def server():
    args = ["--runner", "pooling", "--max-model-len", "4096", "--enforce-eager"]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@large_gpu_test(min_gb=24)
def test_classify_image_policy(server: RemoteOpenAIServer):
    # The template opens a turn per content item, so the image goes first.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {"type": "text", "text": POLICY},
            ],
        }
    ]

    response = requests.post(
        server.url_for("classify"),
        json={"model": MODEL_NAME, "messages": messages},
    )
    response.raise_for_status()

    output = ClassificationResponse.model_validate(response.json())
    assert len(output.data) == 1
    assert len(output.data[0].probs) == 1
    assert output.data[0].probs[0] < 0.5
