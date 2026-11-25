# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import httpx
import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from tests.utils import RemoteLaunchRenderServer
from vllm import envs
from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import TimingContext, profiling_media
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.utils import encode_image_url
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config
from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor
from vllm.transformers_utils.processors.deepseek_vl2 import DeepseekVLV2Processor
from vllm.transformers_utils.processors.unlimited_ocr import (
    UnlimitedOCR2Processor,
    UnlimitedOCRProcessor,
)

from ...utils import build_model_context


def _make_tokenizer():
    tokens = ("[UNK]", "<s>", "</s>", "<image>")
    vocab = {token: index for index, token in enumerate(tokens)}
    return TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["<image>"],
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("architecture", "image_size", "count", "view_count"),
    [
        ("vl2", (8, 8), 3, 6),
        ("vl2", (32, 16), 1, 3),
        ("ocr", (8, 8), 2, 2),
        ("ocr2", (32, 16), 1, 3),
    ],
)
def test_image_preprocessing_budget(
    monkeypatch, architecture, image_size, count, view_count
):
    tokenizer = _make_tokenizer()
    prompt = "<image>" * count
    images = [Image.new("RGB", image_size)] * count
    if architecture == "vl2":
        processor = DeepseekVLV2Processor(
            tokenizer=tokenizer,
            candidate_resolutions=((16, 16), (32, 16)),
            patch_size=4,
            downsample_ratio=2,
        )
        inputs = {"text": prompt, "images": [np.array(image) for image in images]}
    else:
        processor = DeepseekOCRProcessor(
            tokenizer=tokenizer,
            base_size=16,
            image_size=16,
            max_crops=2,
            strategy="v1" if architecture == "ocr" else "v2",
        )
        inputs = {"prompt": prompt, "images": images}

    view_bytes = 3 * 16 * 16 * 4
    required_bytes = 2 * view_count if architecture == "vl2" else view_count + 2
    required_bytes *= view_bytes
    limit = "VLLM_MAX_MEDIA_PREPROCESSING_BYTES"
    monkeypatch.setattr(envs, limit, required_bytes - 1)
    with pytest.raises(VLLMValidationError, match="Image preprocessing exceeds"):
        processor(**inputs)
    with profiling_media():
        processor(**inputs)
    with pytest.raises(VLLMValidationError, match="Image preprocessing exceeds"):
        processor(**inputs)

    monkeypatch.setattr(envs, limit, required_bytes)
    at_limit = processor(**inputs)
    monkeypatch.setattr(envs, limit, 0)
    unlimited = processor(**inputs)
    assert len(at_limit["pixel_values"]) + len(at_limit.get("images_crop", ())) == (
        view_count
    )
    torch.testing.assert_close(at_limit.data, unlimited.data, rtol=0, atol=0)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("processor_cls", "kwargs", "page_size", "page_count", "tiles_per_page"),
    [
        (DeepseekOCRProcessor, {}, (2480, 3508), 9, 6),
        (UnlimitedOCRProcessor, {}, (2550, 3300), 29, 0),
        (
            UnlimitedOCR2Processor,
            {"strategy": "v2", "max_crops": 16},
            (2550, 3300),
            6,
            12,
        ),
    ],
)
def test_default_ocr_budget_accepts_multipage_scans(
    monkeypatch, processor_cls, kwargs, page_size, page_count, tiles_per_page
):
    monkeypatch.delenv("VLLM_MAX_MEDIA_PREPROCESSING_BYTES", raising=False)
    processor = processor_cls(tokenizer=_make_tokenizer(), **kwargs)
    page = Image.new("RGB", page_size, "white")
    result = processor(prompt="<image>" * page_count, images=[page] * page_count)
    assert result["pixel_values"].shape == (page_count, 3, 1024, 1024)
    assert result["images_crop"].shape == (page_count * tiles_per_page, 3, 640, 640)
    assert (result["pixel_values"][:, :, 512, 512] == 1).all()


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("ocr", [False, True], ids=["vl2", "ocr"])
def test_http_rejects_image_preprocessing_over_budget(tmp_path, ocr):
    config = DeepseekVLV2Config(
        architectures=["DeepseekOCRForCausalLM" if ocr else "DeepseekVLV2ForCausalLM"],
        tile_tag="2D",
        candidate_resolutions=((16, 16),),
    ).to_dict()
    config["language_config"] = config.pop("text_config")
    (tmp_path / "config.json").write_text(json.dumps(config))
    if ocr:
        DeepseekOCRProcessor(tokenizer=_make_tokenizer()).save_pretrained(tmp_path)
        # Three global views and one view's scratch fit; four global views do not.
        limit, accepted_count, max_model_len = 64 * (1 << 20), 3, "2048"
    else:
        DeepseekVLV2Processor(
            tokenizer=_make_tokenizer(),
            candidate_resolutions=((16, 16),),
            patch_size=4,
            downsample_ratio=2,
        ).save_pretrained(tmp_path)
        limit, accepted_count, max_model_len = 27000, 1, "128"

    content = [
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(Image.new("RGB", (8, 8), color))},
        }
        for color in ("red", "green", "blue", "white")[: 4 if ocr else 3]
    ]
    model = str(tmp_path)
    with RemoteLaunchRenderServer(
        model,
        [
            "--host",
            "127.0.0.1",
            "--dtype",
            "float32",
            "--max-model-len",
            max_model_len,
            "--mm-processor-cache-gb",
            "0",
        ],
        env_dict={"VLLM_MAX_MEDIA_PREPROCESSING_BYTES": str(limit)},
    ) as server:
        for route in ("v1/chat/completions/render", "tokenize"):
            for images, expected_status in (
                (content, 400),
                (content[:accepted_count], 200),
            ):
                response = httpx.post(
                    server.url_for(route),
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": images}],
                    },
                )
                assert response.status_code == expected_status, response.text[:512]
                if expected_status == 400:
                    assert (
                        "Image preprocessing exceeds"
                        in response.json()["error"]["message"]
                    )


@pytest.fixture
def processor():
    config = ModelConfig(
        model="deepseek-ai/deepseek-vl2-tiny",
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    return MULTIMODAL_REGISTRY.create_processor(config)


def test_profiling_dummy_image_ignores_request_budget(tmp_path, monkeypatch):
    config = DeepseekVLV2Config(
        architectures=["DeepseekVLV2ForCausalLM"],
        tile_tag="2D",
        candidate_resolutions=((16, 16),),
    ).to_dict()
    config["language_config"] = config.pop("text_config")
    (tmp_path / "config.json").write_text(json.dumps(config))
    DeepseekVLV2Processor(
        tokenizer=_make_tokenizer(),
        candidate_resolutions=((16, 16),),
        patch_size=4,
        downsample_ratio=2,
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(model=str(tmp_path), max_model_len=128)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", 1)
    dummy = processor.get_dummy_mm_inputs({"image": 1})
    assert len(dummy["mm_placeholders"]["image"]) == 1


@pytest.mark.parametrize("use_uuid", [False, True])
def test_image_cache_identity_depends_on_cropping(processor, use_uuid: bool):
    tokenizer = processor.info.get_tokenizer()
    images = [
        Image.new("RGB", (768, 768), color)
        for color in ("red", "green", "blue", "yellow")
    ]
    results = []
    for count in (1, 2, 3, 4):
        inputs = ProcessorInputs(
            prompt=tokenizer.encode("<image>\n" * count + "Describe."),
            mm_data_items=processor.info.parse_mm_data({"image": images[:count]}),
            mm_uuid_items={"image": [f"image-{i}" for i in range(count)]}
            if use_uuid
            else None,
        )
        results.append(processor.apply(inputs, TimingContext(enabled=False)))

    hashes = [result["mm_hashes"]["image"][0] for result in results]
    lengths = [result["mm_placeholders"]["image"][0].length for result in results]
    assert lengths[0] == lengths[1] > lengths[2] == lengths[3]
    assert hashes[0] == hashes[1]
    assert hashes[2] == hashes[3]
    assert hashes[0] != hashes[2]


@pytest.mark.parametrize("use_uuid", [False, True])
def test_image_embedding_cache_identity_ignores_image_count(processor, use_uuid: bool):
    tokenizer = processor.info.get_tokenizer()
    embedding = torch.ones(8, processor.info.ctx.model_config.get_hidden_size())
    hashes = []
    for count in (1, 3):
        inputs = ProcessorInputs(
            prompt=tokenizer.encode("<image>\n" * count + "Describe."),
            mm_data_items=processor.info.parse_mm_data({"image": [embedding] * count}),
            mm_uuid_items={"image": [f"embedding-{i}" for i in range(count)]}
            if use_uuid
            else None,
        )
        result = processor.apply(inputs, TimingContext(enabled=False))
        assert result["mm_placeholders"]["image"][0].length == embedding.shape[0]
        hashes.append(result["mm_hashes"]["image"][0])

    assert hashes[0] == hashes[1]
    if use_uuid:
        assert hashes[0] == "uuid:embedding-0"


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("prompt_as_tokens", [False, True])
def test_request_candidate_resolutions_match_engine(prompt_as_tokens):
    """A request crop limit must change placeholders along with image features."""
    kwargs = {"candidate_resolutions": ((384, 384),)}
    image = Image.new("RGB", (1600, 1200))
    results = []
    for on_engine in (False, True):
        ctx = build_model_context(
            "deepseek-ai/deepseek-vl2-tiny",
            mm_processor_kwargs=kwargs if on_engine else None,
            mm_processor_cache_gb=0.01,
        )
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
        hf = processor.info.get_hf_processor(**kwargs)
        prompt = "<image>\nDescribe."
        expected = hf(text=prompt, images=[image])["num_image_tokens"]
        if prompt_as_tokens:
            prompt = ctx.get_tokenizer().encode(prompt)
        for _ in range(2):
            result = processor(
                prompt,
                mm_items=processor.info.parse_mm_data({"image": image}),
                hf_processor_mm_kwargs={} if on_engine else kwargs,
            )
            assert [p.length for p in result["mm_placeholders"]["image"]] == list(
                expected
            )
        results.append(result["prompt_token_ids"])
    assert results[0] == results[1]
