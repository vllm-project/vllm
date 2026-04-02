# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import transformers
from packaging.version import Version

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context

pytestmark = pytest.mark.skip_global_cleanup

GEMMA4_MIN_TRANSFORMERS_VERSION = Version("5.5.0")
GEMMA4_REQUIRED_SYMBOLS = (
    "Gemma4Config",
    "Gemma4ForConditionalGeneration",
    "Gemma4Processor",
)
GEMMA4_MODELS = [
    (
        "google/gemma-4-31B-it",
        "TransformersMultiModalForCausalLM",
    ),
    (
        "google/gemma-4-26B-A4B-it",
        "TransformersMultiModalMoEForCausalLM",
    ),
]


def _check_gemma4_transformers_version() -> None:
    installed = Version(Version(transformers.__version__).base_version)
    if installed < GEMMA4_MIN_TRANSFORMERS_VERSION:
        pytest.skip(
            "Gemma 4 requires transformers "
            f">={GEMMA4_MIN_TRANSFORMERS_VERSION}, got {installed}"
        )

    missing_symbols = [
        name for name in GEMMA4_REQUIRED_SYMBOLS if not hasattr(transformers, name)
    ]
    if missing_symbols:
        pytest.skip(
            "Installed transformers build does not expose Gemma 4 classes: "
            + ", ".join(missing_symbols)
        )


@pytest.mark.parametrize(("model_id", "expected_arch"), GEMMA4_MODELS)
def test_gemma4_uses_transformers_backend(
    model_id: str,
    expected_arch: str,
) -> None:
    _check_gemma4_transformers_version()

    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )

    assert ctx.model_config.architecture == expected_arch
    assert ctx.model_config.using_transformers_backend()


@pytest.mark.parametrize("model_id", [model_id for model_id, _ in GEMMA4_MODELS])
def test_gemma4_processor_emits_image_placeholders(
    image_assets: ImageTestAssets,
    model_id: str,
) -> None:
    _check_gemma4_transformers_version()

    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()

    processed_inputs = processor(
        hf_processor.image_token,
        mm_items=processor.info.parse_mm_data({"image": [image_assets[0].pil_image]}),
        hf_processor_mm_kwargs={},
    )

    image_token_id = processor.info.get_tokenizer().convert_tokens_to_ids(
        hf_processor.image_token
    )
    image_token_count = processed_inputs["prompt_token_ids"].count(image_token_id)

    assert image_token_count > 0
    assert "image" in processed_inputs["mm_placeholders"]
    assert processed_inputs["mm_placeholders"]["image"][0].length == image_token_count
