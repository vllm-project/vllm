import pytest

from .utils import build_model_context


@pytest.mark.parametrize("limit,is_multimodal", [(0, False), (1, True)])
def test_is_multimodal_model_respects_limit(limit, is_multimodal):
    ctx = build_model_context(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        limit_mm_per_prompt={"image": limit},
    )
    assert ctx.model_config.is_multimodal_model is is_multimodal
