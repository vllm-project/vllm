# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from tests.utils import VLLM_PATH
from vllm import LLM, PoolingParams, ScoringRequestOutput
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.utils import encode_image_url, fetch_image
from vllm.platforms import current_platform

MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"
HF_OVERRIDES = {
    "architectures": ["Qwen3VLForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}
TEMPLATE = VLLM_PATH / "examples/pooling/score/template/qwen3_vl_reranker.jinja"

ROCM_ATTN_BACKENDS = [
    "ROCM_ATTN",
    "ROCM_AITER_FA",
    "TRITON_ATTN",
    "FLEX_ATTENTION",
]

ATTN_BACKENDS = ROCM_ATTN_BACKENDS if current_platform.is_rocm() else ["auto"]

# The absolute floor matters only for the low ~0.10 text-vs-text score, where a
# few thousandths of drift would otherwise exceed the relative bound.
REL_TOL = 0.05
ABS_TOL = 0.007

query = "A cat standing in the snow."
document = "This product was excellent and exceeded my expectations."
image_url = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "multimodal_asset/cat_snow.jpg"
)
documents = [
    {
        "type": "text",
        "text": document,
    },
    {
        "type": "image_url",
        "image_url": {"url": image_url},
    },
    {
        "type": "image_url",
        "image_url": {"url": encode_image_url(fetch_image(image_url))},
    },
]

INSTRUCTION = (
    "Given a multimodal retrieval query, retrieve candidates that "
    "visually or textually match the requested scene, object, or action."
)

# One bf16 ULP in attention compounds to ~0.1% here, so ROCm backends each
# settle differently; these are their midpoint across backends and archs.
if current_platform.is_rocm():
    TEXT_VS_TEXT = 0.105243
    TEXT_VS_IMAGE = 0.742472
    TEXT_VS_TEXT_PLUS_IMAGE = 0.544568
else:
    TEXT_VS_TEXT = 0.10040374100208282
    TEXT_VS_IMAGE = 0.7423753142356873
    TEXT_VS_TEXT_PLUS_IMAGE = 0.5298863053321838

_LLM_PARAMS = [pytest.param(backend, id=backend) for backend in ATTN_BACKENDS]


def assert_score(actual: float, expected: float, backend: str, label: str):
    tol = REL_TOL
    abs_tol = ABS_TOL
    diff = abs(actual - expected)
    rel_diff = diff / abs(expected) if expected != 0 else diff
    print(
        f"[{backend}] {label}: actual={actual:.6f} expected={expected:.6f} "
        f"diff={diff:.6f} rel_diff={rel_diff:.4f} tol={tol} abs_tol={abs_tol}"
    )
    assert actual == pytest.approx(expected, rel=tol, abs=abs_tol), (
        f"[{backend}] {label}: score mismatch — "
        f"actual={actual:.6f}, expected={expected:.6f}, "
        f"rel_diff={rel_diff:.4f}, tol={tol}, abs_tol={abs_tol}"
    )


def _extract_scores(outputs: list[ScoringRequestOutput]) -> list[float]:
    scores: list[float] = []
    for out in outputs:
        assert out.finished
        assert out.outputs.score is not None
        scores.append(float(out.outputs.score))
    return scores


@pytest.fixture(scope="module")
def llm(request, vllm_runner):
    backend = request.param
    print(f"\n=== Starting offline LLM with attention backend: {backend} ===")
    mp = pytest.MonkeyPatch()
    if backend != "auto" and backend != "ROCM_AITER_FA":
        mp.setenv("VLLM_ROCM_USE_AITER", "0")

    try:
        with vllm_runner(
            MODEL_NAME,
            runner="pooling",
            max_model_len=8192,
            enforce_eager=True,
            hf_overrides=HF_OVERRIDES,
            chat_template=TEMPLATE,
        ) as runner:
            print(f"=== Offline LLM ready with backend: {backend} ===")
            # pytest caches yielded fixtures until after teardown, so use a proxy to
            # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
            yield weakref.proxy(runner.llm), backend
    finally:
        mp.undo()


@pytest.mark.parametrize("llm", _LLM_PARAMS, indirect=True)
@pytest.mark.skip_global_cleanup
def test_offline_score_text_vs_text(llm: tuple[LLM, str]):
    engine, backend = llm
    outputs = engine.score(
        query,
        document,
        use_tqdm=False,
    )
    scores = _extract_scores(outputs)
    assert len(scores) == 1
    assert_score(scores[0], TEXT_VS_TEXT, backend, "text_vs_text")


@pytest.mark.parametrize("llm", _LLM_PARAMS, indirect=True)
@pytest.mark.skip_global_cleanup
def test_offline_score_text_content_and_image_url(llm: tuple[LLM, str]):
    engine, backend = llm
    outputs = engine.score(
        query,
        [
            {"content": [documents[0]]},
            {"content": [documents[1]]},
        ],
        use_tqdm=False,
    )
    scores = _extract_scores(outputs)
    assert len(scores) == 2
    assert_score(scores[0], TEXT_VS_TEXT, backend, "list[0]_text_vs_text")
    assert_score(scores[1], TEXT_VS_IMAGE, backend, "list[1]_text_vs_image")


@pytest.mark.parametrize("llm", _LLM_PARAMS, indirect=True)
@pytest.mark.skip_global_cleanup
def test_offline_score_image_base64(llm: tuple[LLM, str]):
    engine, backend = llm
    outputs = engine.score(
        query,
        {"content": [documents[2]]},
        use_tqdm=False,
    )
    scores = _extract_scores(outputs)
    assert len(scores) == 1
    assert_score(scores[0], TEXT_VS_IMAGE, backend, "text_vs_image_base64")


@pytest.mark.parametrize("llm", _LLM_PARAMS, indirect=True)
@pytest.mark.skip_global_cleanup
def test_offline_score_image_url_plus_text_content(llm: tuple[LLM, str]):
    engine, backend = llm
    outputs = engine.score(
        query,
        {"content": [documents[0], documents[1]]},
        use_tqdm=False,
    )
    scores = _extract_scores(outputs)
    assert len(scores) == 1
    assert_score(scores[0], TEXT_VS_TEXT_PLUS_IMAGE, backend, "text_vs_text_plus_image")


@pytest.mark.parametrize("llm", _LLM_PARAMS, indirect=True)
@pytest.mark.skip_global_cleanup
def test_offline_score_queries_list_documents_list(llm: tuple[LLM, str]):
    engine, backend = llm
    outputs = engine.score(
        [query] * 4,
        [
            document,
            {"content": [documents[0]]},
            {"content": [documents[1]]},
            {"content": [documents[0], documents[1]]},
        ],
        use_tqdm=False,
    )
    scores = _extract_scores(outputs)
    assert len(scores) == 4
    assert_score(scores[0], TEXT_VS_TEXT, backend, "paired[0]_text_vs_text")
    assert_score(scores[1], TEXT_VS_TEXT, backend, "paired[1]_text_vs_text")
    assert_score(scores[2], TEXT_VS_IMAGE, backend, "paired[2]_text_vs_image")
    assert_score(
        scores[3],
        TEXT_VS_TEXT_PLUS_IMAGE,
        backend,
        "paired[3]_text_vs_text_plus_image",
    )


@pytest.mark.parametrize("llm", _LLM_PARAMS, indirect=True)
@pytest.mark.skip_global_cleanup
def test_offline_score_instruction(llm: tuple[LLM, str]):
    engine, _ = llm
    default_scores = _extract_scores(engine.score(query, document, use_tqdm=False))
    instruction_scores = _extract_scores(
        engine.score(
            query,
            document,
            pooling_params=PoolingParams(
                extra_kwargs={"chat_template_kwargs": {"instruction": INSTRUCTION}}
            ),
            use_tqdm=False,
        )
    )
    assert len(default_scores) == 1
    assert len(instruction_scores) == 1
    assert instruction_scores[0] != pytest.approx(default_scores[0], rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("llm", _LLM_PARAMS, indirect=True)
@pytest.mark.skip_global_cleanup
def test_offline_score_mm_and_cache_kwargs(llm: tuple[LLM, str]):
    engine, backend = llm
    outputs = engine.score(
        query,
        {"content": [documents[1]]},
        media_io_kwargs={"image": {"timeout": 10}},
        mm_processor_kwargs={},
        cache_salt="test-salt-value",
        use_tqdm=False,
    )
    scores = _extract_scores(outputs)
    assert len(scores) == 1
    assert_score(scores[0], TEXT_VS_IMAGE, backend, "mm_and_cache_kwargs")


@pytest.mark.parametrize("llm", _LLM_PARAMS, indirect=True)
@pytest.mark.skip_global_cleanup
def test_offline_score_invalid_cache_salt_rejected(llm: tuple[LLM, str]):
    engine, _ = llm
    with pytest.raises(VLLMValidationError):
        engine.score(
            query,
            {"content": [documents[1]]},
            cache_salt="",
            use_tqdm=False,
        )
