# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for converting a rendered `GenerateRequest` to `vllm.Render`."""

import pybase64
import pytest

pytest.importorskip("grpc")
render_grpc = pytest.importorskip(
    "vllm.entrypoints.scale_out.render.grpc.render_pb2",
    reason="vllm.Render stubs not generated (run compile_protos)",
)

import grpc  # noqa: E402

from vllm.entrypoints.scale_out.render.grpc import inference_pb2  # noqa: E402
from vllm.entrypoints.scale_out.render.grpc.servicer import (  # noqa: E402
    MAX_INLINE_KWARGS_BYTES,
    RenderRpcError,
    to_render_response,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (  # noqa: E402
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
    ReasoningParserKwargs,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams  # noqa: E402


def _rendered(token_ids=(1, 2, 3), **kwargs) -> GenerateRequest:
    kwargs.setdefault("sampling_params", SamplingParams())
    return GenerateRequest(token_ids=list(token_ids), **kwargs)


MM_TOKEN_IDS = range(1, 25)


def test_text_only_response():
    response = to_render_response(_rendered())

    assert list(response.prompt_token_ids) == [1, 2, 3]
    assert not response.HasField("structured_output")
    assert not response.HasField("engine_reasoning_gate")
    assert len(response.media) == 0


@pytest.mark.parametrize(
    ("params", "field", "expected"),
    [
        (
            StructuredOutputsParams(json={"type": "object"}),
            "json",
            '{"type": "object"}',
        ),
        (
            StructuredOutputsParams(json='{"type":"string"}'),
            "json",
            '{"type":"string"}',
        ),
        (StructuredOutputsParams(regex="[a-z]+"), "regex", "[a-z]+"),
        (StructuredOutputsParams(grammar="root ::= 'a'"), "grammar", "root ::= 'a'"),
        (StructuredOutputsParams(json_object=True), "json_object", True),
        (StructuredOutputsParams(structural_tag="{}"), "structural_tag", "{}"),
    ],
)
def test_structured_output(params, field, expected):
    response = to_render_response(
        _rendered(sampling_params=SamplingParams(structured_outputs=params))
    )

    assert response.structured_output.WhichOneof("kind") == field
    assert getattr(response.structured_output, field) == expected


def test_structured_output_choice():
    params = StructuredOutputsParams(choice=["yes", "no"])
    response = to_render_response(
        _rendered(sampling_params=SamplingParams(structured_outputs=params))
    )

    assert list(response.structured_output.choice.choices) == ["yes", "no"]


@pytest.mark.parametrize(
    ("reasoning_ended", "chat_template_kwargs", "has_gate"),
    [
        (None, {}, False),
        (True, {}, True),
        (None, {"enable_thinking": False}, True),
        (False, {"enable_thinking": True}, True),
    ],
)
def test_engine_reasoning_gate(reasoning_ended, chat_template_kwargs, has_gate):
    rendered = _rendered(
        reasoning_ended=reasoning_ended,
        reasoning_parser_kwargs=ReasoningParserKwargs(
            chat_template_kwargs=chat_template_kwargs
        ),
    )
    response = to_render_response(rendered)

    assert response.HasField("engine_reasoning_gate") == has_gate
    if not has_gate:
        return
    gate = response.engine_reasoning_gate
    assert gate.HasField("reasoning_ended") == (reasoning_ended is not None)
    if reasoning_ended is not None:
        assert gate.reasoning_ended == reasoning_ended
    assert dict(gate.chat_template_kwargs) == chat_template_kwargs


def _features(kwargs_data=None) -> MultiModalFeatures:
    return MultiModalFeatures(
        mm_hashes={"audio": ["a0"], "image": ["i0", "i1"]},
        mm_placeholders={
            "audio": [PlaceholderRangeInfo(offset=20, length=2)],
            "image": [
                PlaceholderRangeInfo(offset=1, length=3, is_embed=[True, False, True]),
                PlaceholderRangeInfo(offset=10, length=4),
            ],
        },
        kwargs_data=kwargs_data,
    )


def test_media_ordered_by_placeholder_without_kwargs():
    response = to_render_response(_rendered(MM_TOKEN_IDS, features=_features()))

    assert [
        (m.modality, m.features.identifier, m.features.offset) for m in response.media
    ] == [
        (inference_pb2.MODALITY_IMAGE, "i0", 1),
        (inference_pb2.MODALITY_IMAGE, "i1", 10),
        (inference_pb2.MODALITY_AUDIO, "a0", 20),
    ]
    first = response.media[0].features
    assert first.mm_hash == "i0"
    assert first.length == 3
    assert list(first.is_embed) == [True, False, True]
    assert all(not m.features.HasField("kwargs") for m in response.media)


def test_media_kwargs_are_raw_msgpack_bytes():
    def b64(data: bytes) -> str:
        return pybase64.b64encode(data).decode("ascii")

    kwargs_data = {"audio": [b64(b"audio")], "image": [b64(b"img0"), None]}
    response = to_render_response(
        _rendered(MM_TOKEN_IDS, features=_features(kwargs_data))
    )

    by_id = {m.features.identifier: m.features for m in response.media}
    assert by_id["a0"].kwargs == b"audio"
    assert by_id["i0"].kwargs == b"img0"
    assert not by_id["i1"].HasField("kwargs")


def test_media_kwargs_over_limit():
    big = pybase64.b64encode(b"x" * (MAX_INLINE_KWARGS_BYTES + 1)).decode("ascii")
    features = MultiModalFeatures(
        mm_hashes={"image": ["i0"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=1)]},
        kwargs_data={"image": [big]},
    )

    with pytest.raises(RenderRpcError) as exc_info:
        to_render_response(_rendered(features=features))
    assert exc_info.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
