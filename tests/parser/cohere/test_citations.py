# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Citation source resolution and the streamed citation wire shape."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vllm.parser.cohere_command import (
    _melody_citations_to_vllm,
    _melody_sources_to_vllm,
)

from .utils import make_parser, stream_parser


class TestMelodySourceResolution:
    """Pin the parser-side source resolution (see
    ``_melody_sources_to_vllm``). The parser receives melody's numeric
    ``(tool_call_index, tool_result_indices)`` addressing and resolves
    it against a ``position_to_source`` map handed in by
    ``CohereServingChatV2._apply_cohere_template_kwargs`` -- the same
    map that would otherwise live in the serving layer's resolver.
    """

    @staticmethod
    def _fake_melody_source(bucket: int, indices: list[int]) -> Any:
        return SimpleNamespace(tool_call_index=bucket, tool_result_indices=indices)

    def test_multi_index_source_fans_out(self):
        from vllm.entrypoints.cohere.cohere_chat_message import CitationSource

        position_map: dict[tuple[int, int], CitationSource] = {
            (0, 0): CitationSource(type="document", id="d0", document={"id": "d0"}),
            (0, 1): CitationSource(type="document", id="d1", document={"id": "d1"}),
        }
        raw = [self._fake_melody_source(0, [0, 1])]
        out = _melody_sources_to_vllm(raw, position_map)
        assert [s.id for s in out] == ["d0", "d1"]
        # Verify type / payload were plumbed through, not just the id.
        assert out[0].type == "document"
        assert out[0].document == {"id": "d0"}

    def test_unresolvable_position_skipped(self):
        from vllm.entrypoints.cohere.cohere_chat_message import CitationSource

        position_map: dict[tuple[int, int], CitationSource] = {
            (0, 0): CitationSource(type="document", id="d0"),
        }
        raw = [self._fake_melody_source(9, [0])]
        assert _melody_sources_to_vllm(raw, position_map) == []

    def test_missing_position_map_drops_all_sources(self):
        # A parser instance without a position map (parser wired
        # outside of ``CohereServingChatV2``) can't attribute anything,
        # so every source is dropped. Callers downstream will see the
        # citation with empty ``sources`` and drop it entirely.
        raw = [self._fake_melody_source(0, [0])]
        assert _melody_sources_to_vllm(raw, None) == []

    def test_citations_pass_through_is_thinking_tag(self):
        from vllm.entrypoints.cohere.cohere_chat_message import CitationSource

        position_map: dict[tuple[int, int], CitationSource] = {
            (0, 0): CitationSource(type="document", id="d0"),
        }
        raw = [
            SimpleNamespace(
                start_index=0,
                end_index=5,
                text="hello",
                is_thinking=True,
                sources=[self._fake_melody_source(0, [0])],
            )
        ]
        out = _melody_citations_to_vllm(raw, position_map)
        assert out is not None
        assert out[0].type == "THINKING_CONTENT"
        assert out[0].sources[0].id == "d0"


class TestParserStreamingEndToEnd:
    """End-to-end streaming shape: raw model output text is tokenized
    with the mock byte-level tokenizer and fed one token at a time
    through :meth:`CohereCommand3ReasoningParser.extract_reasoning_streaming`.
    We collect the ordered sequence of ``DeltaMessage``s the parser
    hands back -- i.e. what the client would receive on the wire --
    and pin the exact shape.

    Combines the three flavors that reach ``delta``: a thinking block
    (populates ``reasoning``), a text block (populates ``content``),
    and a resolved citation (populates ``citations``) referencing
    ``(start, end)`` offsets into the already-streamed content.
    """

    _MODEL_OUTPUT = (
        "<|START_THINKING|>Let me check.<|END_THINKING|>"
        "<|START_RESPONSE|>The capital is <co>Paris</co: 0:[1]>.<|END_RESPONSE|>"
    )

    def test_full_wire_stream_shape(self, tokenizer, request_obj):
        from vllm.entrypoints.cohere.cohere_chat_message import CitationSource
        from vllm.renderers.cohere import POSITION_TO_SOURCE_KEY

        # Mimic what ``CohereServingChatV2._apply_cohere_template_kwargs``
        # installs on a request whose tool_call_index=0 tool result at
        # index 1 is the "France" document.
        position_to_source = {
            (0, 1): CitationSource(
                type="document",
                id="doc-paris",
                document={"id": "doc-paris", "title": "France"},
            ),
        }

        parser = make_parser(
            tokenizer,
            "cohere_command3",
            chat_template_kwargs={POSITION_TO_SOURCE_KEY: position_to_source},
        )
        deltas = stream_parser(parser, request_obj, tokenizer, self._MODEL_OUTPUT)

        # -- 1. Reasoning and content stream separately and reach
        # the wire only as their real payload. Framing tokens
        # (``<|START_THINKING|>`` etc.) are consumed by the parser's
        # state machine and never produce a delta.
        reasoning_stream = [d.reasoning for d in deltas if d.reasoning is not None]
        content_stream = [d.content for d in deltas if d.content is not None]
        citation_deltas = [d for d in deltas if getattr(d, "citations", None)]

        assert "".join(reasoning_stream) == "Let me check."
        assert "".join(content_stream) == "The capital is Paris."

        # -- 2. Exactly one citation reaches the wire, resolved to the
        # source from ``position_to_source`` (not the raw
        # ``(bucket, idx)`` coordinates).
        assert len(citation_deltas) == 1
        (cite_delta,) = citation_deltas
        assert cite_delta.citations is not None
        assert len(cite_delta.citations) == 1
        cite = cite_delta.citations[0]
        assert cite.text == "Paris"
        # Char offsets are into the accumulated ``content``: "The
        # capital is " is 15 chars, "Paris" is 5.
        assert cite.start == 15
        assert cite.end == 20
        assert len(cite.sources) == 1
        assert cite.sources[0].id == "doc-paris"
        assert cite.sources[0].type == "document"

        # -- 3. The citation's ``(start, end)`` offsets point into
        # bytes that were already emitted as content-deltas earlier
        # in the stream -- i.e. the citation is anchored to real
        # already-shipped content.
        cite_delta_idx = deltas.index(cite_delta)
        earlier_content = "".join(
            d.content for d in deltas[:cite_delta_idx] if d.content is not None
        )
        assert earlier_content[cite.start : cite.end] == "Paris"

        # Everything after the citation delta is the trailing period.
        later_content = "".join(
            d.content for d in deltas[cite_delta_idx + 1 :] if d.content is not None
        )
        assert later_content == "."

        # -- 4. Reasoning and content are never mixed on the same
        # delta: the parser flips modes on the ``<|END_THINKING|>``
        # boundary and never emits a delta with both fields set.
        for d in deltas:
            assert not (d.reasoning is not None and d.content is not None)
