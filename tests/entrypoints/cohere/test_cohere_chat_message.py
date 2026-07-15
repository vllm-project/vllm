# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for :mod:`vllm.entrypoints.cohere.cohere_chat_message`.

The module hosts the citation-carrying subclasses of the OpenAI chat
completion protocol (``CohereChatMessage`` / ``CohereDeltaMessage``) plus
the shared ``Citation`` / ``CitationSource`` shapes. These tests pin:

1. Serialization behavior we own (empty-``citations`` cleanup on the
   subclass ``_serialize``, ``exclude_none`` shape, round-trip).
2. The constraint that ``type`` discriminators reject unknown values.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.cohere.cohere_chat_message import (
    Citation,
    CitationSource,
    CohereChatMessage,
    CohereDeltaMessage,
)

# ======================================================================
# CitationSource
# ======================================================================


class TestCitationSource:
    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            CitationSource(type="other")  # type: ignore[arg-type]

    def test_none_fields_excluded_from_dump(self):
        s = CitationSource(type="document", id="d1")
        assert s.model_dump(exclude_none=True) == {
            "type": "document",
            "id": "d1",
        }


# ======================================================================
# Citation
# ======================================================================


class TestCitation:
    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError):
            Citation(type="OTHER")  # type: ignore[arg-type]

    def test_dump_excludes_none_fields(self):
        c = Citation(start=0, end=5, text="hello")
        dumped = c.model_dump(exclude_none=True)
        assert dumped == {
            "start": 0,
            "end": 5,
            "text": "hello",
            "sources": [],
        }


# ======================================================================
# CohereDeltaMessage
# ======================================================================


class TestCohereDeltaMessage:
    def test_citations_omitted_from_dump_when_none(self):
        # Non-grounded deltas should never carry ``citations: null`` on
        # the wire even when routed through the Cohere subclass.
        d = CohereDeltaMessage(content="hello")
        dumped = d.model_dump(exclude_none=True)
        assert "citations" not in dumped

    def test_citations_dump_round_trip(self):
        d = CohereDeltaMessage(citations=[Citation(start=0, end=5, text="hi")])
        dumped = d.model_dump(exclude_none=True)
        assert dumped["citations"][0]["text"] == "hi"
        # Round-trip through the subclass to confirm the field validates.
        d2 = CohereDeltaMessage.model_validate(dumped)
        assert d2.citations[0].text == "hi"


# ======================================================================
# CohereChatMessage
# ======================================================================


class TestCohereChatMessage:
    def test_citations_omitted_from_dump_when_none(self):
        m = CohereChatMessage(role="assistant", content="hi")
        dumped = m.model_dump(exclude_none=True)
        assert "citations" not in dumped

    def test_citations_dump_round_trip(self):
        m = CohereChatMessage(
            role="assistant",
            content="hello",
            citations=[
                Citation(
                    start=0,
                    end=5,
                    text="hello",
                    sources=[CitationSource(type="document", id="d1")],
                )
            ],
        )
        dumped = m.model_dump(exclude_none=True)
        assert dumped["role"] == "assistant"
        assert dumped["content"] == "hello"
        assert dumped["citations"][0]["text"] == "hello"
        assert dumped["citations"][0]["sources"][0]["id"] == "d1"
        # Round-trip through the subclass to confirm the field validates.
        m2 = CohereChatMessage.model_validate(dumped)
        assert m2.citations[0].text == "hello"
        assert m2.citations[0].sources[0].id == "d1"
