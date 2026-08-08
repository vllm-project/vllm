# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``store`` field semantics on ResponsesRequest.

``store`` is documented as defaulting to ``true``; an explicit
``store: null`` means "unspecified" and must resolve to that default, the
same as omitting the field. Previously an explicit ``null`` fell through
every ``if request.store:`` gate in the serving layer (the response was
silently never stored, so later retrieval by id 404'd) and incorrectly
tripped the background/store conflict validator.
"""

import pytest

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _make_request(**kwargs) -> ResponsesRequest:
    return ResponsesRequest(model="test-model", input="test input", **kwargs)


@pytest.mark.cpu_test
def test_store_omitted_defaults_to_true():
    assert _make_request().store is True


@pytest.mark.cpu_test
def test_store_explicit_true():
    assert _make_request(store=True).store is True


@pytest.mark.cpu_test
def test_store_explicit_false():
    assert _make_request(store=False).store is False


@pytest.mark.cpu_test
def test_store_explicit_null_resolves_to_default():
    """Regression: explicit ``store: null`` was kept as ``None`` and treated
    as falsy by every ``if request.store:`` consumer."""
    assert _make_request(store=None).store is True


@pytest.mark.cpu_test
def test_background_with_store_null_is_allowed():
    """Regression: ``{"background": true, "store": null}`` incorrectly raised
    "background can only be used when `store` is true" because the validator
    treated ``None`` as falsy rather than resolving it to the default."""
    request = _make_request(background=True, store=None)
    assert request.background is True
    assert request.store is True


@pytest.mark.cpu_test
def test_background_with_store_false_still_rejected():
    with pytest.raises(ValueError, match="background can only be used"):
        _make_request(background=True, store=False)
