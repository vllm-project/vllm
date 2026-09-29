# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the multi-model render dispatcher.

Exercises the pure-dispatch behaviour of `MultiModelOnlineRenderer` /
`MultiModelOnlineDerenderer` without loading a real vLLM engine — the goal
is to lock in the routing contract (primary as default, exact-match lookup,
unknown-name fallback) before wiring up end-to-end integration tests that
require actual tokenizers and processors.
"""

from unittest.mock import MagicMock

import pytest

from vllm.renderers.multi_model import (
    MultiModelOnlineDerenderer,
    MultiModelOnlineRenderer,
)
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer


def _fake(kind: type, name: str) -> MagicMock:
    fake = MagicMock(spec=kind)
    fake._label = name  # noqa: SLF001
    return fake


@pytest.mark.parametrize(
    "wrapper_cls, member_cls, attr",
    [
        (MultiModelOnlineRenderer, OnlineRenderer, "renderers"),
        (MultiModelOnlineDerenderer, OnlineDerenderer, "derenderers"),
    ],
)
def test_resolve_matches_registered_names_and_falls_back_to_primary(
    wrapper_cls, member_cls, attr
) -> None:
    primary = _fake(member_cls, "primary")
    extra_a = _fake(member_cls, "extra_a")
    extra_b = _fake(member_cls, "extra_b")
    members = {"primary": primary, "extra_a": extra_a, "extra_b": extra_b}

    dispatcher = wrapper_cls("primary", members)

    assert getattr(dispatcher, attr) is members
    assert dispatcher.primary is primary
    assert dispatcher.resolve("primary") is primary
    assert dispatcher.resolve("extra_a") is extra_a
    assert dispatcher.resolve("extra_b") is extra_b
    # Unknown or missing model name falls back to the primary so legacy
    # single-model code paths keep working when they pass `None`.
    assert dispatcher.resolve(None) is primary
    assert dispatcher.resolve("not-registered") is primary


def test_primary_must_be_a_registered_member() -> None:
    other = _fake(OnlineRenderer, "other")
    with pytest.raises(AssertionError, match="must be present in renderers"):
        MultiModelOnlineRenderer("missing", {"other": other})


def test_warmup_visits_every_registered_renderer() -> None:
    primary = _fake(OnlineRenderer, "primary")
    extra = _fake(OnlineRenderer, "extra")
    MultiModelOnlineRenderer("primary", {"primary": primary, "extra": extra}).warmup()
    primary.warmup.assert_called_once_with()
    extra.warmup.assert_called_once_with()
