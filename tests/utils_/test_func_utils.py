# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa

import pytest

from vllm.utils.func_utils import get_allowed_kwarg_only_overrides, supports_kw


@pytest.mark.parametrize(
    ("callable", "kw_name", "requires_kw_only", "allow_var_kwargs", "is_supported"),
    [
        # Tests for positional argument support
        (lambda foo: None, "foo", True, True, False),
        (lambda foo: None, "foo", False, True, True),
        # Positional-only names can be independent keys in **kwargs
        (lambda foo, /, **kwargs: None, "foo", True, True, True),
        (lambda foo, /, **kwargs: None, "foo", True, False, False),
        (lambda foo, /: None, "foo", True, True, False),
        (lambda foo, /, **kwargs: None, "foo", False, True, True),
        (lambda foo, /, **kwargs: None, "foo", False, False, True),
        # Tests for positional or keyword / keyword only
        (lambda foo=100: None, "foo", True, True, False),
        (lambda *, foo: None, "foo", False, True, True),
        # Tests to make sure the names of variadic params are NOT supported
        (lambda *args: None, "args", False, True, False),
        (lambda **kwargs: None, "kwargs", False, True, False),
        # Tests for if we allow var kwargs to add support
        (lambda foo: None, "something_else", False, True, False),
        (lambda foo, **kwargs: None, "something_else", False, True, True),
        (lambda foo, **kwargs: None, "kwargs", True, True, False),
        (lambda foo, **kwargs: None, "foo", True, True, False),
    ],
)
def test_supports_kw(
    callable, kw_name, requires_kw_only, allow_var_kwargs, is_supported
):
    assert (
        supports_kw(
            callable=callable,
            kw_name=kw_name,
            requires_kw_only=requires_kw_only,
            allow_var_kwargs=allow_var_kwargs,
        )
        == is_supported
    )


def test_keyword_only_override_can_match_positional_only_name():
    def processor(value, /, **kwargs):
        return value, kwargs

    overrides = get_allowed_kwarg_only_overrides(
        processor, {"value": "override"}, allow_var_kwargs=True
    )
    assert processor("positional", **overrides) == ("positional", {"value": "override"})
