# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa

import pytest

from vllm.utils.func_utils import deprecate_kwargs, supports_kw

from ..utils import error_on_warning


def test_deprecate_kwargs_always():
    @deprecate_kwargs("old_arg", is_deprecated=True)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


def test_deprecate_kwargs_never():
    @deprecate_kwargs("old_arg", is_deprecated=False)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with error_on_warning(DeprecationWarning):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


def test_deprecate_kwargs_dynamic():
    is_deprecated = True

    @deprecate_kwargs("old_arg", is_deprecated=lambda: is_deprecated)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)

    is_deprecated = False

    with error_on_warning(DeprecationWarning):
        dummy(old_arg=1)

    with error_on_warning(DeprecationWarning):
        dummy(new_arg=1)


def test_deprecate_kwargs_additional_message():
    @deprecate_kwargs("old_arg", is_deprecated=True, additional_message="abcd")
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="abcd"):
        dummy(old_arg=1)


@pytest.mark.parametrize(
    ("callable", "kw_name", "requires_kw_only", "allow_var_kwargs", "is_supported"),
    [
        # Tests for positional argument support
        (lambda foo: None, "foo", True, True, False),
        (lambda foo: None, "foo", False, True, True),
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
