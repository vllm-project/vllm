import pytest

from vllm.utils import deprecate_kwargs

from .utils import error_on_warning


def test_deprecate_kwargs_always():

    @deprecate_kwargs("old_arg", is_deprecated=True)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)


def test_deprecate_kwargs_never():

    @deprecate_kwargs("old_arg", is_deprecated=False)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with error_on_warning():
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)


def test_deprecate_kwargs_func():
    is_deprecated = True

    @deprecate_kwargs("old_arg", is_deprecated=lambda: is_deprecated)
    def dummy(*, old_arg: object = None, new_arg: object = None):
        pass

    with pytest.warns(DeprecationWarning, match="'old_arg'"):
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)

    is_deprecated = False

    with error_on_warning():
        dummy(old_arg=1)

    with error_on_warning():
        dummy(new_arg=1)
