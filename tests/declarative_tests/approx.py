# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from collections.abc import Mapping, Sized
from _pytest.python_api import ApproxMapping, ApproxSequenceLike

# This is a hacky extension of pytest's approx function, to get it to work with
# arbitrarily nested dict/list type objects. Any contained floats are compared with a tolerance.


def approx(expected, rel=5e-4, abs=1e-9, nan_ok=False):
    if isinstance(expected, Mapping):
        return ApproxNestedMapping(expected, rel, abs, nan_ok)
    if is_seq_like(expected):
        return ApproxNestedSequenceLike(expected, rel, abs, nan_ok)
    return pytest.approx(expected, rel, abs, nan_ok)


class ApproxNestedMapping(ApproxMapping):
    def _check_type(self):
        return

    def __repr__(self) -> str:
        return "approx({!r})".format(
            {k: approx(v) for k, v in self.expected.items()}
        )

    def _yield_comparisons(self, actual):
        if set(self.expected.keys()) != set(actual.keys()):
            return [(self.expected, actual)]
        return _yield_comparisons(self, super(), actual)

    def _repr_compare(self, other_side) -> list[str]:
        #TODO impl more useful explanation here
        try:
            return super()._repr_compare(other_side)
        except Exception:
            return ["Mismatch"]


class ApproxNestedSequenceLike(ApproxSequenceLike):
    def _check_type(self):
        return

    def __repr__(self) -> str:
        seq_type = type(self.expected)
        if seq_type not in (tuple, list):
            seq_type = list
        return "approx({!r})".format(
            seq_type(approx(x) for x in self.expected)
        )

    def _yield_comparisons(self, actual):
        if len(self.expected) != len(actual):
            return [(self.expected, actual)]
        return _yield_comparisons(self, super(), actual)

    def _repr_compare(self, other_side) -> list[str]:
        #TODO impl more useful explanation here
        try:
            return super()._repr_compare(other_side)
        except Exception:
            return ["Mismatch"]


def _yield_nested(actual, expected, **kwargs):
    if isinstance(expected, Mapping):
        return ApproxNestedMapping(expected, **kwargs)._yield_comparisons(actual)
    if is_seq_like(expected):
        return ApproxNestedSequenceLike(
            expected, **kwargs)._yield_comparisons(actual)
    return [(actual, expected)]


def _yield_comparisons(self, supr, actual):
    for actualItem, expected in supr._yield_comparisons(actual):
        yield from _yield_nested(
            actualItem, expected, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)


def is_seq_like(obj):
    return hasattr(obj, "__getitem__") and isinstance(obj, Sized) \
        and not isinstance(obj, (bytes, str))
