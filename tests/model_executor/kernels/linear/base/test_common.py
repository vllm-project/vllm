# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.kernels.linear.base.common as common

SENTINEL_PRED = torch.tensor([1.0])
SENTINEL_TERMINAL = torch.tensor([2.0])
SENTINEL_NATIVE = torch.tensor([3.0])


class PredTrue(common.PredicateKernel):
    """Default predicate kernel: supported, predicate True, returns SENTINEL_PRED."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    @staticmethod
    def predicate(*args, **kwargs):
        return True

    @staticmethod
    def apply(*args, **kwargs):
        return SENTINEL_PRED

    def apply_weights(self, *args, **kwargs):
        return type(self).apply(*args, **kwargs)


class PredFalse(PredTrue):
    @staticmethod
    def predicate(*args, **kwargs):
        return False


class PredUnsupported(PredTrue):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return False, "unsupported"


class PredCannotImplement(PredTrue):
    @classmethod
    def can_implement(cls, config):
        return False, "cannot implement"


class _PlainBase(common.Kernel):
    """Default plain kernel: supported, apply returns SENTINEL_TERMINAL."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    @staticmethod
    def apply(*args, **kwargs):
        return SENTINEL_TERMINAL

    def apply_weights(self, *args, **kwargs):
        return type(self).apply(*args, **kwargs)


class PlainTerminal(_PlainBase):
    pass


class PlainTerminalUnsupported(_PlainBase):
    @classmethod
    def is_supported(cls, compute_capability=None):
        return False, "unsupported"


def _simple_dispatcher(predicate, primary, fallback_fn):
    """Minimal w16a16-style dispatcher used to compose chains in tests."""

    def dispatch(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if predicate(x, weight, bias):
            return primary.apply(x, weight, bias)
        return fallback_fn(x, weight, bias)

    return dispatch


def _native_impl(*args, **kwargs):
    return SENTINEL_NATIVE


_OP_COUNTER = 0


def _unique_tag(label: str) -> str:
    """Unique scheme_tag per call to avoid op-name collisions across tests."""
    global _OP_COUNTER
    _OP_COUNTER += 1
    return f"test_{label}_{_OP_COUNTER}"


def _composite(chain, *, scheme_tag, native_impl=_native_impl):
    return type(
        f"_Composite_{scheme_tag}",
        (common.Composite,),
        {
            "_scheme_tag": scheme_tag,
            "_chain": chain,
            "_dispatcher_fn": staticmethod(_simple_dispatcher),
            "_native_impl": staticmethod(native_impl),
        },
    )


def _config():
    return SimpleNamespace()


class TestPredicateKernel:
    def test_subclass_without_predicate_cannot_be_instantiated(self):
        class _Missing(common.PredicateKernel):
            @classmethod
            def is_supported(cls, compute_capability=None):
                return True, None

            def apply_weights(self, *a, **kw):
                return type(self).apply(*a, **kw)

            @staticmethod
            def apply(*a, **kw):
                return torch.tensor([0.0])

        with pytest.raises(TypeError, match="abstract"):
            _Missing(config=None)


class TestCompositeChainValidation:
    def test_plain_kernel_in_non_terminal_position_raises(self):
        Comp = _composite(
            [PredTrue, PlainTerminal, PredTrue],
            scheme_tag=_unique_tag("plain_in_middle"),
        )
        with pytest.raises(TypeError, match="must be a PredicateKernel"):
            Comp(_config())

    def test_terminal_can_be_plain_kernel(self):
        Comp = _composite(
            [PredTrue, PlainTerminal], scheme_tag=_unique_tag("plain_term")
        )
        Comp(_config())  # no error

    def test_terminal_can_be_predicate_kernel(self):
        Comp = _composite([PredTrue, PredFalse], scheme_tag=_unique_tag("pred_term"))
        Comp(_config())  # no error


class TestCompositeSelectorGating:
    def test_is_supported_true_when_any_inner_supported(self):
        Comp = _composite(
            [PredUnsupported, PredTrue],
            scheme_tag=_unique_tag("is_sup_any"),
        )
        assert Comp.is_supported()[0] is True

    def test_is_supported_false_when_all_inner_unsupported(self):
        """Selector skips the composite when nothing in the chain is supported."""
        Comp = _composite(
            [PredUnsupported, PlainTerminalUnsupported],
            scheme_tag=_unique_tag("is_sup_none"),
        )
        ok, reason = Comp.is_supported()
        assert ok is False
        assert "no inner kernel supported" in reason

    def test_can_implement_true_when_any_inner_viable(self):
        Comp = _composite(
            [PredTrue, PredCannotImplement],
            scheme_tag=_unique_tag("can_impl_any"),
        )
        assert Comp.can_implement(config=None)[0] is True

    def test_can_implement_false_when_all_inner_unviable(self):
        """All inner kernels either unsupported or cannot_implement. Composite
        signals can_implement=False so the selector falls through to the next
        candidate in _POSSIBLE_*_KERNELS."""
        Comp = _composite(
            [PredUnsupported, PredCannotImplement],
            scheme_tag=_unique_tag("can_impl_none"),
        )
        ok, reason = Comp.can_implement(config=None)
        assert ok is False
        assert "no inner kernel viable" in reason


class TestCompositeDispatch:
    def test_matching_predicate_runs_over_terminal(self):
        Comp = _composite(
            [PredTrue, PlainTerminal],
            scheme_tag=_unique_tag("matches_over_terminal"),
        )
        inst = Comp(_config())
        assert inst._dispatch_fn(torch.zeros(1), torch.zeros(1), None) is SENTINEL_PRED

    def test_falls_through_to_plain_terminal(self):
        Comp = _composite(
            [PredFalse, PredFalse, PlainTerminal],
            scheme_tag=_unique_tag("falls_to_terminal"),
        )
        inst = Comp(_config())
        assert (
            inst._dispatch_fn(torch.zeros(1), torch.zeros(1), None) is SENTINEL_TERMINAL
        )

    def test_falls_through_to_native_when_no_terminal(self):
        """All predicates False, no plain Kernel terminal is native_impl."""
        Comp = _composite(
            [PredFalse, PredFalse],
            scheme_tag=_unique_tag("falls_native_no_term"),
        )
        inst = Comp(_config())
        assert (
            inst._dispatch_fn(torch.zeros(1), torch.zeros(1), None) is SENTINEL_NATIVE
        )

    def test_falls_through_to_native_when_terminal_unviable(self):
        """All predicates False, plain terminal unsupported is native_impl."""
        Comp = _composite(
            [PredFalse, PredFalse, PlainTerminalUnsupported],
            scheme_tag=_unique_tag("falls_native_unviable"),
        )
        inst = Comp(_config())
        assert (
            inst._dispatch_fn(torch.zeros(1), torch.zeros(1), None) is SENTINEL_NATIVE
        )

    def test_unsupported_predicates_filtered_at_init(self):
        """A PredicateKernel whose is_supported returns False is excluded at
        init time, so its (matching) predicate never gets exercised."""
        Comp = _composite(
            [PredUnsupported, PredTrue, PlainTerminal],
            scheme_tag=_unique_tag("filter_unsupported"),
        )
        inst = Comp(_config())
        assert inst._dispatch_fn(torch.zeros(1), torch.zeros(1), None) is SENTINEL_PRED


class TestCompositeOpRegistration:
    def test_op_name_format(self):
        """Op name is `{scheme_tag}_{8 hex chars}`."""
        tag = _unique_tag("name_format")
        Comp = _composite([PredTrue, PlainTerminal], scheme_tag=tag)
        op_name = Comp(_config())._op_name
        assert op_name.startswith(f"{tag}_")
        suffix = op_name[len(tag) + 1 :]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_distinct_chains_register_distinct_ops(self):
        """Different chain composition under the same scheme_tag hashes to a
        different op — the case that used to collide under empty prefix."""
        tag = _unique_tag("distinct_chains")
        CompA = _composite([PredTrue, PlainTerminal], scheme_tag=tag)
        CompB = _composite([PredFalse, PredTrue, PlainTerminal], scheme_tag=tag)
        assert CompA(_config())._op_name != CompB(_config())._op_name

    def test_register_skipped_when_op_already_exists(self):
        """Re-instantiating the same Composite must not re-register —
        torch.library rejects duplicate defines, so the second Comp(...) here
        would raise if the hasattr guard in Composite.__init__ were removed."""
        tag = _unique_tag("registration_skipped")
        Comp = _composite([PredTrue, PlainTerminal], scheme_tag=tag)

        first = Comp(_config())
        before = len(vars(torch.ops.composed_kernel))
        second = Comp(_config())
        after = len(vars(torch.ops.composed_kernel))

        assert first._op_name == second._op_name
        assert hasattr(torch.ops.composed_kernel, first._op_name)
        assert after == before
