# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

import vllm.model_executor.kernels.linear.base.common as common
from vllm.platforms import current_platform

class _AlwaysSupported(common.Kernel):
    """Kernel stub that always reports itself as supported."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    def apply_weights(self, *args, **kwargs):
        return type(self).apply(*args, **kwargs)

    @staticmethod
    def apply(*args, **kwargs):
        return torch.tensor([1.0])


class _NeverSupported(common.Kernel):
    """Kernel that always reports itself as unsupported."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        return False, "not supported"

    def apply_weights(self, *args, **kwargs):
        return type(self).apply(*args, **kwargs)

    @staticmethod
    def apply(*args, **kwargs):
        return torch.tensor([0.0])


class _CannotImplement(common.Kernel):
    """Kernel that is supported but cannot implement the given config."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        return True, None

    @classmethod
    def can_implement(cls, config):
        return False, "cannot implement"

    def apply_weights(self, *args, **kwargs):
        return type(self).apply(*args, **kwargs)

    @staticmethod
    def apply(*args, **kwargs):
        return torch.tensor([0.0])



class TestResolveDispatchFn:

    def test_supported_kernel_returns_apply(self):
        fn = common._resolve_dispatch_fn(_AlwaysSupported, config=None)
        assert fn is _AlwaysSupported.apply

    def test_unsupported_kernel_no_fallback_returns_apply(self):
        fn = common._resolve_dispatch_fn(_NeverSupported, config=None)
        assert fn is _NeverSupported.apply

    def test_unsupported_kernel_falls_back(self):
        _NeverSupported._fallback = _AlwaysSupported
        try:
            fn = common._resolve_dispatch_fn(_NeverSupported, config=None)
            assert fn is _AlwaysSupported.apply
        finally:
            del _NeverSupported._fallback

    def test_cannot_implement_falls_back(self):
        _CannotImplement._fallback = _AlwaysSupported
        try:
            fn = common._resolve_dispatch_fn(_CannotImplement, config=None)
            assert fn is _AlwaysSupported.apply
        finally:
            del _CannotImplement._fallback

    def test_multi_level_fallback_chain(self):
        """Unsupported -> cannot-implement -> supported: resolves to leaf."""
        _NeverSupported._fallback = _CannotImplement
        _CannotImplement._fallback = _AlwaysSupported
        try:
            fn = common._resolve_dispatch_fn(_NeverSupported, config=None)
            assert fn is _AlwaysSupported.apply
        finally:
            del _NeverSupported._fallback
            del _CannotImplement._fallback

    def test_dispatch_fn_stolen_from_instance(self):
        """If the resolved kernel has _dispatch_fn, that is returned instead."""
        sentinel_fn = lambda *a, **kw: None  # noqa: E731

        class _WithDispatchFn(_AlwaysSupported):
            def __init__(self, config):
                self._dispatch_fn = sentinel_fn

        fn = common._resolve_dispatch_fn(_WithDispatchFn, config=None)
        assert fn is sentinel_fn


def _simple_dispatcher(predicate, primary, fallback_fn):
    """Minimal w16a16-style dispatcher for testing."""
    def dispatch(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if predicate(x, weight, bias):
            return primary.apply(x, weight, bias)
        return fallback_fn(x, weight, bias)
    return dispatch


_OP_COUNTER: dict[str, int] = {}


def _unique_op_name(suffix: str) -> str:
    _OP_COUNTER[suffix] = _OP_COUNTER.get(suffix, 0) + 1
    return f"test_predicated_{suffix}_{_OP_COUNTER[suffix]}"


class TestMakePredicated:

    def test_predicate_true_routes_to_primary(self):
        primary_sentinel = torch.tensor([10.0])
        fallback_sentinel = torch.tensor([20.0])

        class _Primary(_AlwaysSupported):
            @staticmethod
            def apply(x, weight, bias):
                return primary_sentinel

        class _Fallback(_AlwaysSupported):
            @staticmethod
            def apply(x, weight, bias):
                return fallback_sentinel

        @common.make_predicated(
            name=_unique_op_name("true_routes_primary"),
            predicate=lambda x, w, b: True,
            fallback=_Fallback,
            dispatcher_fn=_simple_dispatcher,
            fake_impl=_Fallback.apply,
        )
        class _Predicated(_Primary):
            pass

        inst = _Predicated(config=None)
        result = inst._dispatch_fn(torch.zeros(1, 4), torch.zeros(4, 4), None)
        assert result is primary_sentinel

    def test_predicate_false_routes_to_fallback(self):
        primary_sentinel = torch.tensor([10.0])
        fallback_sentinel = torch.tensor([20.0])

        class _Primary(_AlwaysSupported):
            @staticmethod
            def apply(x, weight, bias):
                return primary_sentinel

        class _Fallback(_AlwaysSupported):
            @staticmethod
            def apply(x, weight, bias):
                return fallback_sentinel

        @common.make_predicated(
            name=_unique_op_name("false_routes_fallback"),
            predicate=lambda x, w, b: False,
            fallback=_Fallback,
            dispatcher_fn=_simple_dispatcher,
            fake_impl=_Fallback.apply,
        )
        class _Predicated(_Primary):
            pass

        inst = _Predicated(config=None)
        result = inst._dispatch_fn(torch.zeros(1, 4), torch.zeros(4, 4), None)
        assert result is fallback_sentinel

    def test_registration_guard_fires_once(self):
        register_calls: list[int] = []

        @common.make_predicated(
            name=_unique_op_name("registration_guard"),
            predicate=lambda x, w, b: True,
            fallback=_AlwaysSupported,
            dispatcher_fn=_simple_dispatcher,
            fake_impl=_AlwaysSupported.apply,
        )
        class _Predicated(_AlwaysSupported):
            @staticmethod
            def apply(x, weight, bias):
                return torch.tensor([1.0])

        # Reset the guard so we can observe it being set on the first call.
        _Predicated._registered = False

        with patch(
            "vllm.utils.torch_utils.direct_register_custom_op",
            side_effect=lambda *a, **kw: register_calls.append(1),
        ):
            _Predicated(config=None)
            _Predicated(config=None)
            _Predicated(config=None)

        assert len(register_calls) == 1, (
            "direct_register_custom_op must be called exactly once "
            "regardless of how many layer instances are created"
        )

    def test_predicated_kernel_name(self):
        op_name = _unique_op_name("naming")

        @common.make_predicated(
            name=op_name,
            predicate=lambda x, w, b: True,
            fallback=_AlwaysSupported,
            dispatcher_fn=_simple_dispatcher,
            fake_impl=_AlwaysSupported.apply,
        )
        class _Predicated(_AlwaysSupported):
            @staticmethod
            def apply(x, weight, bias):
                return torch.tensor([1.0])

        assert _Predicated.__name__ == "_Predicated"
        assert _Predicated.__module__ == __name__

    def test_fallback_chain_resolved_transitively(self):
        """Predicate=False on outer + inner resolves to terminal apply."""
        terminal_sentinel = torch.tensor([99.0])

        class _Terminal(_AlwaysSupported):
            @staticmethod
            def apply(x, weight, bias):
                return terminal_sentinel

        @common.make_predicated(
            name=_unique_op_name("inner_chain"),
            predicate=lambda x, w, b: False,
            fallback=_Terminal,
            dispatcher_fn=_simple_dispatcher,
            fake_impl=_Terminal.apply,
        )
        class _Inner(_AlwaysSupported):
            pass

        @common.make_predicated(
            name=_unique_op_name("outer_chain"),
            predicate=lambda x, w, b: False,
            fallback=_Inner,
            dispatcher_fn=_simple_dispatcher,
            fake_impl=_Terminal.apply,
        )
        class _Outer(_AlwaysSupported):
            pass

        inst = _Outer(config=None)
        result = inst._dispatch_fn(torch.zeros(1, 4), torch.zeros(4, 4), None)
        assert result is terminal_sentinel

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike(), reason="op is registered for the platform GPU dispatch key"
    )
    def test_make_predicated_registers_in_torch_ops(self):
        """After instantiation the op is accessible via torch.ops.vllm and
        produces the same result as the underlying dispatch function."""
        device = torch.device("cuda")
        primary_sentinel = torch.tensor([42.0], device=device)

        class _Primary(_AlwaysSupported):
            @staticmethod
            def apply(
                x: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor | None,
            ) -> torch.Tensor:
                return primary_sentinel

        class _Fallback(_AlwaysSupported):
            @staticmethod
            def apply(
                x: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor | None,
            ) -> torch.Tensor:
                return torch.tensor([0.0], device=device)

        op_name = _unique_op_name("torch_ops_reg")

        @common.make_predicated(
            name=op_name,
            predicate=lambda x, w, b: True,
            fallback=_Fallback,
            dispatcher_fn=_simple_dispatcher,
            fake_impl=_Fallback.apply,
        )
        class _Predicated(_Primary):
            pass

        _Predicated(config=None)

        assert hasattr(torch.ops.vllm, op_name)

        x = torch.zeros(1, 4, device=device)
        w = torch.zeros(4, 4, device=device)
        assert torch.equal(
            getattr(torch.ops.vllm, op_name)(x, w, None),
            primary_sentinel,
        )

    def test_make_predicated_raises_if_apply_not_overridden(self):
        """Decorating a kernel that does not override apply raises TypeError."""
        with pytest.raises(TypeError, match="does not override 'apply'"):
            @common.make_predicated(
                name=_unique_op_name("no_apply_guard"),
                predicate=lambda x, w, b: True,
                fallback=_AlwaysSupported,
                dispatcher_fn=_simple_dispatcher,
                fake_impl=_AlwaysSupported.apply,
            )
            class _NoApply(_AlwaysSupported):
                pass
