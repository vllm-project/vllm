# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the OperatorIssue.operator_str patch in env_override.py.

Validates that `_apply_inductor_operator_str_patch` replaces an
`OperatorIssue`-shaped class's `operator_str` with a bounded version that:

  * Renders cheap, useful types (primitives, ``torch.dtype``,
    ``torch.device``, ``torch.Size``, primitive containers) by value so
    the "Creating implicit fallback for: ..." log line stays useful.
  * Never calls ``__str__``/``__repr__`` on opaque arguments — which is
    what was hanging Inductor for many minutes when a deep DAG-shaped IR
    flowed into a fallback custom op (e.g. ``vllm.all_reduce.default``
    with the ROCm AITER MLA fused-RoPE+KV-cache+concat pass enabled).
"""

import time

import pytest
import torch

from vllm.env_override import _apply_inductor_operator_str_patch


def _fresh_operator_issue():
    """Return a stub class shaped like ``torch._inductor.exc.OperatorIssue``.

    Each call returns a fresh class so each test gets an unpatched
    starting point and tests cannot interfere with each other or with
    the real patch installed on import.
    """

    class _OperatorIssueStub:
        @staticmethod
        def operator_str(target, args, kwargs):  # pragma: no cover - unused
            # Mimic the real (unbounded) implementation just well enough
            # that any test which forgets to patch will fail loudly.
            lines = [f"target: {target}"]
            lines += [f"args[{i}]: {a}" for i, a in enumerate(args)]
            if kwargs:
                lines.append(
                    "kwargs: {"
                    + ", ".join(f"{k}={v}" for k, v in kwargs.items())
                    + "}"
                )
            return "\n".join(lines)

    return _OperatorIssueStub


class _StrCounter:
    """Object that counts every ``__str__``/``__repr__`` invocation.

    Used as a stand-in for an ``IRNode``-shaped object whose stringifier
    is the unbounded recursion we are trying to avoid.
    """

    def __init__(self):
        self.calls = 0

    def __str__(self):
        self.calls += 1
        return "<should-not-be-called>"

    def __repr__(self):
        self.calls += 1
        return "<should-not-be-called>"


class _ExplodingStr:
    """Object whose stringifier deliberately raises.

    The unpatched ``operator_str`` would surface this exception (or hang
    on a recursive variant); the patched version must never invoke it.
    """

    def __str__(self):  # pragma: no cover - must never be called
        raise RuntimeError("operator_str should never call __str__ on opaque args")

    __repr__ = __str__


class TestApplyInductorOperatorStrPatch:
    def test_opaque_args_use_type_and_id_not_str(self):
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        opaque1 = _StrCounter()
        opaque2 = _StrCounter()
        out = cls.operator_str(
            "vllm.all_reduce.default", [opaque1], {"group": opaque2}
        )

        assert opaque1.calls == 0, "opaque arg must not be stringified"
        assert opaque2.calls == 0, "opaque kwarg must not be stringified"
        assert "tests.test_inductor_operator_str_patch._StrCounter@" in out
        assert f"@{id(opaque1):#x}" in out
        assert f"@{id(opaque2):#x}" in out

    def test_opaque_args_with_raising_stringifier_do_not_raise(self):
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        out = cls.operator_str("op", [_ExplodingStr()], {"k": _ExplodingStr()})

        assert "tests.test_inductor_operator_str_patch._ExplodingStr@" in out

    def test_primitive_args_rendered_by_value(self):
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        out = cls.operator_str(
            "op",
            [1, 2.5, True, "tp", None],
            {"flag": False, "name": "hidden"},
        )

        assert "args[0]: 1" in out
        assert "args[1]: 2.5" in out
        assert "args[2]: True" in out
        assert "args[3]: 'tp'" in out
        assert "args[4]: None" in out
        assert "flag=False" in out
        assert "name='hidden'" in out

    def test_torch_metadata_types_rendered_by_value(self):
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        out = cls.operator_str(
            "op",
            [torch.float16, torch.device("cpu"), torch.Size([4682, 16384])],
            {},
        )

        assert "args[0]: torch.float16" in out
        assert "args[1]: device(type='cpu')" in out
        assert "args[2]: torch.Size([4682, 16384])" in out

    def test_primitive_containers_rendered_by_value(self):
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        out = cls.operator_str("op", [[1, 2, 3], (4, 5)], {})

        assert "args[0]: [1, 2, 3]" in out
        assert "args[1]: (4, 5)" in out

    def test_mixed_container_falls_back_to_type_and_id(self):
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        opaque = _StrCounter()
        mixed = [1, opaque]
        out = cls.operator_str("op", [mixed], {})

        assert opaque.calls == 0
        assert "builtins.list@" in out
        assert f"@{id(mixed):#x}" in out

    def test_format_shape_target_args_kwargs(self):
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        out = cls.operator_str("vllm.all_reduce.default", [1, 2], {"k": "v"})

        assert "  target: vllm.all_reduce.default" in out
        assert "  args[0]: 1" in out
        assert "  args[1]: 2" in out
        assert "  kwargs: {k='v'}" in out

    def test_no_kwargs_line_when_kwargs_empty(self):
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        out = cls.operator_str("op", [1], {})

        assert "kwargs:" not in out

    def test_bounded_runtime_on_pathological_arg(self):
        """Regression: 100 args with O(N) slow `__str__` must not be called.

        The original hang was combinatorial; we don't try to reproduce
        the full DAG explosion here. Instead we assert the stronger
        contract that the patched function does not invoke ``__str__``
        on opaque args at all, and verify it returns within a tiny
        wall-clock budget regardless of how slow those stringifiers are.
        """
        cls = _fresh_operator_issue()
        _apply_inductor_operator_str_patch(cls)

        class _SlowStr:
            def __str__(self):  # pragma: no cover - must never be called
                time.sleep(10.0)
                return "slow"

            __repr__ = __str__

        args = [_SlowStr() for _ in range(100)]
        start = time.perf_counter()
        out = cls.operator_str("op", args, {})
        elapsed = time.perf_counter() - start

        # The unpatched implementation would take ~1000 seconds on this
        # input; one second is a generous upper bound that gives the
        # test ample headroom on slow CI hardware.
        assert elapsed < 1.0, f"patched operator_str too slow: {elapsed:.3f}s"
        assert out.count("args[") == 100

    def test_idempotent(self):
        cls = _fresh_operator_issue()

        _apply_inductor_operator_str_patch(cls)
        first_fn = cls.operator_str
        _apply_inductor_operator_str_patch(cls)

        assert cls.operator_str is first_fn

    def test_sentinel_attribute_set(self):
        cls = _fresh_operator_issue()

        assert not getattr(cls, "_vllm_safe_operator_str_patched", False)

        _apply_inductor_operator_str_patch(cls)

        assert cls._vllm_safe_operator_str_patched is True  # type: ignore[attr-defined]


def test_patch_applied_in_current_environment():
    """Integration: verify the real torch class is patched on import.

    ``vllm.env_override`` is imported transitively when ``vllm`` itself
    is imported, so by the time this test runs the real
    ``torch._inductor.exc.OperatorIssue`` should already carry the
    sentinel.
    """
    pytest.importorskip("torch._inductor.exc")
    from torch._inductor.exc import OperatorIssue

    import vllm.env_override  # noqa: F401  - ensure patch has been applied

    assert getattr(OperatorIssue, "_vllm_safe_operator_str_patched", False) is True

    # Sanity-check the bounded format on the real class with an opaque
    # arg whose `__str__` would otherwise be called.
    counter = _StrCounter()
    out = OperatorIssue.operator_str("op", [counter], {})
    assert counter.calls == 0
    assert "@" in out
