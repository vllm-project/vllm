# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
    get_ast_full_name,
    zip_inputs,
)


def _next_power_of_2(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def _round_up(value: int, *, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _config(
    *,
    bias: int = 0,
    disabled: bool = False,
    name: str = "base",
    vectorized: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        bias=bias,
        disabled=disabled,
        name=name,
        vectorized=vectorized,
    )


class ToyKernel(VllmJitKernel["ToyKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        block_size: int
        work: int
        vector_width: int
        descriptor: tuple[object, ...]
        enabled: bool

    def dispatch(  # type: ignore[override]
        self,
        *,
        tokens: int,
        cfg: Any,
        lanes: int = 1,
        mode: str = "default",
        debug: int = 0,
    ) -> CompileKey:
        block_size = _next_power_of_2(tokens)
        work: int = block_size * lanes + cfg.bias
        return self.CompileKey(
            block_size=block_size,
            work=work,
            vector_width=4 if cfg.vectorized and block_size >= 4 else 1,
            descriptor=(
                cfg.name,
                mode,
                -block_size,
                block_size % 3,
                block_size**2,
            ),
            enabled=not cfg.disabled,
        )

    def get_warmup_keys(self, max_tokens: int, cfg: Any) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            tokens=WarmupIntRange(1, max_tokens + 1),
            cfg=cfg,
            # This argument is intentionally unused by dispatch expressions.
            debug=WarmupIntRange(0, 100),
        )

    def compile(self, compile_key: CompileKey) -> None:
        pass


class RecordingToyKernel(ToyKernel):
    def __init__(self) -> None:
        self.compiled: list[ToyKernel.CompileKey] = []
        super().__init__()

    def compile(self, compile_key: ToyKernel.CompileKey) -> None:
        self.compiled.append(compile_key)


def test_trace_dispatch_expands_ranges_dedupes_and_ignores_unused_inputs() -> None:
    cfg = _config()

    assert ToyKernel().get_warmup_keys(5, cfg) == [
        ToyKernel.CompileKey(1, 1, 1, ("base", "default", -1, 1, 1), True),
        ToyKernel.CompileKey(2, 2, 1, ("base", "default", -2, 2, 4), True),
        ToyKernel.CompileKey(4, 4, 1, ("base", "default", -4, 1, 16), True),
        ToyKernel.CompileKey(8, 8, 1, ("base", "default", -8, 2, 64), True),
    ]


def test_compile_key_uses_defaults_locals_attributes_and_expressions() -> None:
    cfg = _config(bias=3, disabled=True, name="cfg", vectorized=True)

    assert ToyKernel().compile_key(
        {
            "tokens": 4,
            "cfg": cfg,
            "lanes": 2,
        }
    ) == ToyKernel.CompileKey(
        block_size=4,
        work=11,
        vector_width=4,
        descriptor=("cfg", "default", -4, 1, 16),
        enabled=False,
    )


def test_trace_dispatch_combines_zipped_rows_with_independent_values() -> None:
    cfg = _config(vectorized=True)

    keys = ToyKernel()._trace_dispatch(ToyKernel().dispatch)(
        zip_inputs(
            dict(tokens=1, mode="small"),
            dict(tokens=4, mode="wide"),
        ),
        cfg=cfg,
        lanes=(1, 2),
    )

    assert keys == [
        ToyKernel.CompileKey(1, 1, 1, ("base", "small", -1, 1, 1), True),
        ToyKernel.CompileKey(1, 2, 1, ("base", "small", -1, 1, 1), True),
        ToyKernel.CompileKey(4, 4, 4, ("base", "wide", -4, 1, 16), True),
        ToyKernel.CompileKey(4, 8, 4, ("base", "wide", -4, 1, 16), True),
    ]


def test_zip_inputs_validates_input_rows() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        zip_inputs()
    with pytest.raises(ValueError, match="rows must be mappings"):
        zip_inputs(cast(Any, ("tokens", 1)))
    with pytest.raises(ValueError, match="at least one dispatch input name"):
        zip_inputs({})
    with pytest.raises(ValueError, match="dispatch input names must be strings"):
        zip_inputs(cast(Any, {1: 2}))
    with pytest.raises(ValueError, match="same dispatch input names"):
        zip_inputs({"tokens": 1}, {"mode": "small"})


def test_trace_dispatch_rejects_bad_positional_groups_and_duplicates() -> None:
    kernel = ToyKernel()

    with pytest.raises(TypeError, match="zip_inputs"):
        kernel._trace_dispatch(kernel.dispatch)(
            cast(Any, {"tokens": 1}),
            cfg=_config(),
        )

    with pytest.raises(ValueError, match="specified more than once"):
        kernel._trace_dispatch(kernel.dispatch)(
            zip_inputs(dict(tokens=1, mode="small")),
            tokens=2,
            cfg=_config(),
        )


def test_helper_calls_support_keywords_and_reject_star_kwargs() -> None:
    class HelperKernel(VllmJitKernel["HelperKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            value: int

        def dispatch(  # type: ignore[override]
            self,
            *,
            tokens: int,
            block_size: int,
        ) -> CompileKey:
            return self.CompileKey(value=_round_up(tokens, multiple=block_size))

        def get_warmup_keys(self) -> list[CompileKey]:
            return []

        def compile(self, compile_key: CompileKey) -> None:
            pass

    class StarKwargsKernel(VllmJitKernel["StarKwargsKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            value: int

        def dispatch(  # type: ignore[override]
            self,
            *,
            tokens: int,
            block_size: int,
        ) -> CompileKey:
            return self.CompileKey(value=_round_up(tokens, **{"multiple": block_size}))

        def get_warmup_keys(self) -> list[CompileKey]:
            return []

        def compile(self, compile_key: CompileKey) -> None:
            pass

    assert HelperKernel().compile_key(
        {
            "tokens": 5,
            "block_size": 4,
        }
    ) == HelperKernel.CompileKey(value=8)
    with pytest.raises(ValueError, match=r"cannot use \*\*kwargs"):
        StarKwargsKernel().compile_key({"tokens": 5, "block_size": 4})


def test_dispatch_body_must_be_local_assignments_then_compile_key_return() -> None:
    class BranchKernel(VllmJitKernel["BranchKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            value: int

        def dispatch(self, *, value: int) -> CompileKey:  # type: ignore[override]
            if value > 0:
                value = 1
            return self.CompileKey(value=value)

        def get_warmup_keys(self) -> list[CompileKey]:
            return []

        def compile(self, compile_key: CompileKey) -> None:
            pass

    class KwargsReturnKernel(VllmJitKernel["KwargsReturnKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            value: int

        def dispatch(self, *, value: int) -> CompileKey:  # type: ignore[override]
            return self.CompileKey(**{"value": value})

        def get_warmup_keys(self) -> list[CompileKey]:
            return []

        def compile(self, compile_key: CompileKey) -> None:
            pass

    with pytest.raises(ValueError, match="local assignments"):
        BranchKernel()
    with pytest.raises(ValueError, match=r"cannot use \*\*kwargs in CompileKey"):
        KwargsReturnKernel()


def test_dispatch_reports_unsupported_expression_with_context() -> None:
    class UnsupportedKernel(VllmJitKernel["UnsupportedKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            value: object

        def dispatch(self, *, value: int) -> CompileKey:  # type: ignore[override]
            return self.CompileKey(value={value})

        def get_warmup_keys(self) -> list[CompileKey]:
            return []

        def compile(self, compile_key: CompileKey) -> None:
            pass

    with pytest.raises(ValueError) as exc_info:
        UnsupportedKernel().compile_key({"value": 1})

    message = str(exc_info.value)
    assert "Unsupported dispatch expression" in message
    assert "{value}" in message
    assert "Supported dispatch expressions" in message


def test_warmup_compiles_all_returned_keys_in_order() -> None:
    kernel = RecordingToyKernel()
    cfg = _config()

    kernel.warmup(3, cfg)

    assert kernel.compiled == [
        ToyKernel.CompileKey(1, 1, 1, ("base", "default", -1, 1, 1), True),
        ToyKernel.CompileKey(2, 2, 1, ("base", "default", -2, 2, 4), True),
        ToyKernel.CompileKey(4, 4, 1, ("base", "default", -4, 1, 16), True),
    ]


def test_get_ast_full_name_handles_names_attributes_and_other_nodes() -> None:
    dotted_expr = ast.parse("foo.bar.baz").body[0]
    call_expr = ast.parse("foo()").body[0]
    assert isinstance(dotted_expr, ast.Expr)
    assert isinstance(call_expr, ast.Expr)

    assert get_ast_full_name(dotted_expr.value) == "foo.bar.baz"
    assert get_ast_full_name(call_expr.value) is None
