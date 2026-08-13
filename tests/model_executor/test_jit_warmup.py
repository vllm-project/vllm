# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm.model_executor.warmup.jit_warmup import (
    JitWarmupRegistry,
    VllmJitKernel,
    WarmupIntRange,
    get_ast_full_name,
    zip_inputs,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    triton_scalar_specialization_rep,
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-(1 << 63), 1 << 31),
        (-(1 << 31) - 1, (1 << 31) + 1),
        (-(1 << 31), 16),
        (0, 16),
        (1, 1),
        (2, 2),
        (16, 16),
        ((1 << 31) - 1, 2),
        (1 << 31, 1 << 31),
        ((1 << 31) + 1, (1 << 31) + 1),
        ((1 << 63) - 1, (1 << 31) + 1),
        (1 << 63, 1 << 63),
        ((1 << 63) + 1, (1 << 63) + 1),
        ((1 << 64) - 1, (1 << 63) + 1),
    ],
)
def test_triton_scalar_specialization_rep(value: int, expected: int) -> None:
    assert triton_scalar_specialization_rep(value) == expected


@pytest.mark.parametrize("value", [-(1 << 63) - 1, 1 << 64])
def test_triton_scalar_specialization_rep_rejects_out_of_range(value: int) -> None:
    with pytest.raises(OverflowError, match="outside Triton's scalar range"):
        triton_scalar_specialization_rep(value)


def test_trace_dispatch_expands_ranges_dedupes_and_ignores_unused_inputs() -> None:
    cfg = _config()

    assert ToyKernel().get_warmup_keys(5, cfg) == [
        ToyKernel.CompileKey(1, 1, 1, ("base", "default", -1, 1, 1), True),
        ToyKernel.CompileKey(2, 2, 1, ("base", "default", -2, 2, 4), True),
        ToyKernel.CompileKey(4, 4, 1, ("base", "default", -4, 1, 16), True),
        ToyKernel.CompileKey(8, 8, 1, ("base", "default", -8, 2, 64), True),
    ]


def test_warmup_range_uses_custom_advancement() -> None:
    keys = ToyKernel()._trace_dispatch(ToyKernel().dispatch)(
        tokens=WarmupIntRange(
            1,
            10,
            advance=lambda value: _next_power_of_2(value) + 1,
        ),
        cfg=_config(),
    )

    assert [key.block_size for key in keys] == [1, 2, 4, 8, 16]


def test_warmup_range_validates_custom_advancement() -> None:
    trace_dispatch = ToyKernel()._trace_dispatch(ToyKernel().dispatch)

    with pytest.raises(ValueError, match="both step and advance"):
        trace_dispatch(
            tokens=WarmupIntRange(
                1,
                10,
                step=2,
                advance=lambda value: value + 1,
            ),
            cfg=_config(),
        )
    with pytest.raises(ValueError, match="must return a greater value"):
        trace_dispatch(
            tokens=WarmupIntRange(1, 10, advance=lambda value: value),
            cfg=_config(),
        )


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


def test_trace_dispatch_filters_with_traced_predicate() -> None:
    class PredicateKernel(ToyKernel):
        def _is_valid_warmup_dispatch(
            self,
            *,
            tokens: int,
            lanes: int,
            max_work: int,
        ) -> bool:
            block_size = _next_power_of_2(tokens)
            return block_size * lanes <= max_work

        def get_warmup_keys(
            self, max_tokens: int, cfg: Any
        ) -> list[ToyKernel.CompileKey]:
            return self._trace_dispatch(self.dispatch)(
                tokens=WarmupIntRange(1, max_tokens + 1),
                lanes=(1, 2),
                cfg=cfg,
                max_work=4,
                _when=self._is_valid_warmup_dispatch,
            )

    assert PredicateKernel().get_warmup_keys(5, _config()) == [
        ToyKernel.CompileKey(1, 1, 1, ("base", "default", -1, 1, 1), True),
        ToyKernel.CompileKey(1, 2, 1, ("base", "default", -1, 1, 1), True),
        ToyKernel.CompileKey(2, 2, 1, ("base", "default", -2, 2, 4), True),
        ToyKernel.CompileKey(2, 4, 1, ("base", "default", -2, 2, 4), True),
        ToyKernel.CompileKey(4, 4, 1, ("base", "default", -4, 1, 16), True),
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


def test_dispatch_helper_calls_resolve_python_builtins() -> None:
    class BuiltinKernel(VllmJitKernel["BuiltinKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            value: int

        def dispatch(  # type: ignore[override]
            self,
            *,
            tokens: int,
            limit: int,
        ) -> CompileKey:
            return self.CompileKey(value=max(1, min(tokens, limit)))

        def get_warmup_keys(self) -> list[CompileKey]:
            return []

        def compile(self, compile_key: CompileKey) -> None:
            pass

    assert BuiltinKernel().compile_key({"tokens": 8, "limit": 4}) == (
        BuiltinKernel.CompileKey(value=4)
    )


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
    with pytest.raises(ValueError, match="may only unpack its variadic keyword"):
        KwargsReturnKernel()


def test_dispatch_can_forward_compile_key_fields() -> None:
    class ForwardingKernel(VllmJitKernel["ForwardingKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            mode: int
            block_size: int

        def dispatch(  # type: ignore[override]
            self,
            *,
            tokens: int,
            **compile_key_fields: int,
        ) -> CompileKey:
            return self.CompileKey(
                **compile_key_fields,
                block_size=_round_up(tokens, multiple=4),
            )

        def get_warmup_keys(self) -> list[CompileKey]:
            return self._trace_dispatch(self.dispatch)(
                tokens=(1, 5),
                mode=(2, 3),
            )

        def compile(self, compile_key: CompileKey) -> None:
            pass

    kernel = ForwardingKernel()
    expected = kernel.CompileKey(
        mode=2,
        block_size=8,
    )
    assert kernel.dispatch(tokens=5, mode=2) == expected
    assert kernel.compile_key({"tokens": 5, "mode": 2}) == expected
    assert kernel.get_warmup_keys() == [
        kernel.CompileKey(mode=2, block_size=4),
        kernel.CompileKey(mode=3, block_size=4),
        kernel.CompileKey(mode=2, block_size=8),
        kernel.CompileKey(mode=3, block_size=8),
    ]
    with pytest.raises(TypeError, match="field 'block_size' is specified twice"):
        kernel.compile_key({"tokens": 5, "mode": 2, "block_size": 4})
    with pytest.raises(TypeError, match="unexpected keyword argument 'extra'"):
        kernel.compile_key({"tokens": 5, "mode": 2, "extra": 1})


def test_dispatch_supports_tuple_and_mapping_subscriptions() -> None:
    class SubscriptKernel(VllmJitKernel["SubscriptKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            first: int
            named: int

        def dispatch(  # type: ignore[override]
            self,
            *,
            values: tuple[int, ...],
            config: dict[str, int],
        ) -> CompileKey:
            return self.CompileKey(first=values[0], named=config["named"])

        def get_warmup_keys(self) -> list[CompileKey]:
            return []

        def compile(self, compile_key: CompileKey) -> None:
            pass

    assert SubscriptKernel().compile_key(
        {"values": (3, 5), "config": {"named": 7}}
    ) == SubscriptKernel.CompileKey(first=3, named=7)


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


def test_runtime_cache_miss_compiles_and_caches_executor() -> None:
    class CachedKernel(VllmJitKernel["CachedKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            value: int

        def __init__(self) -> None:
            self.compiled: list[CachedKernel.CompileKey] = []
            super().__init__()

        def dispatch(self, *, value: int) -> CompileKey:  # type: ignore[override]
            return self.CompileKey(value=value)

        def get_warmup_keys(self) -> list[CompileKey]:
            return self._trace_dispatch(self.dispatch)(value=1)

        def compile(self, compile_key: CompileKey) -> None:
            self.compiled.append(compile_key)
            self._compiled_cache[compile_key] = object()

        def __call__(self, value: int) -> None:
            compile_key = self.dispatch(value=value)
            self._get_or_compile(compile_key)

    kernel = CachedKernel()
    kernel.warmup()
    kernel(1)
    kernel(2)

    assert kernel.compiled == [
        CachedKernel.CompileKey(value=1),
        CachedKernel.CompileKey(value=2),
    ]


def test_registry_records_only_inside_model_setup_context() -> None:
    registry = JitWarmupRegistry(_config())
    kernel = RecordingToyKernel()

    kernel.register_warmup(3, _config())
    with registry.activate():
        kernel.register_warmup(3, _config())

    assert len(registry) == 1
    assert kernel.compiled == []


def test_registry_expands_requests_and_deduplicates_owner_keys() -> None:
    registry = JitWarmupRegistry(_config())
    kernel = RecordingToyKernel()

    with registry.activate():
        kernel.register_warmup(3, _config())
        kernel.register_warmup(5, _config())

    registry.warmup()

    assert kernel.compiled == [
        ToyKernel.CompileKey(1, 1, 1, ("base", "default", -1, 1, 1), True),
        ToyKernel.CompileKey(2, 2, 1, ("base", "default", -2, 2, 4), True),
        ToyKernel.CompileKey(4, 4, 1, ("base", "default", -4, 1, 16), True),
        ToyKernel.CompileKey(8, 8, 1, ("base", "default", -8, 2, 64), True),
    ]


def test_registry_passes_vllm_config_to_default_requests() -> None:
    class ConfigKernel(VllmJitKernel["ConfigKernel.CompileKey"]):
        @dataclass(frozen=True)
        class CompileKey:
            value: int

        def __init__(self) -> None:
            self.compiled: list[ConfigKernel.CompileKey] = []
            super().__init__()

        def dispatch(self, *, value: int) -> CompileKey:  # type: ignore[override]
            return self.CompileKey(value=value)

        def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
            return [self.dispatch(value=vllm_config.bias)]

        def compile(self, compile_key: CompileKey) -> None:
            self.compiled.append(compile_key)

    registry = JitWarmupRegistry(_config(bias=7))
    kernel = ConfigKernel()

    with registry.activate():
        kernel.register_warmup()
        kernel.register_warmup()
    assert len(registry) == 1
    registry.warmup()

    assert kernel.compiled == [ConfigKernel.CompileKey(value=7)]


def test_get_ast_full_name_handles_names_attributes_and_other_nodes() -> None:
    dotted_expr = ast.parse("foo.bar.baz").body[0]
    call_expr = ast.parse("foo()").body[0]
    assert isinstance(dotted_expr, ast.Expr)
    assert isinstance(call_expr, ast.Expr)

    assert get_ast_full_name(dotted_expr.value) == "foo.bar.baz"
    assert get_ast_full_name(call_expr.value) is None
