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


def test_eagle_prepare_inputs_dispatch_covers_single_and_multi_request() -> None:
    from vllm.v1.spec_decode.utils import EaglePrepareInputsPaddedKernel

    kernel = EaglePrepareInputsPaddedKernel()
    keys = kernel._trace_dispatch(kernel.dispatch)(
        num_reqs=[1, 50],
    )
    assert keys == [
        EaglePrepareInputsPaddedKernel.CompileKey(num_reqs=1),
        EaglePrepareInputsPaddedKernel.CompileKey(num_reqs=50),
    ]


def test_eagle_step_slot_mapping_stride_follows_n_blocks() -> None:
    from vllm.v1.spec_decode.utils import EagleStepSlotMappingMetadataKernel

    kernel = EagleStepSlotMappingMetadataKernel()
    keys = kernel._trace_dispatch(kernel.dispatch)(
        block_size=16,
        max_model_len=8192,
        n_blocks_per_req=[512, 1024],
        PAD_ID=-1,
        batch_size=[1, 50],
    )
    assert len(keys) == 4
    for k in keys:
        assert k.block_table_stride == k.n_blocks_per_req


def test_eagle_prepare_next_token_stride_follows_num_sampled() -> None:
    from vllm.v1.spec_decode.utils import EaglePrepareNextTokenPaddedKernel

    kernel = EaglePrepareNextTokenPaddedKernel()
    keys = kernel._trace_dispatch(kernel.dispatch)(
        BLOCK_SIZE_TOKENS=16,
        vocab_size=129280,
        num_sampled_tokens_per_req=[1, 9],
        num_reqs=[1, 50],
    )
    assert len(keys) == 4
    for k in keys:
        assert k.stride_sampled_token_ids == k.num_sampled_tokens_per_req


def test_dflash_prepare_inputs_grid_enumeration() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        PrepareDflashInputsKernel,
    )

    kernel = PrepareDflashInputsKernel()
    keys = kernel._trace_dispatch(kernel.dispatch)(
        SAMPLE_FROM_ANCHOR=False,
        PAD_SLOT_ID=-1,
        BLOCK_SIZE=[1, 256],
        block_table_stride=128,
        parallel_drafting_token_id=0,
        block_size=128,
        num_query_per_req=8,
        num_speculative_steps=8,
        max_num_reqs=50,
        max_num_tokens=512,
        max_model_len=8192,
        grid_num_reqs=[1, 50],
        grid_num_blocks=[1, 8],
    )
    assert len(keys) == 8  # 2 BLOCK_SIZE × 2 grid_reqs × 2 grid_blocks
    block_sizes = {k.BLOCK_SIZE for k in keys}
    assert block_sizes == {1, 256}


# ---------------------------------------------------------------------------
# Spec-decode wrapper: dispatch + get_warmup_keys end-to-end coverage.
#
# These tests build a minimal ``VllmConfig``-shaped ``SimpleNamespace`` mock
# so ``get_warmup_keys(...)`` can run without a real model.  They verify the
# CompileKey search space each wrapper enumerates matches the runtime
# specializations documented in the wrapper docstring.
# ---------------------------------------------------------------------------


def _spec_vllm_config(
    *,
    num_speculative_tokens: int = 8,
    max_num_seqs: int = 50,
    vocab_size: int = 129280,
    hidden_size: int = 4096,
    block_size: int = 16,
    max_model_len: int = 8192,
    decode_context_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
) -> Any:
    """Build a minimal VllmConfig-shaped mock for spec-decode warmup tests."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            get_vocab_size=lambda: vocab_size,
            get_hidden_size=lambda: hidden_size,
            hf_config=SimpleNamespace(hidden_size=hidden_size),
            max_model_len=max_model_len,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
        ),
        cache_config=SimpleNamespace(block_size=block_size),
    )


def test_mtp_shared_head_rmsnorm_dispatch_and_warmup_keys() -> None:
    from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
        MtpSharedHeadRmsnormKernel,
    )

    kernel = MtpSharedHeadRmsnormKernel()

    # dispatch: BLOCK_SIZE follows HIDDEN via next_power_of_2
    key = kernel.dispatch(HIDDEN=4096)
    assert key.HIDDEN == 4096
    assert key.BLOCK_SIZE == 4096

    # get_warmup_keys: exactly one key per deployment (HIDDEN is config-fixed)
    cfg = _spec_vllm_config(hidden_size=4096)
    keys = kernel.get_warmup_keys(cfg)
    assert keys == [MtpSharedHeadRmsnormKernel.CompileKey(HIDDEN=4096, BLOCK_SIZE=4096)]

    # Non-power-of-2 HIDDEN rounds up BLOCK_SIZE
    cfg_odd = _spec_vllm_config(hidden_size=3000)
    keys_odd = kernel.get_warmup_keys(cfg_odd)
    assert keys_odd == [
        MtpSharedHeadRmsnormKernel.CompileKey(HIDDEN=3000, BLOCK_SIZE=4096)
    ]


def test_eagle_prepare_inputs_get_warmup_keys_uses_max_num_seqs() -> None:
    from vllm.v1.spec_decode.utils import EaglePrepareInputsPaddedKernel

    kernel = EaglePrepareInputsPaddedKernel()
    cfg = _spec_vllm_config(max_num_seqs=128)

    keys = kernel.get_warmup_keys(cfg)
    assert keys == [
        EaglePrepareInputsPaddedKernel.CompileKey(num_reqs=1),
        EaglePrepareInputsPaddedKernel.CompileKey(num_reqs=128),
    ]


def test_eagle_step_get_warmup_keys_enumerates_cp_and_hybrid_factors() -> None:
    from vllm.utils.math_utils import cdiv
    from vllm.v1.spec_decode.utils import EagleStepSlotMappingMetadataKernel

    kernel = EagleStepSlotMappingMetadataKernel()
    block_size = 16
    max_model_len = 8192
    cfg = _spec_vllm_config(
        block_size=block_size,
        max_model_len=max_model_len,
        max_num_seqs=64,
    )

    keys = kernel.get_warmup_keys(
        cfg, block_size=block_size, max_model_len=max_model_len
    )

    # cp_factors = {1, 1} = {1} (no CP)
    # base = cdiv(8192, 16*1) = 512
    # hybrid factors (1, 2, 4) -> n_blocks = {512, 1024, 2048}
    # batch_size = [1, 64]
    # Total = 3 n_blocks × 2 batch = 6 keys
    expected_n_blocks = {max(cdiv(max_model_len, block_size) * f, 1) for f in (1, 2, 4)}
    actual_n_blocks = {k.n_blocks_per_req for k in keys}
    assert actual_n_blocks == expected_n_blocks

    # stride always equals n_blocks_per_req (contiguous block_table)
    for k in keys:
        assert k.block_table_stride == k.n_blocks_per_req
        assert k.batch_size in (1, 64)

    assert len(keys) == len(expected_n_blocks) * 2


def test_eagle_prepare_next_token_get_warmup_keys_reads_spec_config() -> None:
    from vllm.v1.spec_decode.utils import EaglePrepareNextTokenPaddedKernel

    kernel = EaglePrepareNextTokenPaddedKernel()
    cfg = _spec_vllm_config(
        num_speculative_tokens=8,  # num_sampled = 9
        vocab_size=129280,
        max_num_seqs=50,
    )

    keys = kernel.get_warmup_keys(cfg)

    # BLOCK_SIZE_TOKENS covers all powers of two from 1 up to
    # next_power_of_2(9) = 16, so [1, 2, 4, 8, 16] (5 values).
    # num_sampled = [1, 9], num_reqs = [1, 50]
    # stride follows num_sampled
    # Total = 5 × 2 × 2 = 20 keys
    assert len(keys) == 20
    block_sizes = {k.BLOCK_SIZE_TOKENS for k in keys}
    assert block_sizes == {1, 2, 4, 8, 16}
    for k in keys:
        assert k.vocab_size == 129280
        assert k.num_sampled_tokens_per_req in (1, 9)
        assert k.stride_sampled_token_ids == k.num_sampled_tokens_per_req

    # No-op when spec decode is disabled
    cfg_disabled = _spec_vllm_config(num_speculative_tokens=0)
    assert kernel.get_warmup_keys(cfg_disabled) == []


def test_dflash_get_warmup_keys_reads_speculator_scalars() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        PrepareDflashInputsKernel,
    )

    kernel = PrepareDflashInputsKernel()

    # Mock speculator: only the attributes read by get_warmup_keys
    block_table = cast(Any, SimpleNamespace(stride=lambda dim: 128))
    block_tables = SimpleNamespace(
        kernel_block_sizes=[128],
        input_block_tables=[block_table],
    )
    speculator = SimpleNamespace(
        block_tables=block_tables,
        sample_from_anchor=False,
        parallel_drafting_token_id=0,
        num_query_per_req=8,
        num_speculative_steps=8,
        max_num_reqs=50,
        max_num_tokens=512,
        max_model_len=8192,
    )
    cfg = _spec_vllm_config()

    keys = kernel.get_warmup_keys(cfg, speculator=speculator)

    # 9 BLOCK_SIZE × 2 grid_reqs × 2 grid_blocks = 36 keys
    assert len(keys) == 36
    block_sizes = {k.BLOCK_SIZE for k in keys}
    assert block_sizes == {1, 2, 4, 8, 16, 32, 64, 128, 256}
    # All keys share the deployment-fixed scalars
    for k in keys:
        assert k.block_table_stride == 128
        assert k.SAMPLE_FROM_ANCHOR is False
        assert k.PAD_SLOT_ID == -1


# ---------------------------------------------------------------------------
# compile() argument alignment: verify that compile() passes the CompileKey
# fields to kernel.warmup() in the same order and with the same dtypes as
# __call__ passes them to kernel[grid](...).
#
# We monkey-patch ``kernel.kernel.warmup`` with a MagicMock that records the
# call, then assert the positional args and constexpr kwargs match the
# CompileKey.
# ---------------------------------------------------------------------------


def test_eagle_prepare_inputs_compile_passes_correct_args(monkeypatch) -> None:
    from vllm.v1.spec_decode.utils import EaglePrepareInputsPaddedKernel

    kernel = EaglePrepareInputsPaddedKernel()
    calls: list = []
    monkeypatch.setattr(kernel.kernel, "warmup", lambda *a, **k: calls.append((a, k)))

    key = EaglePrepareInputsPaddedKernel.CompileKey(num_reqs=50)
    kernel.compile(key)

    assert len(calls) == 1
    args, kwargs = calls[0]
    # 5 TritonWarmupTensor pointers + num_reqs scalar
    assert len(args) == 6
    assert args[5] == 50  # num_reqs
    assert kwargs == {"grid": (1,)}


def test_eagle_step_compile_passes_correct_args(monkeypatch) -> None:
    from vllm.v1.spec_decode.utils import EagleStepSlotMappingMetadataKernel

    kernel = EagleStepSlotMappingMetadataKernel()
    calls: list = []
    monkeypatch.setattr(kernel.kernel, "warmup", lambda *a, **k: calls.append((a, k)))

    key = EagleStepSlotMappingMetadataKernel.CompileKey(
        block_size=16,
        max_model_len=8192,
        n_blocks_per_req=512,
        PAD_ID=-1,
        batch_size=50,
        block_table_stride=512,
    )
    kernel.compile(key)

    assert len(calls) == 1
    args, kwargs = calls[0]
    # 5 pointers + 1 stride scalar = 6 positional
    assert len(args) == 6
    assert args[2] == 512  # block_table_stride
    assert kwargs == {
        "block_size": 16,
        "max_model_len": 8192,
        "n_blocks_per_req": 512,
        "PAD_ID": -1,
        "batch_size": 50,
        "grid": (1,),
    }


def test_dflash_compile_passes_correct_args(monkeypatch) -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        PrepareDflashInputsKernel,
    )

    kernel = PrepareDflashInputsKernel()
    calls: list = []
    monkeypatch.setattr(kernel.kernel, "warmup", lambda *a, **k: calls.append((a, k)))

    key = PrepareDflashInputsKernel.CompileKey(
        SAMPLE_FROM_ANCHOR=False,
        PAD_SLOT_ID=-1,
        BLOCK_SIZE=256,
        block_table_stride=128,
        parallel_drafting_token_id=0,
        block_size=128,
        num_query_per_req=8,
        num_speculative_steps=8,
        max_num_reqs=50,
        max_num_tokens=512,
        max_model_len=8192,
        grid_num_reqs=50,
        grid_num_blocks=8,
    )
    kernel.compile(key)

    assert len(calls) == 1
    args, kwargs = calls[0]
    # 18 pointers + 8 scalars = 26 positional
    assert len(args) == 26
    assert args[18] == 128  # block_table_stride
    assert args[19] == 0  # parallel_drafting_token_id
    assert args[20] == 128  # block_size
    assert args[21] == 8  # num_query_per_req
    assert args[22] == 8  # num_speculative_steps
    assert args[23] == 50  # max_num_reqs
    assert args[24] == 512  # max_num_tokens
    assert args[25] == 8192  # max_model_len
    assert kwargs == {
        "SAMPLE_FROM_ANCHOR": False,
        "PAD_SLOT_ID": -1,
        "BLOCK_SIZE": 256,
        "grid": (50, 8),
    }
