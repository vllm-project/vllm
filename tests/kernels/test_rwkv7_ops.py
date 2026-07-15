# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch


class OpCase(NamedTuple):
    name: str
    args: Callable[[str], tuple]
    expected: Callable[[tuple], list[tuple[tuple[int, ...], torch.dtype]]]


def _h(device: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.float16)


def _bf(device: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.bfloat16)


def _i32(device: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.int32)


def _same(index: int, count: int = 1):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        x = args[index]
        return [(tuple(x.shape), x.dtype) for _ in range(count)]

    return expected


def _last_dim(index: int, dim: int):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        x = args[index]
        return [((*x.shape[:-1], dim), x.dtype)]

    return expected


def _last_dim_from_weight(index: int, weight_index: int, weight_dim: int):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        x = args[index]
        weight = args[weight_index]
        return [((*x.shape[:-1], weight.shape[weight_dim]), x.dtype)]

    return expected


def _multi_last_dim(specs: tuple[tuple[int, int, int], ...]):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        return [
            (
                (*args[x_index].shape[:-1], args[w_index].shape[weight_dim]),
                args[x_index].dtype,
            )
            for x_index, w_index, weight_dim in specs
        ]

    return expected


def _add_last_expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
    x = args[0]
    return [((x.shape[0], x.shape[2]), x.dtype)]


def _emb_expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
    emb = args[0]
    return [(tuple(emb.shape), torch.float16)]


def _shape_expected(
    shape: tuple[int, ...],
    dtype_index: int,
):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        return [(shape, args[dtype_index].dtype)]

    return expected


def _ln_args(device: str, c: int = 8) -> tuple:
    return _h(device, (2, 3, c)), _h(device, (c,)), _h(device, (c,))


def _linear_args(device: str) -> tuple:
    return _h(device, (2, 3, 8)), _h(device, (8, 5))


def _linear_orig_args(device: str) -> tuple:
    return _h(device, (2, 3, 8)), _h(device, (5, 8))


def _rank_tensor_args(device: str) -> tuple:
    return _h(device, (2, 3, 8)), _h(device, (2, 3, 8)), _h(device, (2, 3, 8))


def _rwkv7_import_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV7 custom op registration")
    try:
        import vllm.rwkv7_ops  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"RWKV7 extension is unavailable: {exc!r}")
    import vllm._custom_ops  # noqa: F401


def _assert_repeatable(
    run: Callable[[], tuple[torch.Tensor, torch.Tensor]], repeats: int = 200
) -> None:
    expected = run()
    torch.cuda.synchronize()
    for _ in range(repeats - 1):
        actual = run()
        torch.cuda.synchronize()
        for got, want in zip(actual, expected, strict=True):
            assert torch.equal(got, want)


def _op(name: str):
    namespace, op_name = name.split("::", 1)
    return getattr(getattr(torch.ops, namespace), op_name)


V3A_RETURNING_CASES = [
    OpCase("rwkv7_v3a_ops::layer_norm_f16", _ln_args, _same(0)),
    OpCase(
        "rwkv7_v3a_ops::emb_ln0_bf16_to_f16",
        lambda d: (_bf(d, (11, 8)), _bf(d, (8,)), _bf(d, (8,))),
        _emb_expected,
    ),
    OpCase("rwkv7_v3a_ops::linear_f16", _linear_args, _last_dim(0, 5)),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_m1_splitk",
        lambda d: (_h(d, (1, 8)), _h(d, (8, 5))),
        _last_dim(0, 5),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_t_f16",
        _linear_orig_args,
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_t_act_f16",
        lambda d: (*_linear_orig_args(d), 1),
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_t_vres_f16",
        lambda d: (
            _h(d, (2, 3, 8)),
            _h(d, (5, 8)),
            _h(d, (2, 3, 5)),
            _h(d, (2, 3, 5)),
            _h(d, (5,)),
        ),
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_wag_rank_in_f16",
        lambda d: (*_rank_tensor_args(d), _h(d, (4, 8)), _h(d, (3, 8)), _h(d, (2, 8))),
        _multi_last_dim(((0, 3, 0), (1, 4, 0), (2, 5, 0))),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_wagv_rank_in_f16",
        lambda d: (
            *_rank_tensor_args(d),
            _h(d, (2, 3, 8)),
            _h(d, (4, 8)),
            _h(d, (3, 8)),
            _h(d, (2, 8)),
            _h(d, (6, 8)),
        ),
        _multi_last_dim(((0, 4, 0), (1, 5, 0), (2, 6, 0), (3, 7, 0))),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_wag_rank_out_f16",
        lambda d: (
            _h(d, (2, 3, 4)),
            _h(d, (2, 3, 3)),
            _h(d, (2, 3, 2)),
            _h(d, (8, 4)),
            _h(d, (8, 3)),
            _h(d, (8, 2)),
        ),
        _multi_last_dim(((0, 3, 0), (1, 4, 0), (2, 5, 0))),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_wagv_rank_out_f16",
        lambda d: (
            _h(d, (2, 3, 4)),
            _h(d, (2, 3, 3)),
            _h(d, (2, 3, 2)),
            _h(d, (2, 3, 6)),
            _h(d, (8, 4)),
            _h(d, (8, 3)),
            _h(d, (8, 2)),
            _h(d, (8, 6)),
            _h(d, (2, 3, 8)),
            _h(d, (2, 3, 8)),
            _h(d, (8,)),
        ),
        _multi_last_dim(((0, 4, 0), (1, 5, 0), (2, 6, 0), (3, 7, 0))),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_f16",
        lambda d: (_h(d, (2, 3, 8)), _h(d, (2, 3, 8))),
        _same(0),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_f16",
        lambda d: (*_ln_args(d), _h(d, (8,))),
        _same(0, 2),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_last_layer_norm_f16",
        lambda d: (*_ln_args(d), _h(d, (8,))),
        _add_last_expected,
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_cmix_mix_f16",
        lambda d: (
            _h(d, (2, 1, 8)),
            _h(d, (2, 1, 8)),
            _h(d, (2, 8)),
            _h(d, (8,)),
            _h(d, (8,)),
            _h(d, (8,)),
        ),
        _same(0, 2),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_cmix_mix_f16_slots",
        lambda d: (
            _h(d, (2, 1, 8)),
            _h(d, (2, 1, 8)),
            _h(d, (4, 8)),
            _h(d, (8,)),
            _h(d, (8,)),
            _h(d, (8,)),
            _i32(d, (2,)),
        ),
        _same(0, 2),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_tmix_mix6_f16",
        lambda d: (
            _h(d, (2, 1, 8)),
            _h(d, (2, 1, 8)),
            _h(d, (2, 8)),
            _h(d, (8,)),
            _h(d, (8,)),
            *[_h(d, (8,)) for _ in range(6)],
        ),
        _same(0, 7),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_tmix_mix6_f16_slots",
        lambda d: (
            _h(d, (2, 1, 8)),
            _h(d, (2, 1, 8)),
            _h(d, (4, 8)),
            _h(d, (8,)),
            _h(d, (8,)),
            *[_h(d, (8,)) for _ in range(6)],
            _i32(d, (2,)),
        ),
        _same(0, 7),
    ),
]


FAST_RETURNING_CASES = [
    OpCase(
        "rwkv7_fast_ops_fp16::tmix_mix6",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (2, 8)),
            *[_h(d, (8,)) for _ in range(6)],
        ),
        _same(3, 6),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::tmix_mix6_slot",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (5, 8)),
            _i32(d, (2,)),
            *[_h(d, (8,)) for _ in range(6)],
        ),
        _same(3, 6),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::tmix_mix6_varlen",
        lambda d: (
            2,
            5,
            8,
            _h(d, (5, 8)),
            _h(d, (6, 8)),
            _i32(d, (2,)),
            *[_h(d, (8,)) for _ in range(6)],
            _i32(d, (3,)),
            _i32(d, (5,)),
        ),
        _same(3, 6),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::tmix_kk_a_gate",
        lambda d: (
            2,
            3,
            128,
            2,
            _h(d, (2, 3, 128)),
            _h(d, (128,)),
            _h(d, (128,)),
            _h(d, (2, 3, 128)),
            _h(d, (128,)),
        ),
        _same(4, 3),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::tmix_lnx_rkvres_xg",
        lambda d: (
            2,
            3,
            128,
            2,
            *[_h(d, (2, 3, 128)) for _ in range(4)],
            _h(d, (128,)),
            _h(d, (128,)),
            _h(d, (128,)),
            _h(d, (2, 3, 128)),
        ),
        _same(4),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::tmix_vres_gate",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (2, 3, 8)),
            _h(d, (8,)),
            _h(d, (2, 3, 8)),
        ),
        _same(3),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_sparse_down_relu_one",
        lambda d: (128, 128, _h(d, (128,)), _h(d, (128, 128))),
        _shape_expected((128,), 2),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_sparse_down_relu_rows",
        lambda d: (2, 3, 128, 128, _h(d, (2, 3, 128)), _h(d, (128, 128))),
        _shape_expected((2, 3, 128), 4),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_sparse_down_relu_rows_t512",
        lambda d: (2, 3, 512, 512, _h(d, (2, 3, 512)), _h(d, (512, 512))),
        _shape_expected((2, 3, 512), 4),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_mix",
        lambda d: (2, 3, 8, _h(d, (2, 3, 8)), _h(d, (2, 8)), _h(d, (8,))),
        _same(3),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_mix_slot",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (5, 8)),
            _i32(d, (2,)),
            _h(d, (8,)),
        ),
        _same(3),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_mix_varlen",
        lambda d: (
            2,
            5,
            8,
            _h(d, (5, 8)),
            _h(d, (6, 8)),
            _i32(d, (2,)),
            _h(d, (8,)),
            _i32(d, (3,)),
            _i32(d, (5,)),
        ),
        _same(3),
    ),
    OpCase("rwkv7_fast_ops_fp16::relu_square", lambda d: (_h(d, (2, 3, 8)),), _same(0)),
    OpCase("rwkv7_fast_ops_fp16::act_tanh", lambda d: (_h(d, (2, 3, 8)),), _same(0)),
    OpCase("rwkv7_fast_ops_fp16::act_sigmoid", lambda d: (_h(d, (2, 3, 8)),), _same(0)),
    OpCase(
        "rwkv7_fast_ops_fp16::add_vec",
        lambda d: (8, _h(d, (2, 3, 8)), _h(d, (8,))),
        _same(1),
    ),
]


RETURNING_CASES = V3A_RETURNING_CASES + FAST_RETURNING_CASES


@pytest.fixture(scope="module", autouse=True)
def rwkv7_ops_registered() -> None:
    _rwkv7_import_or_skip()


def test_rwkv7_linear_op_honors_fp16_accumulation() -> None:
    torch.manual_seed(20260710)
    x = torch.randn((64, 1024), device="cuda", dtype=torch.float16)
    weight_orig = torch.randn((256, 1024), device="cuda", dtype=torch.float16)
    weight = weight_orig.t().contiguous()
    op = torch.ops.rwkv7_v3a_ops.linear_f16
    reference = torch.nn.functional.linear(x.float(), weight_orig.float()).half()

    default_accumulation = op(x, weight)
    fp32_accumulation = op(x, weight, False)
    fp16_accumulation = op(x, weight, True)
    torch.accelerator.synchronize()

    assert fp16_accumulation.shape == reference.shape
    assert fp16_accumulation.dtype == reference.dtype
    assert fp16_accumulation.device == reference.device
    assert torch.isfinite(fp16_accumulation).all()
    assert torch.equal(default_accumulation, fp32_accumulation)
    fp32_relative_l2 = (
        fp32_accumulation.float() - reference.float()
    ).norm() / reference.float().norm()
    fp16_relative_l2 = (
        fp16_accumulation.float() - reference.float()
    ).norm() / reference.float().norm()
    fp32_cosine = torch.nn.functional.cosine_similarity(
        fp32_accumulation.float().flatten(),
        reference.float().flatten(),
        dim=0,
    )
    fp16_cosine = torch.nn.functional.cosine_similarity(
        fp16_accumulation.float().flatten(),
        reference.float().flatten(),
        dim=0,
    )
    assert fp32_relative_l2 < 1e-3
    assert fp32_cosine > 0.999999
    assert fp16_relative_l2 < 5e-3
    assert fp16_cosine > 0.99999
    assert torch.count_nonzero(fp16_accumulation != fp32_accumulation) > 0


@pytest.mark.parametrize("batch,time", [(2, 1), (3, 1), (2, 2), (1, 8)])
def test_rwkv7_wkv_fp16_is_repeatable(batch: int, time: int) -> None:
    torch.manual_seed(20260714 + batch * 10 + time)
    C, H = 64, 1
    state = torch.randn((batch, H, 64, 64), device="cuda", dtype=torch.float16)
    payload = torch.randn((6, batch, time, C), device="cuda", dtype=torch.float16)
    w0 = torch.randn((C,), device="cuda", dtype=torch.float16)
    elapsed = torch.arange(batch, device="cuda", dtype=torch.int32)
    op = torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        current_state = state.clone()
        y = torch.empty((batch, time, C), device="cuda", dtype=torch.float16)
        op(
            batch,
            time,
            C,
            H,
            current_state,
            *payload[:2],
            w0,
            *payload[2:],
            y,
            elapsed,
        )
        return y, current_state

    _assert_repeatable(run)


def test_rwkv7_wkv_fp16_varlen_is_repeatable() -> None:
    torch.manual_seed(20260714)
    B, total_tokens, max_t, C, H = 2, 5, 3, 64, 1
    query_start_loc = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    slot_indices = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    state = torch.randn((5, H, 64, 64), device="cuda", dtype=torch.float16)
    payload = torch.randn((6, total_tokens, C), device="cuda", dtype=torch.float16)
    w0 = torch.randn((C,), device="cuda", dtype=torch.float16)
    elapsed = torch.arange(5, device="cuda", dtype=torch.int32)
    op = torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_varlen

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        current_state = state.clone()
        y = torch.empty((total_tokens, C), device="cuda", dtype=torch.float16)
        op(
            B,
            total_tokens,
            max_t,
            C,
            H,
            query_start_loc,
            slot_indices,
            current_state,
            *payload[:2],
            w0,
            *payload[2:],
            y,
            elapsed,
        )
        return y, current_state

    _assert_repeatable(run)


@pytest.mark.parametrize("case", RETURNING_CASES, ids=lambda c: c.name)
def test_rwkv7_returning_ops_schema_matches_meta_contract(case: OpCase) -> None:
    torch.library.opcheck(
        _op(case.name),
        case.args("meta"),
        test_utils=("test_schema",),
    )


@pytest.mark.parametrize("case", RETURNING_CASES, ids=lambda c: c.name)
def test_rwkv7_returning_ops_fake_meta_shapes(case: OpCase) -> None:
    args = case.args("meta")
    result = _op(case.name)(*args)
    outputs = result if isinstance(result, list | tuple) else [result]
    expected = case.expected(args)

    assert len(outputs) == len(expected)
    for output, (shape, dtype) in zip(outputs, expected, strict=True):
        assert tuple(output.shape) == shape
        assert output.dtype == dtype
        assert output.device.type == "meta"


@pytest.mark.parametrize("case", RETURNING_CASES, ids=lambda c: c.name)
def test_rwkv7_returning_ops_opcheck_faketensor(case: OpCase) -> None:
    torch.library.opcheck(
        _op(case.name),
        case.args("meta"),
        test_utils=("test_faketensor",),
    )


def test_rwkv7_advance_i32_schema_opcheck() -> None:
    torch.library.opcheck(
        torch.ops.rwkv7_v3a_ops.advance_i32,
        (_i32("cuda", (4,)), 3),
        test_utils=("test_schema",),
    )


def test_rwkv7_advance_i32_slots_schema_opcheck() -> None:
    torch.library.opcheck(
        torch.ops.rwkv7_v3a_ops.advance_i32_slots,
        (_i32("cuda", (6,)), torch.tensor([3, 0], device="cuda", dtype=torch.int32), 3),
        test_utils=("test_schema",),
    )


def test_rwkv7_advance_i32_varlen_schema_opcheck() -> None:
    torch.library.opcheck(
        torch.ops.rwkv7_v3a_ops.advance_i32_varlen,
        (
            _i32("cuda", (6,)),
            torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
            torch.tensor([3, 0], device="cuda", dtype=torch.int32),
        ),
        test_utils=("test_schema",),
    )


def test_rwkv7_advance_i32_varlen_updates_only_mapped_slots() -> None:
    elapsed = torch.tensor([10, 20, 30, 40], device="cuda", dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    slot_indices = torch.tensor([3, 0], device="cuda", dtype=torch.int32)

    torch.ops.rwkv7_v3a_ops.advance_i32_varlen(elapsed, query_start_loc, slot_indices)

    assert elapsed.cpu().tolist() == [13, 20, 30, 42]


@pytest.mark.parametrize(
    "op_name,args",
    [
        (
            "rwkv7_wkv_fp16_v2::wkv_seq_slot",
            lambda: (
                2,
                1,
                64,
                1,
                _h("cuda", (5, 1, 64, 64)),
                *[_h("cuda", (2, 1, 64)) for _ in range(6)],
                _h("cuda", (2, 1, 64)),
                torch.tensor([3, 0], device="cuda", dtype=torch.int32),
                _i32("cuda", (5,)),
            ),
        ),
        (
            "rwkv7_wkv_fp16_v2::wkv_seq_w0_slot",
            lambda: (
                2,
                1,
                64,
                1,
                _h("cuda", (5, 1, 64, 64)),
                _h("cuda", (2, 1, 64)),
                _h("cuda", (2, 1, 64)),
                _h("cuda", (64,)),
                *[_h("cuda", (2, 1, 64)) for _ in range(4)],
                _h("cuda", (2, 1, 64)),
                torch.tensor([3, 0], device="cuda", dtype=torch.int32),
                _i32("cuda", (5,)),
            ),
        ),
        (
            "rwkv7_wkv_fp16_v2::wkv_seq_varlen",
            lambda: (
                2,
                5,
                3,
                64,
                1,
                torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
                torch.tensor([3, 0], device="cuda", dtype=torch.int32),
                _h("cuda", (5, 1, 64, 64)),
                *[_h("cuda", (5, 64)) for _ in range(6)],
                _h("cuda", (5, 64)),
                _i32("cuda", (5,)),
            ),
        ),
        (
            "rwkv7_wkv_fp16_v2::wkv_seq_w0_varlen",
            lambda: (
                2,
                5,
                3,
                64,
                1,
                torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
                torch.tensor([3, 0], device="cuda", dtype=torch.int32),
                _h("cuda", (5, 1, 64, 64)),
                _h("cuda", (5, 64)),
                _h("cuda", (5, 64)),
                _h("cuda", (64,)),
                *[_h("cuda", (5, 64)) for _ in range(4)],
                _h("cuda", (5, 64)),
                _i32("cuda", (5,)),
            ),
        ),
        (
            "rwkv7_wkv_fp32_v2::forward_slot",
            lambda: (
                2,
                1,
                64,
                1,
                torch.empty((5, 1, 64, 64), device="cuda", dtype=torch.float32),
                *[_h("cuda", (2, 1, 64)) for _ in range(6)],
                _h("cuda", (2, 1, 64)),
                torch.tensor([3, 0], device="cuda", dtype=torch.int32),
            ),
        ),
        (
            "rwkv7_wkv_fp32_v2::forward_varlen",
            lambda: (
                2,
                5,
                3,
                64,
                1,
                torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
                torch.tensor([3, 0], device="cuda", dtype=torch.int32),
                torch.empty((5, 1, 64, 64), device="cuda", dtype=torch.float32),
                *[_h("cuda", (5, 64)) for _ in range(6)],
                _h("cuda", (5, 64)),
            ),
        ),
    ],
)
def test_rwkv7_wkv_slot_schema_opcheck(op_name, args) -> None:
    torch.library.opcheck(_op(op_name), args(), test_utils=("test_schema",))


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_add_layer_norm_cmix_mix_slot_matches_scattered_reference(
    hidden: int,
) -> None:
    torch.manual_seed(6)
    device = "cuda"
    batch, slots = 3, 6
    eps = 1e-5
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((batch, 1, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()
    compact_shift_state = initial_shift_state[slot_indices.long()].clone()

    ref_x_out, ref_mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
        x, residual, compact_shift_state, weight, bias, x_k, eps
    )

    x_out, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16_slots(
        x, residual, shift_state, weight, bias, x_k, slot_indices, eps
    )

    expected_shift_state = initial_shift_state.clone()
    expected_shift_state[slot_indices.long()] = compact_shift_state

    torch.testing.assert_close(x_out, ref_x_out, atol=0, rtol=0)
    torch.testing.assert_close(mixed, ref_mixed, atol=0, rtol=0)
    torch.testing.assert_close(shift_state, expected_shift_state, atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_add_layer_norm_tmix_mix6_slot_matches_scattered_reference(
    hidden: int,
) -> None:
    torch.manual_seed(7)
    device = "cuda"
    batch, slots = 3, 6
    eps = 1e-5
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((batch, 1, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)
    mix_weights = [
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    ]
    initial_shift_state = shift_state.clone()
    compact_shift_state = initial_shift_state[slot_indices.long()].clone()

    ref_outputs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
        x, residual, compact_shift_state, weight, bias, *mix_weights, eps
    )

    outputs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16_slots(
        x, residual, shift_state, weight, bias, *mix_weights, slot_indices, eps
    )

    expected_shift_state = initial_shift_state.clone()
    expected_shift_state[slot_indices.long()] = compact_shift_state

    for output, ref_output in zip(outputs, ref_outputs, strict=True):
        torch.testing.assert_close(output, ref_output, atol=0, rtol=0)
    torch.testing.assert_close(shift_state, expected_shift_state, atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_tmix_mix6_batched_shift_state_matches_reference(hidden: int) -> None:
    torch.manual_seed(hidden)
    device = "cuda"
    batch, seq_len = 4, 5
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((batch, hidden), device=device, dtype=torch.float16)
    x_r, x_w, x_k, x_v, x_a, x_g = (
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    )
    initial_shift_state = shift_state.clone()

    op = torch.ops.rwkv7_fast_ops_fp16.tmix_mix6
    args = (batch, seq_len, hidden, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g)
    outputs = op(*args)

    prev = torch.cat((initial_shift_state[:, None, :], x[:, :-1, :]), dim=1)
    delta = prev.float() - x.float()
    for output, mix_weight in zip(
        outputs,
        (x_r, x_w, x_k, x_v, x_a, x_g),
        strict=True,
    ):
        expected = (x.float() + delta * mix_weight.float()).to(torch.float16)
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(shift_state, x[:, -1, :], atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_tmix_mix6_slot_matches_scattered_reference(hidden: int) -> None:
    torch.manual_seed(hidden + 2)
    device = "cuda"
    batch, seq_len, slots = 3, 4, 6
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    x_r, x_w, x_k, x_v, x_a, x_g = (
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    )
    initial_shift_state = shift_state.clone()

    op = torch.ops.rwkv7_fast_ops_fp16.tmix_mix6_slot
    args = (
        batch,
        seq_len,
        hidden,
        x,
        shift_state,
        slot_indices,
        x_r,
        x_w,
        x_k,
        x_v,
        x_a,
        x_g,
    )
    outputs = op(*args)

    prev = torch.cat(
        (initial_shift_state[slot_indices.long(), None, :], x[:, :-1, :]), dim=1
    )
    delta = prev.float() - x.float()
    for output, mix_weight in zip(
        outputs,
        (x_r, x_w, x_k, x_v, x_a, x_g),
        strict=True,
    ):
        expected = (x.float() + delta * mix_weight.float()).to(torch.float16)
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    expected_state = initial_shift_state.clone()
    expected_state[slot_indices.long()] = x[:, -1, :]
    torch.testing.assert_close(shift_state, expected_state, atol=0, rtol=0)


def test_rwkv7_tmix_mix6_varlen_matches_scattered_reference() -> None:
    torch.manual_seed(4)
    device = "cuda"
    lengths = [2, 4, 1]
    batch, total_tokens, hidden, slots = len(lengths), sum(lengths), 64, 7
    query_start_loc = torch.tensor([0, 2, 6, 7], device=device, dtype=torch.int32)
    req_id = torch.tensor([0, 0, 1, 1, 1, 1, 2], device=device, dtype=torch.int32)
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((total_tokens, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    x_r, x_w, x_k, x_v, x_a, x_g = (
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    )
    initial_shift_state = shift_state.clone()

    outputs = torch.ops.rwkv7_fast_ops_fp16.tmix_mix6_varlen(
        batch,
        total_tokens,
        hidden,
        x,
        shift_state,
        slot_indices,
        x_r,
        x_w,
        x_k,
        x_v,
        x_a,
        x_g,
        query_start_loc,
        req_id,
    )

    prev_parts = []
    for local_req, length in enumerate(lengths):
        start = int(query_start_loc[local_req].item())
        end = int(query_start_loc[local_req + 1].item())
        slot = int(slot_indices[local_req].item())
        prev_parts.append(
            torch.cat((initial_shift_state[slot : slot + 1], x[start : end - 1]))
        )
        assert end - start == length
    prev = torch.cat(prev_parts, dim=0)
    delta = prev.float() - x.float()
    for output, mix_weight in zip(
        outputs,
        (x_r, x_w, x_k, x_v, x_a, x_g),
        strict=True,
    ):
        expected = (x.float() + delta * mix_weight.float()).to(torch.float16)
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    expected_state = initial_shift_state.clone()
    for local_req, slot in enumerate(slot_indices.long().tolist()):
        end = int(query_start_loc[local_req + 1].item())
        expected_state[slot] = x[end - 1]
    torch.testing.assert_close(shift_state, expected_state, atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_cmix_mix_batched_shift_state_matches_reference(hidden: int) -> None:
    torch.manual_seed(hidden + 1)
    device = "cuda"
    batch, seq_len = 4, 5
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((batch, hidden), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()

    op = torch.ops.rwkv7_fast_ops_fp16.cmix_mix
    args = (batch, seq_len, hidden, x, shift_state, x_k)
    output = op(*args)

    prev = torch.cat((initial_shift_state[:, None, :], x[:, :-1, :]), dim=1)
    expected = (x.float() + (prev.float() - x.float()) * x_k.float()).to(torch.float16)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(shift_state, x[:, -1, :], atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_cmix_mix_slot_matches_scattered_reference(hidden: int) -> None:
    torch.manual_seed(hidden + 3)
    device = "cuda"
    batch, seq_len, slots = 3, 4, 6
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()

    op = torch.ops.rwkv7_fast_ops_fp16.cmix_mix_slot
    args = (batch, seq_len, hidden, x, shift_state, slot_indices, x_k)
    output = op(*args)

    prev = torch.cat(
        (initial_shift_state[slot_indices.long(), None, :], x[:, :-1, :]), dim=1
    )
    expected = (x.float() + (prev.float() - x.float()) * x_k.float()).to(torch.float16)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    expected_state = initial_shift_state.clone()
    expected_state[slot_indices.long()] = x[:, -1, :]
    torch.testing.assert_close(shift_state, expected_state, atol=0, rtol=0)


def test_rwkv7_cmix_mix_varlen_matches_scattered_reference() -> None:
    torch.manual_seed(5)
    device = "cuda"
    lengths = [2, 4, 1]
    batch, total_tokens, hidden, slots = len(lengths), sum(lengths), 64, 7
    query_start_loc = torch.tensor([0, 2, 6, 7], device=device, dtype=torch.int32)
    req_id = torch.tensor([0, 0, 1, 1, 1, 1, 2], device=device, dtype=torch.int32)
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((total_tokens, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()

    output = torch.ops.rwkv7_fast_ops_fp16.cmix_mix_varlen(
        batch,
        total_tokens,
        hidden,
        x,
        shift_state,
        slot_indices,
        x_k,
        query_start_loc,
        req_id,
    )

    prev_parts = []
    for local_req, length in enumerate(lengths):
        start = int(query_start_loc[local_req].item())
        end = int(query_start_loc[local_req + 1].item())
        slot = int(slot_indices[local_req].item())
        prev_parts.append(
            torch.cat((initial_shift_state[slot : slot + 1], x[start : end - 1]))
        )
        assert end - start == length
    prev = torch.cat(prev_parts, dim=0)
    expected = (x.float() + (prev.float() - x.float()) * x_k.float()).to(torch.float16)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    expected_state = initial_shift_state.clone()
    for local_req, slot in enumerate(slot_indices.long().tolist()):
        end = int(query_start_loc[local_req + 1].item())
        expected_state[slot] = x[end - 1]
    torch.testing.assert_close(shift_state, expected_state, atol=0, rtol=0)
