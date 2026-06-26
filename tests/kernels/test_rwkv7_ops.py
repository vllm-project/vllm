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
    OpCase(
        "rwkv7_v3a_ops::layer_norm_f16_small",
        lambda d: _ln_args(d, 4096),
        _same(0),
    ),
    OpCase(
        "rwkv7_v3a_ops::layer_norm_f16_small512",
        lambda d: _ln_args(d, 4096),
        _same(0),
    ),
    OpCase("rwkv7_v3a_ops::linear_f16", _linear_args, _last_dim(0, 5)),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_orig",
        _linear_orig_args,
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_orig_rows_f16",
        lambda d: (*_linear_orig_args(d), 2, 2),
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_orig_rows_cfg_f16",
        lambda d: (*_linear_orig_args(d), 64, 2, 2),
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_orig_rows_exact_f16",
        lambda d: (*_linear_orig_args(d), 64, 2, True),
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_orig_wmma16_f16",
        _linear_orig_args,
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_orig_lt",
        _linear_orig_args,
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_orig_lt_cfg",
        lambda d: (*_linear_orig_args(d), 0, 0),
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase("rwkv7_v3a_ops::linear_f16_lt", _linear_args, _last_dim(0, 5)),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_m1_splitk",
        lambda d: (_h(d, (1, 8)), _h(d, (8, 5))),
        _last_dim(0, 5),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_m1_splitk_cfg",
        lambda d: (_h(d, (1, 8)), _h(d, (8, 5)), 64),
        _last_dim(0, 5),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_m1_splitk_tile",
        lambda d: (_h(d, (1, 8)), _h(d, (8, 5)), 64, 2),
        _last_dim(0, 5),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_m1_splitk_warpred_tile",
        lambda d: (_h(d, (1, 8)), _h(d, (8, 5)), 64, 2),
        _last_dim(0, 5),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_rows_splitk",
        lambda d: (*_linear_args(d), 64),
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
        "rwkv7_v3a_ops::add_layer_norm_tmix_mix6_f16_cfg",
        lambda d: (
            _h(d, (2, 1, 4096)),
            _h(d, (2, 1, 4096)),
            _h(d, (2, 4096)),
            _h(d, (4096,)),
            _h(d, (4096,)),
            *[_h(d, (4096,)) for _ in range(6)],
            1e-5,
            256,
        ),
        _same(0, 7),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_tmix_mix6_f16_scalar_stats",
        lambda d: (
            _h(d, (2, 1, 4096)),
            _h(d, (2, 1, 4096)),
            _h(d, (2, 4096)),
            _h(d, (4096,)),
            _h(d, (4096,)),
            *[_h(d, (4096,)) for _ in range(6)],
        ),
        _same(0, 7),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_cmix_mix_f16_cfg",
        lambda d: (
            _h(d, (2, 1, 4096)),
            _h(d, (2, 1, 4096)),
            _h(d, (2, 4096)),
            _h(d, (4096,)),
            _h(d, (4096,)),
            _h(d, (4096,)),
            1e-5,
            256,
        ),
        _same(0, 2),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_cmix_mix_f16_scalar_stats",
        lambda d: (
            _h(d, (2, 1, 4096)),
            _h(d, (2, 1, 4096)),
            _h(d, (2, 4096)),
            _h(d, (4096,)),
            _h(d, (4096,)),
            _h(d, (4096,)),
        ),
        _same(0, 2),
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
        "rwkv7_fast_ops_fp16::tmix_mix6_cfg",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (2, 8)),
            *[_h(d, (8,)) for _ in range(6)],
            256,
        ),
        _same(3, 6),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::tmix_mix6_t1_c4096",
        lambda d: (
            2,
            _h(d, (2, 1, 4096)),
            _h(d, (2, 4096)),
            *[_h(d, (4096,)) for _ in range(6)],
            256,
            1,
            False,
        ),
        _same(1, 6),
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
        "rwkv7_fast_ops_fp16::tmix_kk_a_gate_update_shift",
        lambda d: (
            2,
            1,
            128,
            2,
            _h(d, (2, 1, 128)),
            _h(d, (128,)),
            _h(d, (128,)),
            _h(d, (2, 1, 128)),
            _h(d, (128,)),
            _h(d, (2, 1, 128)),
            _h(d, (2, 128)),
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
        "rwkv7_fast_ops_fp16::cmix_sparse_one",
        lambda d: (
            128,
            128,
            _h(d, (1, 1, 128)),
            _h(d, (1, 128)),
            _h(d, (128,)),
            _h(d, (128, 128)),
            _h(d, (128, 128)),
        ),
        _shape_expected((1, 1, 128), 2),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_sparse_rows",
        lambda d: (
            2,
            3,
            128,
            128,
            _h(d, (2, 3, 128)),
            _h(d, (2, 128)),
            _h(d, (128,)),
            _h(d, (128, 128)),
            _h(d, (128, 128)),
        ),
        _shape_expected((2, 3, 128), 4),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_sparse_down_one",
        lambda d: (128, 128, _h(d, (128,)), _h(d, (128, 128))),
        _shape_expected((128,), 2),
    ),
    OpCase(
        "rwkv7_fast_ops_fp16::cmix_sparse_down_rows",
        lambda d: (2, 3, 128, 128, _h(d, (2, 3, 128)), _h(d, (128, 128))),
        _shape_expected((2, 3, 128), 4),
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
        "rwkv7_fast_ops_fp16::cmix_mix_cfg",
        lambda d: (2, 3, 8, _h(d, (2, 3, 8)), _h(d, (2, 8)), _h(d, (8,)), 256),
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

VOID_SCHEMA_OPS = {
    "rwkv7_v3a_ops": ["advance_i32"],
    "rwkv7_wkv_fp16_v2": ["wkv_seq", "wkv_seq_w0", "wkv_one", "wkv_one_w0"],
    "rwkv7_wkv_fp32_v2": ["forward", "forward_seq", "forward_small", "forward_block"],
}


@pytest.fixture(scope="module", autouse=True)
def rwkv7_ops_registered() -> None:
    _rwkv7_import_or_skip()


@pytest.mark.parametrize("case", RETURNING_CASES, ids=lambda c: c.name)
def test_rwkv7_returning_ops_have_registered_schema(case: OpCase) -> None:
    schema = _op(case.name)._schemas[""]
    assert "-> Tensor" in str(schema)


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


def test_rwkv7_void_ops_have_schema_evidence() -> None:
    for namespace, op_names in VOID_SCHEMA_OPS.items():
        ops = getattr(torch.ops, namespace)
        for op_name in op_names:
            assert getattr(ops, op_name)._schemas[""]


def test_rwkv7_advance_i32_schema_opcheck() -> None:
    torch.library.opcheck(
        torch.ops.rwkv7_v3a_ops.advance_i32,
        (_i32("cuda", (4,)), 3),
        test_utils=("test_schema",),
    )


@pytest.mark.parametrize("use_cfg", [False, True])
def test_rwkv7_tmix_mix6_batched_shift_state_matches_reference(
    use_cfg: bool,
) -> None:
    torch.manual_seed(0)
    device = "cuda"
    batch, seq_len, hidden = 4, 5, 64
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((batch, hidden), device=device, dtype=torch.float16)
    x_r, x_w, x_k, x_v, x_a, x_g = (
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    )
    initial_shift_state = shift_state.clone()

    op = (
        torch.ops.rwkv7_fast_ops_fp16.tmix_mix6_cfg
        if use_cfg
        else torch.ops.rwkv7_fast_ops_fp16.tmix_mix6
    )
    args = (batch, seq_len, hidden, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g)
    outputs = op(*args, 128) if use_cfg else op(*args)

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


@pytest.mark.parametrize("use_cfg", [False, True])
def test_rwkv7_cmix_mix_batched_shift_state_matches_reference(
    use_cfg: bool,
) -> None:
    torch.manual_seed(1)
    device = "cuda"
    batch, seq_len, hidden = 4, 5, 64
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((batch, hidden), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()

    op = (
        torch.ops.rwkv7_fast_ops_fp16.cmix_mix_cfg
        if use_cfg
        else torch.ops.rwkv7_fast_ops_fp16.cmix_mix
    )
    args = (batch, seq_len, hidden, x, shift_state, x_k)
    output = op(*args, 128) if use_cfg else op(*args)

    prev = torch.cat((initial_shift_state[:, None, :], x[:, :-1, :]), dim=1)
    expected = (x.float() + (prev.float() - x.float()) * x_k.float()).to(torch.float16)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(shift_state, x[:, -1, :], atol=0, rtol=0)
