# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fake and meta registrations for RWKV7 custom operators."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


def _register_fake_if_exists(name):
    namespace, op_name = name.split("::", 1)
    if hasattr(getattr(torch.ops, namespace), op_name):
        return register_fake(name)
    return lambda fn: fn


def _rwkv7_empty_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _rwkv7_empty_like_last_dim(x: torch.Tensor, last_dim: int) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], last_dim), device=x.device, dtype=x.dtype)


def _rwkv7_linear_out(x: torch.Tensor, out_features: int) -> torch.Tensor:
    return _rwkv7_empty_like_last_dim(x, out_features)


@_register_fake_if_exists("rwkv7_v3a_ops::layer_norm_f16")
def _rwkv7_layer_norm_f16_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return _rwkv7_empty_like(x)


@_register_fake_if_exists("rwkv7_v3a_ops::emb_ln0_bf16_to_f16")
def _rwkv7_emb_ln0_bf16_to_f16_fake(
    emb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.empty(emb.shape, device=emb.device, dtype=torch.float16)


@_register_fake_if_exists("rwkv7_v3a_ops::linear_f16")
@_register_fake_if_exists("rwkv7_v3a_ops::linear_f16_m1_splitk")
def _rwkv7_linear_f16_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    allow_fp16_accumulation: bool = False,
) -> torch.Tensor:
    return _rwkv7_linear_out(x, weight.shape[1])


@_register_fake_if_exists("rwkv7_v3a_ops::linear_t_f16")
def _rwkv7_linear_t_f16_fake(
    x: torch.Tensor,
    weight_t: torch.Tensor,
) -> torch.Tensor:
    return _rwkv7_linear_out(x, weight_t.shape[0])


@_register_fake_if_exists("rwkv7_v3a_ops::linear_t_act_f16")
def _rwkv7_linear_t_act_f16_fake(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    act: int,
) -> torch.Tensor:
    return _rwkv7_linear_out(x, weight_t.shape[0])


@_register_fake_if_exists("rwkv7_v3a_ops::linear_t_vres_f16")
def _rwkv7_linear_t_vres_f16_fake(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
) -> torch.Tensor:
    return _rwkv7_linear_out(x, weight_t.shape[0])


@_register_fake_if_exists("rwkv7_v3a_ops::linear_wag_rank_in_f16")
def _rwkv7_linear_wag_rank_in_f16_fake(
    xw: torch.Tensor,
    xa: torch.Tensor,
    xg: torch.Tensor,
    w1_t: torch.Tensor,
    a1_t: torch.Tensor,
    g1_t: torch.Tensor,
) -> list[torch.Tensor]:
    return [
        _rwkv7_linear_out(xw, w1_t.shape[0]),
        _rwkv7_linear_out(xa, a1_t.shape[0]),
        _rwkv7_linear_out(xg, g1_t.shape[0]),
    ]


@_register_fake_if_exists("rwkv7_v3a_ops::linear_wagv_rank_in_f16")
def _rwkv7_linear_wagv_rank_in_f16_fake(
    xw: torch.Tensor,
    xa: torch.Tensor,
    xg: torch.Tensor,
    xv: torch.Tensor,
    w1_t: torch.Tensor,
    a1_t: torch.Tensor,
    g1_t: torch.Tensor,
    v1_t: torch.Tensor,
) -> list[torch.Tensor]:
    return [
        _rwkv7_linear_out(xw, w1_t.shape[0]),
        _rwkv7_linear_out(xa, a1_t.shape[0]),
        _rwkv7_linear_out(xg, g1_t.shape[0]),
        _rwkv7_linear_out(xv, v1_t.shape[0]),
    ]


@_register_fake_if_exists("rwkv7_v3a_ops::linear_wag_rank_out_f16")
def _rwkv7_linear_wag_rank_out_f16_fake(
    w1: torch.Tensor,
    a1: torch.Tensor,
    g1: torch.Tensor,
    w2_t: torch.Tensor,
    a2_t: torch.Tensor,
    g2_t: torch.Tensor,
) -> list[torch.Tensor]:
    return [
        _rwkv7_linear_out(w1, w2_t.shape[0]),
        _rwkv7_linear_out(a1, a2_t.shape[0]),
        _rwkv7_linear_out(g1, g2_t.shape[0]),
    ]


@_register_fake_if_exists("rwkv7_v3a_ops::linear_wagv_rank_out_f16")
def _rwkv7_linear_wagv_rank_out_f16_fake(
    w1: torch.Tensor,
    a1: torch.Tensor,
    g1: torch.Tensor,
    v1: torch.Tensor,
    w2_t: torch.Tensor,
    a2_t: torch.Tensor,
    g2_t: torch.Tensor,
    v2_t: torch.Tensor,
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
) -> list[torch.Tensor]:
    return [
        _rwkv7_linear_out(w1, w2_t.shape[0]),
        _rwkv7_linear_out(a1, a2_t.shape[0]),
        _rwkv7_linear_out(g1, g2_t.shape[0]),
        _rwkv7_linear_out(v1, v2_t.shape[0]),
    ]


@_register_fake_if_exists("rwkv7_v3a_ops::add_f16")
def _rwkv7_add_f16_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _rwkv7_empty_like(x)


@_register_fake_if_exists("rwkv7_v3a_ops::add_layer_norm_f16")
def _rwkv7_add_layer_norm_f16_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(x), _rwkv7_empty_like(x)]


@_register_fake_if_exists("rwkv7_v3a_ops::add_last_layer_norm_f16")
def _rwkv7_add_last_layer_norm_f16_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)


@_register_fake_if_exists("rwkv7_v3a_ops::add_layer_norm_cmix_mix_f16")
def _rwkv7_add_layer_norm_cmix_mix_f16_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    shift_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_k: torch.Tensor,
    eps: float = 1e-5,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(x), _rwkv7_empty_like(x)]


@_register_fake_if_exists("rwkv7_v3a_ops::add_layer_norm_cmix_mix_f16_slots")
def _rwkv7_add_layer_norm_cmix_mix_f16_slots_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    shift_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_k: torch.Tensor,
    slot_indices: torch.Tensor,
    eps: float = 1e-5,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(x), _rwkv7_empty_like(x)]


@_register_fake_if_exists("rwkv7_v3a_ops::add_layer_norm_tmix_mix6_f16")
def _rwkv7_add_layer_norm_tmix_mix6_f16_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    shift_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    eps: float = 1e-5,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(x) for _ in range(7)]


@_register_fake_if_exists("rwkv7_v3a_ops::add_layer_norm_tmix_mix6_f16_slots")
def _rwkv7_add_layer_norm_tmix_mix6_f16_slots_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    shift_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    slot_indices: torch.Tensor,
    eps: float = 1e-5,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(x) for _ in range(7)]


@_register_fake_if_exists("rwkv7_fast_ops_fp16::tmix_mix6")
def _rwkv7_tmix_mix6_fake(
    B: int,
    T: int,
    C: int,
    x: torch.Tensor,
    shift_state: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(x) for _ in range(6)]


@_register_fake_if_exists("rwkv7_fast_ops_fp16::tmix_mix6_slot")
def _rwkv7_tmix_mix6_slot_fake(
    B: int,
    T: int,
    C: int,
    x: torch.Tensor,
    shift_state: torch.Tensor,
    slot_indices: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(x) for _ in range(6)]


@_register_fake_if_exists("rwkv7_fast_ops_fp16::tmix_mix6_varlen")
def _rwkv7_tmix_mix6_varlen_fake(
    B: int,
    total_tokens: int,
    C: int,
    x: torch.Tensor,
    shift_state: torch.Tensor,
    slot_indices: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id: torch.Tensor,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(x) for _ in range(6)]


@_register_fake_if_exists("rwkv7_fast_ops_fp16::tmix_kk_a_gate")
def _rwkv7_tmix_kk_a_gate_fake(
    B: int,
    T: int,
    C: int,
    H: int,
    k: torch.Tensor,
    k_k: torch.Tensor,
    a0: torch.Tensor,
    a12: torch.Tensor,
    k_a: torch.Tensor,
) -> list[torch.Tensor]:
    return [_rwkv7_empty_like(k) for _ in range(3)]


@_register_fake_if_exists("rwkv7_fast_ops_fp16::tmix_lnx_rkvres_xg")
def _rwkv7_tmix_lnx_rkvres_xg_fake(
    B: int,
    T: int,
    C: int,
    H: int,
    x: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    return _rwkv7_empty_like(x)


@_register_fake_if_exists("rwkv7_fast_ops_fp16::tmix_vres_gate")
def _rwkv7_tmix_vres_gate_fake(
    B: int,
    T: int,
    C: int,
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> torch.Tensor:
    return _rwkv7_empty_like(v)


@_register_fake_if_exists("rwkv7_fast_ops_fp16::cmix_sparse_down_relu_one")
def _rwkv7_cmix_sparse_down_relu_one_fake(
    C: int,
    F: int,
    preact: torch.Tensor,
    value_fc: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((C,), device=preact.device, dtype=preact.dtype)


@_register_fake_if_exists("rwkv7_fast_ops_fp16::cmix_sparse_down_relu_rows")
@_register_fake_if_exists("rwkv7_fast_ops_fp16::cmix_sparse_down_relu_rows_t512")
def _rwkv7_cmix_sparse_down_relu_rows_fake(
    B: int,
    T: int,
    C: int,
    F: int,
    preact: torch.Tensor,
    value_fc: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((B, T, C), device=preact.device, dtype=preact.dtype)


@_register_fake_if_exists("rwkv7_fast_ops_fp16::cmix_mix")
def _rwkv7_cmix_mix_fake(
    B: int,
    T: int,
    C: int,
    x: torch.Tensor,
    shift_state: torch.Tensor,
    x_k: torch.Tensor,
) -> torch.Tensor:
    return _rwkv7_empty_like(x)


@_register_fake_if_exists("rwkv7_fast_ops_fp16::cmix_mix_slot")
def _rwkv7_cmix_mix_slot_fake(
    B: int,
    T: int,
    C: int,
    x: torch.Tensor,
    shift_state: torch.Tensor,
    slot_indices: torch.Tensor,
    x_k: torch.Tensor,
) -> torch.Tensor:
    return _rwkv7_empty_like(x)


@_register_fake_if_exists("rwkv7_fast_ops_fp16::cmix_mix_varlen")
def _rwkv7_cmix_mix_varlen_fake(
    B: int,
    total_tokens: int,
    C: int,
    x: torch.Tensor,
    shift_state: torch.Tensor,
    slot_indices: torch.Tensor,
    x_k: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_id: torch.Tensor,
) -> torch.Tensor:
    return _rwkv7_empty_like(x)


@_register_fake_if_exists("rwkv7_fast_ops_fp16::relu_square")
@_register_fake_if_exists("rwkv7_fast_ops_fp16::act_tanh")
@_register_fake_if_exists("rwkv7_fast_ops_fp16::act_sigmoid")
def _rwkv7_unary_fake(x: torch.Tensor) -> torch.Tensor:
    return _rwkv7_empty_like(x)


@_register_fake_if_exists("rwkv7_fast_ops_fp16::add_vec")
def _rwkv7_add_vec_fake(C: int, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _rwkv7_empty_like(x)


@_register_fake_if_exists("rwkv7_v3a_ops::advance_i32")
@_register_fake_if_exists("rwkv7_v3a_ops::advance_i32_slots")
@_register_fake_if_exists("rwkv7_v3a_ops::advance_i32_varlen")
def _rwkv7_advance_i32_fake(*args) -> None:
    return None


@_register_fake_if_exists("rwkv7_wkv_fp16_v2::wkv_seq")
@_register_fake_if_exists("rwkv7_wkv_fp16_v2::wkv_seq_slot")
@_register_fake_if_exists("rwkv7_wkv_fp16_v2::wkv_seq_w0")
@_register_fake_if_exists("rwkv7_wkv_fp16_v2::wkv_seq_w0_slot")
@_register_fake_if_exists("rwkv7_wkv_fp16_v2::wkv_seq_varlen")
@_register_fake_if_exists("rwkv7_wkv_fp16_v2::wkv_seq_w0_varlen")
def _rwkv7_wkv_fp16_fake(*args) -> None:
    return None


@_register_fake_if_exists("rwkv7_wkv_fp32_v2::forward")
@_register_fake_if_exists("rwkv7_wkv_fp32_v2::forward_slot")
@_register_fake_if_exists("rwkv7_wkv_fp32_v2::forward_varlen")
def _rwkv7_wkv_fp32_fake(*args) -> None:
    return None
