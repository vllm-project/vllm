import torch
import triton
import triton.language as tl

import vllm
from vllm import _custom_ops as ops

from typing import Tuple
from functools import wraps

import torch
import functools


@triton.jit
def moe_gather(
    a_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    M,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_cm,
    stride_ck,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    topk: tl.constexpr,
    splitk: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid // splitk
    pid_n = pid % splitk

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // topk * stride_am + offs_k[None, :] * stride_ak
    )

    c_ptrs = c_ptr + (offs_token_id[:, None] * stride_cm + offs_k[None, :] * stride_ck)
    w_token_mask = offs_token_id < num_tokens_post_padded

    SPLITED_K = tl.cdiv(K, BLOCK_SIZE_K) // splitk

    a_ptrs = a_ptrs + pid_n * SPLITED_K * BLOCK_SIZE_K * stride_ak
    c_ptrs = c_ptrs + pid_n * SPLITED_K * BLOCK_SIZE_K * stride_ck

    for k in range(pid_n * SPLITED_K, (pid_n + 1) * SPLITED_K):
        a_mask = token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        c_mask = w_token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        tl.store(c_ptrs, a, mask=c_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        c_ptrs += BLOCK_SIZE_K * stride_ck


@triton.jit
def moe_scatter(
    a_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    topk_weights_ptr,
    M,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_cm,
    stride_ck,
    # Meta-parameters
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    topk: tl.constexpr,
    splitk: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid // splitk
    pid_n = pid % splitk

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = a_ptr + (offs_token_id[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_token_mask = offs_token_id < num_tokens_post_padded

    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    w_token_mask = offs_token < num_valid_tokens

    c_ptrs = c_ptr + (offs_token[:, None] * stride_cm + offs_k[None, :] * stride_ck)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=w_token_mask, other=0)

    SPLITED_K = tl.cdiv(K, BLOCK_SIZE_K) // splitk
    a_ptrs = a_ptrs + pid_n * SPLITED_K * BLOCK_SIZE_K * stride_ak
    c_ptrs = c_ptrs + pid_n * SPLITED_K * BLOCK_SIZE_K * stride_ck

    for k in range(pid_n * SPLITED_K, (pid_n + 1) * SPLITED_K):
        a_mask = a_token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        c_mask = w_token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        if MUL_ROUTED_WEIGHT:
            a = a * moe_weight[:, None]
        tl.store(c_ptrs, a, mask=c_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        c_ptrs += BLOCK_SIZE_K * stride_ck

def invoke_moe_gather(
    inp,
    outp,
    sorted_token_ids,
    num_tokens_post_padded,
    topk_ids,
    block_m,
    block_k,
    topk,
    splitk=1,
):
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], block_m) * splitk,)

    moe_gather[grid](
        inp,
        outp,
        sorted_token_ids,
        num_tokens_post_padded,
        inp.size(0),
        inp.size(1),
        sorted_token_ids.size(0),
        topk_ids.numel(),
        inp.stride(0),
        inp.stride(1),
        outp.stride(0),
        outp.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
        topk=topk,
        splitk=splitk,
    )


def invoke_moe_scatter(
    inp,
    outp,
    sorted_token_ids,
    num_tokens_post_padded,
    topk_ids,
    block_m,
    block_k,
    topk,
    splitk=1,
    topk_weights=None,
):
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], block_m) * splitk,)

    moe_scatter[grid](
        inp,
        outp,
        sorted_token_ids,
        num_tokens_post_padded,
        topk_weights,
        inp.size(0),
        inp.size(1),
        sorted_token_ids.size(0),
        topk_ids.numel(),
        inp.stride(0),
        inp.stride(1),
        outp.stride(0),
        outp.stride(1),
        topk_weights is not None,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
        topk=topk,
        splitk=splitk,
    )
