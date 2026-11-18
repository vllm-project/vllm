# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expand packed prefixes once for the standard FP16 attention implementation."""

import ctypes as ct

import torch

from .hip import (
    DTYPES,
    _Activation,
    _activation,
    _cache_view,
    _check_buffer,
    _PagedCache,
    _vector,
)


class ExpandedPrefill:
    def __init__(self, native):
        self.native = native
        ptr, i32, i64, f32 = ct.c_void_p, ct.c_int, ct.c_int64, ct.c_float
        self.dequantize = native._bind(
            "dequantize_prefill",
            [
                ptr,
                ptr,
                _Activation,
                _Activation,
                _PagedCache,
                ptr,
                i64,
                ptr,
                ptr,
                i32,
                i32,
                ptr,
                i32,
                i32,
                ptr,
            ],
        )
        self.repair = native._bind(
            "prefill_repair",
            [
                ptr,
                _Activation,
                _Activation,
                _Activation,
                _PagedCache,
                ptr,
                i64,
                ptr,
                ptr,
                i32,
                ptr,
                i32,
                i32,
                f32,
                i32,
                i32,
                ptr,
            ],
        )

    def __call__(
        self,
        query,
        key,
        value,
        cache,
        table,
        starts,
        lengths,
        signs,
        cache_dtype,
        *,
        key_tmp,
        value_tmp,
        query_tmp,
        table_tmp,
        out,
        max_query_len,
        scale,
    ):
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention

        native = self.native
        device, dtype = query.device, query.dtype
        layout = native.layout(cache_dtype)
        q, k, v = (
            _activation(x, device, dtype, native.head_dim) for x in (query, key, value)
        )
        pages = _cache_view(cache, device, layout)
        requests, blocks = table_tmp.shape
        capacity = blocks * 128
        expected_cache = (requests * blocks, 128, pages.heads, native.head_dim)
        if (
            table.ndim != 2
            or table.shape[0] != requests
            or table.stride(1) != 1
            or table.dtype != torch.int32
            or table.device != device
            or table_tmp.dtype != torch.int32
            or not table_tmp.is_contiguous()
            or table_tmp.device != device
            or key.shape != value.shape
            or key.shape[0] != query.shape[0]
            or key.shape[1] != pages.heads
            or query.shape[1] % pages.heads
            or capacity <= 0
            or max_query_len <= 0
        ):
            raise ValueError("Invalid expanded-prefill metadata")
        for tensor, shape in (
            (key_tmp, expected_cache),
            (value_tmp, expected_cache),
            (query_tmp, tuple(query.shape)),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.float16
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError("Expected contiguous FP16 expansion buffers")
        output_ptr = _check_buffer(out, query.shape, device)
        starts_ptr = _vector(starts, requests + 1, torch.int32, device)
        lengths_ptr = _vector(lengths, requests, torch.int32, device)
        signs_ptr = _vector(signs, native.head_dim // 32, torch.int32, device)
        torch.arange(table_tmp.numel(), out=table_tmp.view(-1))
        native._launch(
            self.dequantize,
            device,
            key_tmp.data_ptr(),
            value_tmp.data_ptr(),
            k,
            v,
            pages,
            table.data_ptr(),
            table.stride(0),
            starts_ptr,
            lengths_ptr,
            requests,
            capacity,
            signs_ptr,
            DTYPES[dtype],
            layout.kernel_code,
        )
        query_tmp.copy_(query)
        unified_attention(
            q=query_tmp,
            k=key_tmp,
            v=value_tmp,
            out=out,
            cu_seqlens_q=starts,
            max_seqlen_q=max_query_len,
            seqused_k=lengths,
            max_seqlen_k=capacity,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=table_tmp,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )
        native._launch(
            self.repair,
            device,
            output_ptr,
            q,
            k,
            v,
            pages,
            table.data_ptr(),
            table.stride(0),
            starts_ptr,
            lengths_ptr,
            requests,
            signs_ptr,
            query.shape[0],
            query.shape[1],
            scale,
            DTYPES[dtype],
            layout.kernel_code,
        )
        stream = torch.cuda.current_stream(device)
        for tensor in (
            query,
            key,
            value,
            cache,
            table,
            starts,
            lengths,
            signs,
            key_tmp,
            value_tmp,
            query_tmp,
            table_tmp,
            out,
        ):
            tensor.record_stream(stream)
        return out
