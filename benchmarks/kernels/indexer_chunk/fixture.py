# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata


def make_inputs(rows, heads, width, compression, context, dtype, q_len, topk):
    device, dim, page = "cuda", 128, 64
    requests = math.ceil(rows / q_len)
    per_req_rows = [min(q_len, rows - i * q_len) for i in range(requests)]
    lengths = []
    for count in per_req_rows:
        lengths.extend((context - count + j + 1) // compression for j in range(count))
    if min(lengths) < 0 or max(lengths) > width:
        raise ValueError("Context/query lengths exceed the configured capacity")
    pages_per_req = math.ceil(max(lengths) / page)
    num_pages = requests * pages_per_req
    generator = torch.Generator(device=device).manual_seed(1741)
    if dtype == "mxfp4":
        cache = torch.empty((num_pages, page * 68), device=device, dtype=torch.uint8)
        cache[:, : page * 64] = torch.randint(
            0,
            256,
            (num_pages, page * 64),
            generator=generator,
            device=device,
            dtype=torch.uint8,
        )
        cache[:, page * 64 :] = 127
        cache = cache.view(num_pages, page, 68)
        cache = kv_cache_as_quant_view(cache, dim, True)
        q = torch.randint(
            0,
            256,
            (rows, 1, heads, 64),
            device=device,
            generator=generator,
            dtype=torch.uint8,
        ).view(torch.int8)
    else:
        values = torch.randn(
            (num_pages, page, dim),
            device=device,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
        cache = torch.empty((num_pages, page * 132), device=device, dtype=torch.uint8)
        cache[:, : page * 128] = values.view(num_pages, page * dim).view(torch.uint8)
        cache[:, page * 128 :] = 127
        cache = cache.view(num_pages, page, 1, 132)
        del values
        q = torch.randn(
            (rows, 1, heads, dim),
            device=device,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
    scales = torch.full((rows, 1, heads), 0x7F7F7F7F, device=device, dtype=torch.int32)
    weights = (torch.randn((rows, heads), device=device, generator=generator) * 0.1).to(
        torch.bfloat16
    )
    request_table = (
        torch.randperm(num_pages, device=device, generator=generator)
        .to(torch.int32)
        .view(requests, pages_per_req)
    )
    indices = torch.arange(requests, device=device, dtype=torch.int32)
    counts = torch.tensor(per_req_rows, device=device)
    indices = indices.repeat_interleave(counts)
    table = request_table.repeat_interleave(counts, dim=0).contiguous()
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)[:, None]
    schedule = get_paged_mqa_logits_metadata(
        seq_lens, page, get_num_sms(), indices=indices
    )
    return dict(
        q=q,
        qs=scales,
        weights=weights,
        kv=cache,
        ctx=seq_lens,
        bt=table,
        indices=indices,
        schedule=schedule,
        out=torch.full((rows, topk), -1, device=device, dtype=torch.int32),
        width=width,
        max_valid=max(lengths),
        valid_scores=sum(lengths),
    )
