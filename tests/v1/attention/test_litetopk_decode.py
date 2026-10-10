# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual sparse indexer route, including candidate publication."""

import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm.model_executor.layers.litetopk_decode import (
    get_litetopk_bf16_metadata,
    get_litetopk_workspace,
    has_litetopk_decode,
    litetopk_bf16_scores,
)
from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="requires SM100"
)


@pytest.mark.parametrize(
    "fp4,n,candidate,backend,enabled,expected_lite",
    [
        (False, 4, None, "auto", True, True),
        (False, 5, None, "auto", True, False),
        (True, 1, "write", "auto", True, True),
        (True, 5, "write", "auto", True, True),
        (True, 6, None, "auto", True, True),
        (True, 6, "mask", "auto", True, False),
        (True, 6, None, "torch", True, False),
        (True, 6, None, "auto", False, False),
        (True, 7, None, "auto", True, False),
    ],
)
@pytest.mark.parametrize("reuse_schedule", [False, True])
@torch.inference_mode()
def test_decode_route_and_live_graph(
    fp4, n, candidate, backend, enabled, expected_lite, reuse_schedule
):
    if not has_litetopk_decode():
        pytest.skip("requires the native op and companion DeepGEMM histogram patch")
    torch.manual_seed(37)
    rows, width, requests = 3 * n, 8192, 3
    page, head_bytes, k = (128, 64, 512) if fp4 else (64, 128, 2048)
    pages = width // page
    if fp4:
        q = torch.randint(
            0, 256, (rows, 32, head_bytes), device="cuda", dtype=torch.uint8
        )
        sf = torch.full((rows, 32), 0x7D7D7D7D, device="cuda", dtype=torch.int32)
        cache = torch.randint(
            0,
            256,
            (requests * pages, page * (head_bytes + 4)),
            device="cuda",
            dtype=torch.uint8,
        )
        cache[:, page * head_bytes :] = 125
    else:
        q = torch.randn((rows, 32, head_bytes), device="cuda").to(torch.float8_e4m3fn)
        sf = None
        cache = torch.empty(
            (requests * pages, page * (head_bytes + 4)),
            device="cuda",
            dtype=torch.uint8,
        )
        cache[:, : page * head_bytes] = (
            torch.randn((requests * pages, page * head_bytes), device="cuda")
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        cache[:, page * head_bytes :] = torch.ones(
            (requests * pages, page), device="cuda"
        ).view(torch.uint8)
    cache = cache.view(-1, page, head_bytes + 4)
    weights = torch.randn((rows, 32), device="cuda") / 8
    table = torch.randperm(requests * pages, device="cuda").int().view(requests, pages)
    table = table.repeat_interleave(n, 0).contiguous()
    ids = torch.arange(rows, device="cuda", dtype=torch.int32) // n
    lengths = torch.full((rows, 1), width, device="cuda", dtype=torch.int32)
    schedule = get_paged_mqa_logits_metadata(lengths, page, get_num_sms(), indices=ids)
    decode = DeepSeekV32IndexerDecodeMetadata(
        table,
        lengths,
        torch.ones(rows, device="cuda", dtype=torch.int32),
        False,
        schedule,
        write_max_decode_len=n,
        indices=ids,
        litetopk_bf16_schedule=(
            get_litetopk_bf16_metadata(lengths, ids, n)
            if fp4 and expected_lite and reuse_schedule
            else None
        ),
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=lengths.flatten(),
        max_seq_len=width,
        slot_mapping=torch.zeros(rows, device="cuda", dtype=torch.int64),
        num_decodes=requests,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    output = torch.empty((rows + 3, k), device="cuda", dtype=torch.int32)
    candidates = None
    if candidate:
        candidates = (
            torch.arange(512, device="cuda", dtype=torch.int32).expand(rows, -1).clone()
        )
    hidden = torch.empty((rows, 1), device="cuda")
    cfg = VllmConfig()
    init_workspace_manager(torch.device("cuda"))

    def run():
        return sparse_attn_indexer(
            hidden,
            "indexer",
            cache,
            q,
            sf,
            None,
            weights,
            128,
            "ue8m0",
            k,
            128,
            width,
            width,
            output,
            True,
            False,
            False,
            "",
            use_fp4_cache=fp4,
            candidate_blocks=candidates,
            candidate_block_size=8 if candidate else 0,
            candidate_write=candidate == "write",
            topk_backend=backend,
            enable_litetopk_decode=enabled,
        )

    def reference():
        query = q.view(torch.int8) if fp4 else q
        query = (query.unsqueeze(1), sf.unsqueeze(1) if sf is not None else None)
        if expected_lite and fp4:
            hist = torch.zeros((rows, 1024), device="cuda", dtype=torch.int32)
            scores = litetopk_bf16_scores(
                query, cache.unsqueeze(2), weights, lengths, table, ids, width, n, hist
            )
        else:
            scores = fp8_fp4_paged_mqa_logits(
                query,
                cache.unsqueeze(2),
                weights,
                lengths,
                table,
                schedule,
                width,
                False,
                indices=ids,
            )
        if candidate == "mask":
            apply_candidate_mask(scores, None, lengths.flatten(), candidates, 8)
        return scores

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream), set_forward_context({"indexer": metadata}, cfg):
            run()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as prof:
                run()
            selected_lite = any(
                e.key == "_C::litetopk_decode" for e in prof.key_averages()
            )
            assert selected_lite == expected_lite
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                run()
            for live in (width - 1, 513, 0):
                lengths.copy_(
                    (live - n + 1 + torch.arange(rows, device="cuda").int() % n)
                    .clamp_min(0)
                    .view(-1, 1)
                )
                ids.copy_(
                    torch.arange(rows, device="cuda").int() // (n if live % 2 else 1)
                )
                schedule.copy_(
                    get_paged_mqa_logits_metadata(
                        lengths, page, get_num_sms(), indices=ids
                    )
                )
                if decode.litetopk_bf16_schedule is not None:
                    decode.litetopk_bf16_schedule.copy_(
                        get_litetopk_bf16_metadata(lengths, ids, n)
                    )
                graph.replay()
                scores = reference()
                for row, length in enumerate(lengths.flatten().tolist()):
                    take = min(k, length)
                    got = output[row, :take].long()
                    assert (got >= 0).all() and (got < length).all()
                    assert got.unique().numel() == take
                    assert (output[row, take:] == -1).all()
                    assert torch.equal(
                        scores[row, got].sort().values,
                        scores[row, :length].topk(take).values.sort().values,
                    )
                if candidate == "write":
                    expected = torch.empty_like(candidates)
                    select_candidate_blocks(
                        scores, None, lengths.flatten(), 512, 8, expected
                    )
                    assert torch.equal(candidates, expected)
                if expected_lite:
                    histogram, _ = get_litetopk_workspace(output.shape[0])
                    assert not histogram.any()
    finally:
        torch.cuda.current_stream().wait_stream(stream)
        reset_workspace_manager()
