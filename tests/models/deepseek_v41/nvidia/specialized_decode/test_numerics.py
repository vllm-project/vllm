# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank numerics of the DeepSeek-V4.1 specialized decode at TP4.

Each check runs one rank late, so a missing wait reads a peer's stale data:

* the fused all-reduce + mHC at 1, 6, 48 and 384 rows (its whole capacity),
  plain, with the MXFP8 layer input, with a deferred MoE finalize and
  residual-only, bitwise against the separate all-reduce and mHC kernels;
* the executor's sublayer boundary on it, with the next mHC coefficients on
  the side stream and inline, against the separate kernels and the stock mHC
  pre;
* the ``wo_b`` GEMM that publishes into the fused kernel's mailbox, per
  split-K tactic, bitwise against FlashInfer's GEMM plus the separate publish;
* logits gathered with the custom all-gather, bitwise against NCCL;
* the row-sharded decode indexer against the replicated one (equal top-k
  sets up to exactly tied scores);
* one CUDA graph of all of these, replayed ``REPLAYS`` times bitwise.

Needs four SM100-family GPUs with NVLink multicast and flashinfer-python
0.7.2; about 5 minutes. For a race stress run, run this file as a module
(``python -m tests.models.deepseek_v41.nvidia.specialized_decode.test_numerics``)
with ``--replays 100000``.
"""

import argparse
import importlib.metadata
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

TP = 4
HIDDEN, HC, TOP_K = 5120, 4, 6
CAPACITY = 384
REPLAYS = 300
LATE_CYCLES = 20_000


def _late(rank: int, step: int) -> None:
    """Delay one rank per step, a different one each step."""
    if rank == step % TP:
        torch.cuda._sleep(LATE_CYCLES)


def _randn(generator, *shape, dtype=torch.bfloat16):
    return torch.randn(*shape, generator=generator, device="cuda").to(dtype)


def _mix(rows: int):
    """The replicated mHC inputs of one boundary, equal on every rank."""
    generator = torch.Generator(device="cuda").manual_seed(rows)
    return dict(
        residual=_randn(generator, rows, HC, HIDDEN),
        post=torch.rand(rows, HC, generator=generator, device="cuda"),
        comb=_randn(generator, rows, HC, HC, dtype=torch.float32) * 0.1,
        pre=torch.rand(rows, HC, generator=generator, device="cuda"),
        norm_weight=_randn(generator, HIDDEN),
        norm_eps=1e-6,
    )


def _partial(rows: int, rank: int, seed: int = 0) -> torch.Tensor:
    """This rank's partial sum, with +0/-0 pairs that look like the Lamport
    sentinel."""
    x = _randn(
        torch.Generator(device="cuda").manual_seed(seed + 97 * rank), rows, HIDDEN
    )
    x[:, :16] = 0
    x[:, 9:16:2] = -0.0
    return x


def _moe_partial(rows: int, rank: int):
    """An unfinalized MoE output (padded permuted GEMM2 rows, routing with a
    route to an expert this rank does not hold) and its shared output."""
    generator = torch.Generator(device="cuda").manual_seed(7 + rank)
    num_rows = rows * TOP_K + 5
    gemm2 = _randn(generator, num_rows, HIDDEN)
    permuted = torch.randperm(num_rows, generator=generator, device="cuda")
    permuted = permuted[: rows * TOP_K].view(rows, TOP_K).int()
    permuted[0, -1] = -1
    weights = torch.rand(rows, TOP_K, generator=generator, device="cuda")
    return gemm2, weights, permuted, _randn(generator, rows, HIDDEN)


def _fma_finalize(gemm2, weights, permuted, shared) -> torch.Tensor:
    """One FP32 FMA per route in route order, the shared add, one rounding."""
    acc = torch.zeros(shared.shape, device="cuda")
    for k in range(TOP_K):
        valid = (permuted[:, k] >= 0).unsqueeze(-1)
        rows = gemm2[permuted[:, k].clamp_min(0).long()].float()
        torch.addcmul(
            acc,
            torch.where(valid, rows, 0.0),
            torch.where(valid, weights[:, k : k + 1], 0.0),
            out=acc,
        )
    return (acc + shared.float()).bfloat16()


def _assert_equal(got, want, what):
    for a, b in zip(got, want):
        if a is None or b is None:
            assert a is None and b is None, what
        else:
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), what


# ------------------------------------------------- fused all-reduce + mHC


def _check_all_reduce_mhc(arm, rank):
    from tests.distributed.test_custom_all_reduce import (
        _bf16_ulps,
        _unfused_all_reduce_mhc,
    )
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.fused_moe.moe_output import (
        MoEOutput,
        UnfinalizedMoEOutput,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary import (
        Boundaries,
        BoundaryState,
        empty_quantized_layer_input,
    )

    boundaries = Boundaries(arm, torch.cuda.Stream())
    step = 0
    for rows in (1, 6, 48, 384):
        mix = _mix(rows)
        x = _partial(rows, rank)
        _late(rank, step := step + 1)
        streams, layer_input = arm(x, **mix)
        peers = get_tp_group().all_gather(x, dim=0).view(TP, rows, HIDDEN)
        reference = _unfused_all_reduce_mhc(peers, *mix.values())
        # The RMSNorm sums squares in another order, with an approximate rsqrt.
        assert torch.equal(streams, reference[0]), rows
        assert _bf16_ulps(layer_input, reference[1]) <= 1, rows

        # The layer input also as MXFP8, as fused_wqa_wkv would quantize it.
        quantized = empty_quantized_layer_input(rows, HIDDEN, x.device)
        _late(rank, step := step + 1)
        out = arm(x, **mix, mxfp8_out=(quantized.data, quantized.scale))
        _assert_equal(out, (streams, layer_input), f"MXFP8 output, {rows} rows")
        data, scale = mxfp8_e4m3_quantize(layer_input, is_sf_swizzled_layout=True)
        _assert_equal(
            (quantized.data, quantized.scale),
            (data, scale.reshape(-1)),
            f"MXFP8 layer input, {rows} rows",
        )

        _late(rank, step := step + 1)
        out = arm(x, mix["residual"], mix["post"], mix["comb"], None, None, 1e-6)
        _assert_equal(out, (streams, None), f"residual only, {rows} rows")
        # The DSpark draft's last layer: a plain partial, residual-only.
        state = BoundaryState(mix["residual"], mix["post"], mix["comb"], mix["pre"], x)
        _late(rank, step := step + 1)
        got = boundaries.post_only(state, fuse_partial=True)
        _assert_equal((got,), (streams,), f"residual-only partial, {rows} rows")

        # A deferred MoE finalize reduces as the plain call on the finalized
        # sum; also residual-only, at an Engram or the last layer.
        gemm2, weights, permuted, shared = _moe_partial(rows, rank)
        finalized = _fma_finalize(gemm2, weights, permuted, shared)
        _late(rank, step := step + 1)
        want = arm(finalized, **mix)
        _late(rank, step := step + 1)
        got = arm.finalize(gemm2, weights, permuted, shared, *mix.values())
        _assert_equal(got, want, f"finalize, {rows} rows")
        moe_output = MoEOutput(
            UnfinalizedMoEOutput(
                gemm2_permuted=gemm2,
                expert_weights=weights,
                expanded_idx_to_permuted_idx=permuted,
            ),
            shared_output=shared,
        )
        state = BoundaryState(
            mix["residual"], mix["post"], mix["comb"], mix["pre"], moe_output
        )
        _late(rank, step := step + 1)
        got = boundaries.post_only(state)
        _assert_equal((got,), want[:1], f"residual-only finalize, {rows} rows")


def _check_boundaries(arm, rank):
    """``Boundaries.sublayer`` on a partial sum and on a deferred MoE output,
    with the next coefficients on the side stream and inline: the streams
    bitwise against the separate all-reduce and mHC post, the mixes bitwise
    against the stock mHC pre on those streams, and the layer input within
    the fused RMSNorm's ulp of it."""
    from tests.distributed.test_custom_all_reduce import (
        _bf16_ulps,
        _unfused_all_reduce_mhc,
    )
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.fused_moe.moe_output import (
        MoEOutput,
        UnfinalizedMoEOutput,
    )
    from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_pre_delayed_overlap
    from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary import (
        Boundaries,
        BoundaryState,
    )

    generator = torch.Generator(device="cuda").manual_seed(3)  # equal on all ranks
    mix_size = HC * (HC + 2)
    fn = torch.randn(mix_size, HC * HIDDEN, generator=generator, device="cuda") * 0.02
    scale = torch.tensor([0.5, 0.25, 1.0], device="cuda")
    base = torch.randn(mix_size, generator=generator, device="cuda")
    side = torch.cuda.Stream()
    boundaries = Boundaries(arm, side)
    step = 0
    for rows in (1, 6, 48, 384):
        mix = _mix(rows)
        layer = SimpleNamespace(
            hc_attn_fn=fn,
            hc_attn_scale=scale,
            hc_attn_base=base,
            attn_norm=SimpleNamespace(weight=mix["norm_weight"], variance_epsilon=1e-6),
            rms_norm_eps=1e-6,
            hc_eps=1e-6,
            hc_post_alpha=2.0,
            hc_sinkhorn_iters=20,
        )
        gemm2, weights, permuted, shared = _moe_partial(rows, rank)
        deferred = MoEOutput(
            UnfinalizedMoEOutput(
                gemm2_permuted=gemm2,
                expert_weights=weights,
                expanded_idx_to_permuted_idx=permuted,
            ),
            shared_output=shared,
        )
        # (what the sublayer left, the partial sum it amounts to)
        cases = (
            ("partial", _partial(rows, rank), _partial(rows, rank)),
            ("deferred", deferred, _fma_finalize(gemm2, weights, permuted, shared)),
        )
        for name, pending, finalized in cases:
            peers = get_tp_group().all_gather(finalized, dim=0)
            streams, _ = _unfused_all_reduce_mhc(
                peers.view(TP, rows, HIDDEN), *mix.values()
            )
            want = mhc_pre_delayed_overlap(
                streams,
                fn,
                scale,
                base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                20,
                pre_mix=mix["pre"],
                norm_weight=mix["norm_weight"],
                norm_eps=1e-6,
                stream=side,
            )
            torch.cuda.current_stream().wait_stream(side)
            for side_stream in (True, False):
                what = f"{name}, {rows} rows, side stream {side_stream}"
                state = BoundaryState(
                    mix["residual"], mix["post"], mix["comb"], mix["pre"], pending
                )
                decode_step = SimpleNamespace(side_stream=side_stream)
                _late(rank, step := step + 1)
                state, layer_input, _, aux = boundaries.sublayer(
                    layer, False, state, decode_step, capture_aux=True
                )
                boundaries.join(decode_step)
                _assert_equal(
                    (state.residual, aux),
                    (streams, streams.mean(dim=1)),
                    f"streams, {what}",
                )
                _assert_equal(
                    (state.post, state.comb, state.pre),
                    (want[0], want[1], want[3]),
                    f"mixes, {what}",
                )
                assert _bf16_ulps(layer_input, want[2]) <= 1, what


# ------------------------------------------------------- publishing wo_b

WO_B_K = 2048  # o_lora_rank x local groups at TP4


def _wo_b(rank: int, index: int):
    """A ``wo_b`` linear with its weights processed for the low-latency
    MXFP8 kernel, different on each rank."""
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    generator = torch.Generator(device="cuda").manual_seed(1000 * rank + index)
    w = _randn(generator, HIDDEN, WO_B_K, dtype=torch.float32) / WO_B_K**0.5
    w_q, w_s = mxfp8_e4m3_quantize(w.bfloat16(), is_sf_swizzled_layout=False)
    linear = torch.nn.Module()
    linear.weight = torch.nn.Parameter(w_q, requires_grad=False)
    linear.weight_scale = torch.nn.Parameter(
        w_s.view(HIDDEN, WO_B_K // 32), requires_grad=False
    )
    kernel = FlashInferCutedslMxfp8LinearKernel(
        Mxfp8LinearLayerConfig((HIDDEN, WO_B_K))
    )
    kernel.process_weights_after_loading(linear)
    linear.quant_method = SimpleNamespace(kernel=kernel)
    return linear


def _attention_output(rows: int, rank: int):
    """``wo_b``'s MXFP8 input as the fused WO-A kernel emits it, with a zero
    row (zero outputs, whose negative zeros the publish clears)."""
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    x = _randn(torch.Generator(device="cuda").manual_seed(rank), rows, WO_B_K)
    x[-1] = 0
    data, scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    return QuantizedActivation(
        data, scale.reshape(-1), torch.bfloat16, x.shape, kMxfp8Dynamic
    )


def _flashinfer_wo_b(linear, qa, tactic):
    """FlashInfer's low-latency MXFP8 GEMM with ``tactic`` (None: the
    default), and the valid tactics."""
    from flashinfer.gemm import gemm_base

    out = torch.empty(qa.data.shape[0], HIDDEN, dtype=torch.bfloat16, device="cuda")
    major, minor = torch.cuda.get_device_capability()
    runner = gemm_base._cutedsl_low_latency_blockscaled_gemm_runner(
        major * 10 + minor, True
    )
    workspace = gemm_base._get_cache_buf(
        "mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, out.device
    )
    inputs = [
        qa.data,
        linear.weight,
        qa.scale,
        linear.weight_scale,
        torch.bfloat16,
        out,
        workspace,
    ]
    runner.forward(inputs, tactic=tactic)
    return out, [tuple(t) for t in runner.get_valid_tactics(inputs, None)]


# Per boundary of a sequence, whether wo_b publishes; it crosses the mailbox
# generations several times on both paths.
PUBLISH_PATTERN = (True, True, True, False, True, False, False, True, True)


def _check_publishing_wo_b(arm, publisher, rank):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary import (
        publishing_gemm,
    )

    linears = [_wo_b(rank, i) for i in range(2)]
    step = 0
    for rows in (1, 6, 8):
        mix = _mix(rows)
        qa = _attention_output(rows, rank)
        assert all(publisher.supports(linear, rows) for linear in linears)
        _, valid = _flashinfer_wo_b(linears[0], qa, None)
        # The default tactic and the first of each split-K (each epilogue path).
        by_split: dict[int, tuple] = {}
        for tactic in valid:
            by_split.setdefault(tactic[3], tactic)
        for tactic in dict.fromkeys([valid[0], *by_split.values()]):
            gemms = [_flashinfer_wo_b(linear, qa, tactic)[0] for linear in linears]
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(
                    publishing_gemm, "_selected_tactic", lambda *_, t=tactic: t
                )
                for index, published in enumerate(PUBLISH_PATTERN):
                    linear, gemm = linears[index % 2], gemms[index % 2]
                    what = f"{rows} rows, tactic {tactic}, boundary {index}"
                    _late(rank, step := step + 1)
                    if published:
                        x = publisher(linear, qa)
                        _assert_equal((x,), (gemm,), f"wo_b output, {what}")
                        got = arm(x, **mix)
                    else:
                        got = arm(gemm.clone(), **mix)
                    _assert_equal(got, arm(gemm.clone(), **mix), what)


# -------------------------------------------------------------- logits


def _check_logits_gather(rank):
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.models.deepseek_v41.nvidia.specialized_decode.logits.logits import (
        SpecializedDecodeLogitsProcessor,
    )

    vocab = 129280
    processor = SpecializedDecodeLogitsProcessor(vocab)
    generator = torch.Generator(device="cuda").manual_seed(rank)
    for rows in (1, 6, 16, 48):
        logits = _randn(generator, rows, vocab // TP)
        if rows <= 16:
            # Otherwise both sides would take NCCL.
            custom = get_tp_group().device_communicator.custom_all_gather(logits)
            assert custom is not None, f"no custom all-gather for {rows} rows"
        got = processor._gather_logits(logits)
        want = LogitsProcessor._gather_logits(processor, logits)
        _assert_equal((got,), (want,), f"logits, {rows} rows")


# ---------------------------------------------------- row-sharded indexer

CONTEXTS = (84_000, 163_000, 300_000, 360_000, 406_000, 490_000, 556_000, 648_000)
NEXT_N = 6  # rows per request: DSpark verification of five drafts
INDEX_HEADS, INDEX_HEAD_DIM, INDEX_TOPK = 32, 128, 512
CANDIDATE_BLOCKS, CANDIDATE_BLOCK_SIZE = 2048, 8
MAX_MODEL_LEN = 655_360
PAGE_TOKENS = 128
ROPE_POSITIONS = 2048
# (requests, of which trailing padding requests).
INDEX_BATCHES = ((8, 0), (32, 10))
# A dense index source (ratio 2) and the candidate source (ratio 1).
INDEX_LAYERS = ((2, False), (1, True))


def _index_keys(ratio: int, generator):
    """A paged MXFP4 indexer key cache holding every context once."""
    from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store

    row_bytes = INDEX_HEAD_DIM // 2 + INDEX_HEAD_DIM // 32
    page_states = PAGE_TOKENS // ratio
    pages = [-(-(ctx // ratio) // page_states) for ctx in CONTEXTS]
    page_bytes = -(-(page_states * row_bytes) // 512) * 512
    storage = torch.empty(sum(pages), page_bytes, dtype=torch.uint8, device="cuda")
    cache = storage.as_strided(
        (sum(pages), page_states, row_bytes), (page_bytes, row_bytes, 1)
    )
    angles = torch.rand(2 * ROPE_POSITIONS, 32, generator=generator, device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    norm_weight = torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    num_slots = sum(pages) * page_states
    for first in range(0, num_slots, 1 << 20):
        slots = torch.arange(first, min(first + (1 << 20), num_slots), device="cuda")
        k_pre = _randn(generator, slots.numel(), INDEX_HEAD_DIM)
        positions = (slots % ROPE_POSITIONS) * ratio + ratio - 1
        indexer_k_norm_rope_store(
            k_pre, positions, cos_sin, norm_weight, 1e-6, cache, slots, ratio, True
        )
    starts = [sum(pages[:i]) for i in range(len(pages))]
    return SimpleNamespace(
        cache=cache, page_states=page_states, pages=pages, starts=starts
    )


def _decode_step(requests, padding, ratio, keys, generator):
    """The indexer metadata and queries of a pure-decode step of ``NEXT_N``
    rows per request; padding requests have rows of length 0."""
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata
    from vllm.utils.platform_utils import num_compute_units
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    rows = requests * NEXT_N
    real = requests - padding
    lens, tables, indices = [], [], []
    for request in range(real):
        i = request % len(CONTEXTS)
        table = torch.zeros(max(keys.pages), dtype=torch.int32)
        table[: keys.pages[i]] = torch.arange(
            keys.starts[i], keys.starts[i] + keys.pages[i]
        )
        for j in range(NEXT_N):
            lens.append((CONTEXTS[i] - NEXT_N + j + 1) // ratio)
            tables.append(table)
            indices.append(request)
    for row in range(real * NEXT_N, rows):
        # Padding rows: no context, each its own index.
        lens.append(0)
        tables.append(torch.zeros(max(keys.pages), dtype=torch.int32))
        indices.append(requests + row - real * NEXT_N)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda").unsqueeze(1)
    indices = torch.tensor(indices, dtype=torch.int32, device="cuda")
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=torch.stack(tables).cuda(),
        seq_lens=seq_lens,
        decode_lens=torch.ones(rows, dtype=torch.int32, device="cuda"),
        requires_padding=False,
        schedule_metadata=get_paged_mqa_logits_metadata(
            seq_lens, keys.page_states, num_compute_units(), indices=indices
        ),
        indices=indices,
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.zeros(requests, dtype=torch.int32, device="cuda"),
        max_seq_len=max(CONTEXTS),
        slot_mapping=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        num_decodes=requests,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    angles = torch.linspace(0, 6.3, ROPE_POSITIONS * 32, device="cuda").view(-1, 32)
    (q, q_scale), weights = fused_indexer_q_rope_quant(
        torch.arange(rows, device="cuda") + 100,
        _randn(generator, rows, INDEX_HEADS, INDEX_HEAD_DIM),
        torch.cat((angles.cos(), angles.sin()), dim=-1),
        _randn(generator, rows, INDEX_HEADS),
        INDEX_HEAD_DIM**-0.5,
        INDEX_HEADS**-0.5,
        use_fp4=True,
        weights_out_dtype=torch.float32,
    )
    return SimpleNamespace(
        metadata=metadata, rows=rows, q=(q, q_scale), weights=weights
    )


def _indexer(keys, ratio, candidates):
    """The decode indexer of a layer, writing ``topk`` (and ``candidates``)."""
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    rows = max(r for r, _ in INDEX_BATCHES) * NEXT_N
    return SparseAttnIndexer(
        SimpleNamespace(prefix="indexer", kv_cache=keys.cache),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=INDEX_TOPK,
        head_dim=INDEX_HEAD_DIM,
        max_model_len=MAX_MODEL_LEN // ratio,
        max_total_seq_len=0,
        topk_indices_buffer=torch.empty(
            rows, INDEX_TOPK, dtype=torch.int32, device="cuda"
        ),
        skip_k_cache_insert=True,
        use_fp4_cache=True,
        compress_ratio=ratio,
        candidate_blocks=torch.empty(
            rows, CANDIDATE_BLOCKS, dtype=torch.int32, device="cuda"
        )
        if candidates
        else None,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
        candidate_write=candidates,
    )


def _run_indexer(indexer, step, stock=False):
    """One decode forward, stock or the executor's; returns the top-k and
    candidate rows it wrote."""
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.indexer import (
        decode_rows,
    )

    hidden = torch.empty(step.rows, 1, device="cuda")
    with override_forward_context(ForwardContext({}, {"indexer": step.metadata}, {})):
        if stock:
            indexer.forward_cuda(hidden, step.q, None, step.weights)
        else:
            decode_rows(indexer, hidden, step.q, step.weights)
    candidates = indexer.candidate_blocks
    return (
        indexer.topk_indices_buffer[: step.rows],
        None if candidates is None else candidates[: step.rows],
    )


def _index_scores(indexer, step):
    """The replicated logits, and the candidate block scores (newest block
    pinned) the candidate selection ranks."""
    from vllm.model_executor.layers.sparse_attn_indexer import (
        as_fp32_logits,
        kv_cache_as_quant_view,
    )
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, mqa_logits_mx_inputs

    decode = step.metadata.decode
    q, q_scale = step.q
    q = q.unsqueeze(1).view(torch.int8)
    q_scale, weights = mqa_logits_mx_inputs(q, q_scale.unsqueeze(1), step.weights)
    ends = decode.seq_lens.view(-1)
    logits = as_fp32_logits(
        fp8_fp4_paged_mqa_logits(
            (q, q_scale),
            kv_cache_as_quant_view(indexer.k_cache.kv_cache, INDEX_HEAD_DIM, True),
            weights,
            decode.seq_lens,
            decode.block_table,
            decode.schedule_metadata,
            max_model_len=indexer.max_model_len,
            indices=decode.indices,
        ),
        ends,
    )
    width = logits.shape[1]
    valid = logits.masked_fill(
        torch.arange(width, device="cuda") >= ends.unsqueeze(1), float("-inf")
    )
    valid = torch.nn.functional.pad(
        valid, (0, -width % CANDIDATE_BLOCK_SIZE), value=float("-inf")
    )
    blocks = valid.view(step.rows, -1, CANDIDATE_BLOCK_SIZE).amax(-1)
    live = (ends > 0).nonzero().view(-1)
    blocks[live, ((ends[live] - 1) // CANDIDATE_BLOCK_SIZE).long()] = float("inf")
    return logits, blocks


def _assert_same_sets(got, want, scores, limits, k, what):
    """Per row: equal index sets, except that indices whose scores tie
    exactly at the selection boundary may be swapped; padding rows are -1."""
    got, want, limits = got.cpu(), want.cpu(), limits.cpu()
    for row in range(got.shape[0]):
        g = got[row][got[row] >= 0].tolist()
        w = want[row][want[row] >= 0].tolist()
        assert len(set(g)) == len(g) == min(int(limits[row]), k), (what, row)
        assert (got[row][got[row] < 0] == -1).all(), (what, row)
        if set(g) != set(w):
            row_scores = scores[row]
            boundary = row_scores[torch.tensor(w, device="cuda")].min()
            swapped = torch.tensor(sorted(set(g) ^ set(w)), device="cuda")
            assert (row_scores[swapped] == boundary).all(), (what, row)


def _check_row_sharded_indexer(rank):
    generator = torch.Generator(device="cuda").manual_seed(0)  # equal on all ranks
    keys = {ratio: _index_keys(ratio, generator) for ratio, _ in INDEX_LAYERS}
    for requests, padding in INDEX_BATCHES:
        for ratio, candidates in INDEX_LAYERS:
            what = f"{requests} requests ({padding} padding), ratio {ratio}"
            generator.manual_seed(1000 * requests + ratio)
            step = _decode_step(requests, padding, ratio, keys[ratio], generator)
            indexer = _indexer(keys[ratio], ratio, candidates)
            want = [
                t if t is None else t.clone()
                for t in _run_indexer(indexer, step, stock=True)
            ]
            _late(rank, requests)
            got = _run_indexer(indexer, step)
            logits, blocks = _index_scores(indexer, step)
            ends = step.metadata.decode.seq_lens.view(-1)
            _assert_same_sets(got[0], want[0], logits, ends, INDEX_TOPK, what)
            if candidates:
                block_ends = (ends + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                _assert_same_sets(
                    got[1], want[1], blocks, block_ends, CANDIDATE_BLOCKS, what
                )
    return keys


# ------------------------------------------------------- graph replays


def _check_replays(arm, publisher, keys, rank, replays):
    """One CUDA graph of a decode step's collectives: a published ``wo_b``
    boundary, a deferred finalize with the MXFP8 layer input, a residual-only
    boundary, a full-capacity boundary and the row-sharded indexer, with the
    ranks arriving in a fixed skew. Every replay must equal the first; the
    indexer's top-k rows as sets up to boundary ties, since top-k writes them
    in any order and picks any of the indices that tie at the boundary."""
    from vllm.distributed.parallel_state import graph_capture
    from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.boundary import (
        empty_quantized_layer_input,
    )

    linear = _wo_b(rank, 0)
    qa = _attention_output(6, rank)
    moe = _moe_partial(6, rank)
    quantized = empty_quantized_layer_input(6, HIDDEN, qa.data.device)
    mixes = {rows: _mix(rows) for rows in (6, 48, CAPACITY)}
    partials = {rows: _partial(rows, rank, seed=rows) for rows in (48, CAPACITY)}
    step = _decode_step(8, 0, 2, keys[2], torch.Generator(device="cuda").manual_seed(1))
    indexer = _indexer(keys[2], 2, candidates=False)

    def decode_step():
        torch.cuda._sleep(rank * LATE_CYCLES)
        out = list(arm(publisher(linear, qa), **mixes[6]))
        out += arm.finalize(
            *moe, *mixes[6].values(), mxfp8_out=(quantized.data, quantized.scale)
        )
        out += [quantized.data, quantized.scale]
        mix = mixes[48]
        streams, _ = arm(
            partials[48], mix["residual"], mix["post"], mix["comb"], None, None, 1e-6
        )
        out.append(streams)
        out += arm(partials[CAPACITY], **mixes[CAPACITY])
        out.append(_run_indexer(indexer, step)[0])
        return [t for t in out if t is not None]

    with graph_capture(device=torch.device("cuda")) as context:
        decode_step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=context.stream):
            outputs = decode_step()
    graph.replay()
    torch.accelerator.synchronize()
    *collectives, topk = outputs
    first = [t.clone() for t in collectives]
    first_topk = topk.clone()
    logits, _ = _index_scores(indexer, step)
    ends = step.metadata.decode.seq_lens.view(-1)
    for i in range(replays):
        graph.replay()
        if i % 50 == 49 or i == replays - 1:
            _assert_equal(collectives, first, f"replay {i}")
            _assert_same_sets(topk, first_topk, logits, ends, INDEX_TOPK, f"replay {i}")


# ---------------------------------------------------------------- driver


def _worker(rank: int, port: int, replays: int, skip_path: str) -> None:
    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
    from vllm.models.deepseek_v41.nvidia.ops import mhc
    from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.all_reduce import (
        BoundaryAllReduce,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.boundary.publishing_gemm import (  # noqa: E501
        PublishingMxfp8Linear,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()), torch.inference_mode():
        init_test_distributed_environment(TP, 1, rank, str(port), local_rank=rank)
        init_workspace_manager(torch.device("cuda", rank))
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=TP, enable_expert_parallel=False
            ),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hidden_size=HIDDEN, hc_mult=HC, num_experts_per_tok=TOP_K
                )
            ),
        )
        supported = torch.tensor([int(mhc.supports_mhc_all_reduce(config))])
        torch.distributed.all_reduce(
            supported, torch.distributed.ReduceOp.MIN, get_tp_group().cpu_group
        )
        if not supported.item():
            if rank == 0:
                with open(skip_path, "w") as f:
                    f.write("no NVLink multicast for the fused all-reduce + mHC")
            cleanup_dist_env_and_memory()
            return
        arm = BoundaryAllReduce(config.model_config.hf_config, True)
        assert arm.capacity == CAPACITY and arm.quantize_layer_input
        publisher = PublishingMxfp8Linear(arm)
        _check_all_reduce_mhc(arm, rank)
        _check_boundaries(arm, rank)
        _check_publishing_wo_b(arm, publisher, rank)
        _check_logits_gather(rank)
        keys = _check_row_sharded_indexer(rank)
        _check_replays(arm, publisher, keys, rank, replays)
        torch.accelerator.synchronize()
        cleanup_dist_env_and_memory()


def run(replays: int) -> str | None:
    """Run the checks on four GPUs; returns why they were skipped, if so."""
    from vllm.utils.network_utils import get_open_port

    with tempfile.TemporaryDirectory() as tmp:
        skip_path = os.path.join(tmp, "skipped")
        torch.multiprocessing.spawn(
            _worker, args=(get_open_port(), replays, skip_path), nprocs=TP
        )
        if os.path.exists(skip_path):
            with open(skip_path) as f:
                return f.read()
    return None


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="needs SM100"
)
def test_specialized_decode_numerics():
    if torch.accelerator.device_count() < TP:
        pytest.skip("needs four GPUs")
    from vllm.models.deepseek_v41.nvidia.specialized_decode import FLASHINFER_VERSION

    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != FLASHINFER_VERSION:
        pytest.skip(f"needs flashinfer-python {FLASHINFER_VERSION}, found {version}")
    skipped = run(REPLAYS)
    if skipped:
        pytest.skip(skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replays", type=int, default=REPLAYS)
    skipped = run(parser.parse_args().replays)
    print(f"skipped: {skipped}" if skipped else "passed")
