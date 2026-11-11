# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import math
import statistics
import types
from types import SimpleNamespace

import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
from vllm.models.deepseek_v41.amd.rocm_mega_attn import (
    DeepseekV41ROCMAiterMegaAttnAttention,
)
from vllm.models.deepseek_v41.common.ops import quantize_and_insert_k_cache
from vllm.platforms import current_platform
from vllm.platforms.rocm import get_cdna_version
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
)
from vllm.v1.worker.workspace import init_workspace_manager

HEAD_DIM, ROPE_DIM, WINDOW, BLOCK, RECORD_BYTES = 512, 64, 128, 64, 584


def make_cos_sin_cache(max_pos, device):
    inv_freq = 10000 ** (
        -torch.arange(0, ROPE_DIM, 2, device=device).float() / ROPE_DIM
    )
    angles = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def inverse_rope_reference(x, positions, cos_sin):
    out = x.float().clone()
    cos, sin = cos_sin[positions].chunk(2, dim=-1)
    tail = out[..., -ROPE_DIM:].reshape(x.shape[0], x.shape[1], -1, 2)
    even, odd = tail[..., 0].clone(), tail[..., 1].clone()
    tail[..., 0] = even * cos[:, None] + odd * sin[:, None]
    tail[..., 1] = odd * cos[:, None] - even * sin[:, None]
    return out


def rope_q(layer, q, positions):
    """Q after the RoPE the ordinary Q prep applies; CDNA3 mega takes Q before."""
    cache = layer.swa_cache_layer.kv_cache
    return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        torch.zeros(q.shape[0], HEAD_DIM, dtype=q.dtype, device=q.device),
        cache.view(cache.shape[0], -1),
        torch.empty(0, dtype=torch.int64, device=q.device),
        positions,
        layer.rotary_emb.cos_sin_cache,
        q.shape[1],
        layer.eps,
        BLOCK,
        False,
        False,
        True,
        False,
    )


def make_cache(rows, device, use_fnuz=False):
    blocks = math.ceil(rows / BLOCK)
    cache = torch.zeros(blocks, BLOCK, RECORD_BYTES, dtype=torch.uint8, device=device)
    kv = torch.randn(rows, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.25
    quantize_and_insert_k_cache(
        kv,
        cache,
        torch.arange(rows, device=device),
        block_size=BLOCK,
        bytes_per_token=RECORD_BYTES,
        use_fnuz=use_fnuz,
    )
    if not use_fnuz and current_platform.is_fp8_fnuz():
        # As the compressed-cache writer stores OCP records on FNUZ platforms.
        nope = cache.view(blocks, -1)[:, : BLOCK * 576].view(blocks, BLOCK, 576)
        nope[..., :448].masked_fill_(nope[..., :448] == 0x80, 0)
    return cache


def decode_cache_reference(cache, use_fnuz=False):
    """Read ROCm's block-wide data and scale planes independently of Triton."""
    blocks = cache.reshape(cache.shape[0], -1)
    records = blocks[:, : BLOCK * 576].reshape(-1, 576)
    fp8_dtype = torch.float8_e4m3fnuz if use_fnuz else torch.float8_e4m3fn
    nope = records[:, :448].contiguous().view(fp8_dtype).float()
    rope = records[:, 448:].contiguous().view(torch.bfloat16).float()
    scales = blocks[:, BLOCK * 576 :].reshape(-1, 8)[:, :7]
    nope *= torch.exp2(scales.float() - 127).repeat_interleave(64, dim=-1)
    return torch.cat((nope, rope), dim=-1)


def output_buffers(out):
    return (out,) if isinstance(out, torch.Tensor) else (out.data, out.scale)


def dequant_output(out):
    # CDNA3 stores BF16 [tokens, heads, 512]; CDNA4 stores MXFP8 rows.
    if isinstance(out, torch.Tensor):
        return out.float()
    scales = torch.exp2(out.scale.float() - 127).repeat_interleave(32, dim=-1)
    return (out.data.float() * scales).reshape(-1, out.data.shape[1] // HEAD_DIM, HEAD_DIM)


class BenchmarkAttention(DeepseekV41ROCMAiterMegaAttnAttention):
    """Construct only the attention state; skip model projections and weights."""

    def __init__(self, heads, n_decode, n_prefill, topk, ratio, device):
        torch.nn.Module.__init__(self)
        n_tokens = n_decode + n_prefill
        seq_len = max(WINDOW, ratio * topk) + max(n_prefill, 1)
        self.prefill_start = seq_len - n_prefill
        self.n_local_heads = self.padded_heads = heads
        self.head_dim, self.rope_head_dim = HEAD_DIM, ROPE_DIM
        self.nope_head_dim = HEAD_DIM - ROPE_DIM
        self.scale = HEAD_DIM**-0.5
        self.eps = 1e-6
        self.kv_mxfp8 = False
        # As with CUDA graphs capped at 512 tokens.
        self._attn_rotates_q_from = 513 if get_cdna_version() == 3 else None
        self._prefill_rotate_q = False
        self.window_size = WINDOW
        self.compress_ratio = ratio if topk else 0
        self.max_model_len = seq_len
        self.max_num_batched_tokens = n_tokens
        self.PREFILL_CHUNK_SIZE = 1
        self._use_aiter_sparse_mla = False
        self._has_kv_transfer = False
        self.kv_cache_dtype = "fp8_ds_mla"
        self.compressed_cache_prefix = "compressed" if topk else None
        self.swa_cache_layer = SimpleNamespace(
            prefix="swa",
            kv_cache=make_cache(seq_len, device, current_platform.is_fp8_fnuz()),
        )
        self.extra_cache = make_cache(seq_len // ratio, device) if topk else None
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=make_cos_sin_cache(seq_len + 16, device)
        )
        self.attn_sink = torch.linspace(-1, 1, heads, device=device)
        self._index_source_prefix = "source"
        self._static_forward_context = {"source": self}
        self._topk_ragged_cache = {}
        self.topk_indices_buffer = torch.randint(
            0,
            (seq_len - n_prefill) // ratio,
            (n_tokens, topk),
            device=device,
            dtype=torch.int32,
        )
        if topk:
            self.topk_indices_buffer[1::2, topk // 2 :] = -1
        swa_indices = torch.randint(
            0, seq_len, (n_decode, WINDOW), device=device, dtype=torch.int32
        )
        swa_lens = torch.full((n_decode,), WINDOW, device=device, dtype=torch.int32)
        if n_decode > 1:
            # An empty padded row must emit zeros even with a finite sink.
            swa_lens[0] = 0
            self.topk_indices_buffer[0] = -1
        ragged, indptr = (
            build_ragged_indices_from_dense(swa_indices, swa_lens)
            if n_decode
            else (None, None)
        )
        n_reqs = n_decode + bool(n_prefill)
        starts_cpu = torch.tensor(
            list(range(n_decode + 1)) + ([n_tokens] if n_prefill else []),
            dtype=torch.int32,
        )
        table = torch.arange(
            self.swa_cache_layer.kv_cache.shape[0], device=device, dtype=torch.int32
        ).repeat(n_reqs, 1)
        seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
        gather_len = WINDOW + n_prefill
        compressed_rows = seq_len // ratio if topk else 0
        self.metadata = SimpleNamespace(
            num_decodes=n_decode,
            num_decode_tokens=n_decode,
            num_prefills=int(bool(n_prefill)),
            num_prefill_tokens=n_prefill,
            block_size=BLOCK,
            slot_mapping=torch.empty(0, device=device, dtype=torch.int64),
            block_table=table,
            decode_swa_indices=swa_indices,
            decode_swa_lens=swa_lens,
            decode_swa_ragged_indices=ragged,
            decode_swa_ragged_indptr=indptr,
            token_to_req_indices=torch.cat(
                (
                    torch.arange(n_decode, device=device, dtype=torch.int32),
                    torch.full(
                        (n_prefill,), n_decode, device=device, dtype=torch.int32
                    ),
                )
            ),
            is_valid_token=torch.ones(n_tokens, device=device, dtype=torch.bool),
            prefill_seq_lens=seq_lens,
            prefill_seq_lens_cpu=seq_lens.cpu(),
            prefill_gather_lens=torch.tensor(
                [gather_len], device=device, dtype=torch.int32
            ),
            query_start_loc_cpu=starts_cpu,
            query_start_loc=starts_cpu.to(device),
            get_prefill_chunk_plan=lambda **kwargs: [
                (0, 1, compressed_rows, compressed_rows + gather_len)
            ],
        )
        extra_metadata = None
        if topk:
            extra_metadata = SimpleNamespace(
                block_size=BLOCK * self.compress_ratio,
                block_table=torch.arange(
                    self.extra_cache.shape[0], device=device, dtype=torch.int32
                ).repeat(n_reqs, 1),
            )
        self.context = ForwardContext(
            no_compile_layers={},
            attn_metadata={"swa": self.metadata, "compressed": extra_metadata},
            slot_mapping={},
        )

    def _compressed_kv_cache(self):
        return self.extra_cache


def baseline_forward_mqa(layer, *args):
    # forward_mqa dispatches through self, so the mega overrides must be
    # shadowed for the baseline arm to take the ordinary prefill path.
    base = DeepseekV41ROCMAiterMLAAttention
    layer._prefill_rotates_bf16 = base._prefill_rotates_bf16
    layer._forward_prefill = types.MethodType(base._forward_prefill, layer)
    layer._reserve_prefill_workspace = types.MethodType(
        base._reserve_prefill_workspace, layer
    )
    try:
        base.forward_mqa(layer, *args)
    finally:
        del layer._prefill_rotates_bf16
        del layer._forward_prefill
        del layer._reserve_prefill_workspace


def attention_reference(layer, q, positions):
    swa = decode_cache_reference(
        layer.swa_cache_layer.kv_cache, current_platform.is_fp8_fnuz()
    )
    extra = (
        None if layer.extra_cache is None else decode_cache_reference(layer.extra_cache)
    )
    meta = layer.metadata
    topk_rows = layer.topk_indices_buffer.cpu().tolist()
    decode_rows = meta.decode_swa_indices.cpu().tolist()
    decode_lens = meta.decode_swa_lens.cpu().tolist()
    result = torch.zeros_like(q, dtype=torch.float32)
    for row in range(q.shape[0]):
        if row < meta.num_decode_tokens:
            slots = decode_rows[row][: decode_lens[row]]
            candidates = topk_rows[row]
        else:
            pos = layer.prefill_start + row - meta.num_decode_tokens
            slots = list(range(max(0, pos - WINDOW + 1), pos + 1))
            limit = (pos + 1) // max(layer.compress_ratio, 1)
            candidates = topk_rows[row][: min(limit, len(topk_rows[row]))]
        selected = [swa[slots]]
        if extra is not None:
            selected.append(extra[[i for i in candidates if i >= 0]])
        kv = torch.cat(selected)
        if kv.shape[0]:
            logits = q[row].float() @ kv.T * layer.scale
            probs = torch.softmax(torch.cat((logits, layer.attn_sink[:, None]), -1), -1)
            result[row] = probs[:, :-1] @ kv
    return inverse_rope_reference(result, positions, layer.rotary_emb.cos_sin_cache)


def time_graph(calls, iters, rounds):
    for call in calls:
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    for _ in range(5):
        graph.replay()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(rounds)
    ]
    for start, end in events:
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
    torch.accelerator.synchronize()
    return statistics.median(
        start.elapsed_time(end) * 1000 / (iters * len(calls)) for start, end in events
    )


def bench_case(n_decode, n_prefill, args, device):
    calls = [[], []]
    checks = []
    for _ in range(args.rotating_slots):
        layer = BenchmarkAttention(
            args.local_heads,
            n_decode,
            n_prefill,
            args.topk_extra,
            args.compress_ratio,
            device,
        )
        n_tokens = n_decode + n_prefill
        q = torch.randn(
            n_tokens,
            args.local_heads,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        ) * 0.25
        positions = torch.cat((
            torch.full(
                (n_decode,),
                layer.max_model_len - 1,
                device=device,
                dtype=torch.int64,
            ),
            torch.arange(layer.prefill_start, layer.max_model_len, device=device),
        ))
        kv = torch.zeros(n_tokens, HEAD_DIM, device=device, dtype=torch.bfloat16)
        # On large CDNA3 batches mega rotates Q itself, decode rows in place;
        # otherwise, and for the baseline and the reference, Q is after RoPE.
        raw_q = layer._attn_rotates_q(q)
        q_rope = rope_q(layer, q, positions) if raw_q else q
        outputs = [layer._alloc_attn_out(n_tokens, q) for _ in range(2)]
        expected = attention_reference(layer, q_rope, positions)
        for arm, method in enumerate(
            (DeepseekV41ROCMAiterMegaAttnAttention.forward_mqa, baseline_forward_mqa)
        ):
            arm_q = q if arm == 0 else q_rope
            restore = arm_q[:n_decode].clone() if raw_q and arm == 0 else None
            output = outputs[arm]
            buffers = output_buffers(output)
            dummy = ForwardContext(
                no_compile_layers={}, attn_metadata=None, slot_mapping={}
            )
            with override_forward_context(dummy):
                method(layer, arm_q, kv, positions, output)
            for buffer in buffers:
                assert torch.count_nonzero(buffer.float()).item() == 0
            buffers[0].fill_(float("nan"))

            def call(
                method=method,
                layer=layer,
                q=arm_q,
                kv=kv,
                restore=restore,
                positions=positions,
                output=outputs[arm],
            ):
                if restore is not None:
                    q[: restore.shape[0]].copy_(restore)
                with override_forward_context(layer.context):
                    method(layer, q, kv, positions, output)

            call()
            actual = dequant_output(outputs[arm])
            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
            calls[arm].append(call)
            checks.append((outputs[arm], expected))
        torch.testing.assert_close(
            dequant_output(outputs[0]), dequant_output(outputs[1]), atol=args.atol, rtol=args.rtol,
        )
    if args.check_only:
        return None
    timings = tuple(time_graph(arm, args.iters, args.rounds) for arm in calls)
    for output, expected in checks:
        torch.testing.assert_close(
            dequant_output(output), expected, atol=args.atol, rtol=args.rtol
        )
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-s-q", nargs="+", type=int, default=[32, 128, 512, 2048])
    parser.add_argument("--prefill-s-q", nargs="+", type=int, default=[32, 128, 512, 2048])
    parser.add_argument(
        "--mixed", nargs=2, type=int, default=[4, 16], metavar=("DECODE", "PREFILL")
    )
    parser.add_argument("--topk-extra", type=int, default=512)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--local-heads", type=int, choices=[8, 16, 32, 64], default=16)
    parser.add_argument(
        "--rotating-slots",
        type=int,
        default=2,
        help="independent input/cache sets per HIP graph; small sets "
        "may remain cache-resident",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--atol", type=float, default=3e-3)
    parser.add_argument("--rtol", type=float, default=0.08)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not current_platform.is_rocm() or not torch.cuda.is_available():
        parser.error("a ROCm GPU is required")
    if get_cdna_version() not in (3, 4):
        parser.error("mega attention prefill requires CDNA3 or CDNA4")
    if (
        min(args.rotating_slots, args.iters, args.rounds, args.compress_ratio) <= 0
        or args.topk_extra < 0
    ):
        parser.error("slots/iters/rounds must be positive and topk-extra nonnegative")
    cases = [(n, 0) for n in args.decode_s_q] + [(0, n) for n in args.prefill_s_q]
    cases.append(tuple(args.mixed))
    device = torch.device("cuda")
    init_workspace_manager(device)
    torch.manual_seed(0)
    print(
        f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"HIP={torch.version.hip} heads={args.local_heads} topk={args.topk_extra} "
        f"ratio={args.compress_ratio} "
        f"atol={args.atol} rtol={args.rtol} "
        f"slots={args.rotating_slots}; mega uses sparse fallback"
    )
    print(
        f"{'decode':>7} {'prefill':>7} {'mega_us':>10} {'baseline_us':>12} {'ratio':>8}"
    )
    for n_decode, n_prefill in cases:
        if min(n_decode, n_prefill) < 0 or n_decode + n_prefill == 0:
            continue
        timings = bench_case(n_decode, n_prefill, args, device)
        if timings is None:
            print(f"{n_decode:7d} {n_prefill:7d} correctness PASS")
        else:
            mega, baseline = timings
            print(
                f"{n_decode:7d} {n_prefill:7d} {mega:10.2f} "
                f"{baseline:12.2f} {baseline / mega:8.3f}"
            )


if __name__ == "__main__":
    main()
