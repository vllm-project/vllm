# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-layer MXFP8 arithmetic, cache mappings, padding and TP fault fencing."""

import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 only", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402

from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (  # noqa: E402
    gemma_fused_add_rmsnorm,
)
from vllm.models.minimax_m3.amd.specialized_decode.runtime.compile import (  # noqa: E402
    _writable_flydsl_cache,
)

E, H, INTER = 129, 6144, 768
EPS, LIMIT, ALPHA, BETA, SHARED = 1e-6, 7.0, 1.702, 1.0, 128


@pytest.mark.parametrize("query_lens", [(64, 3), (129, 65)])
@pytest.mark.parametrize("quantized_query", [False, True])
def test_layer_dense_prefill_preserves_cache_and_query_scales(
    query_lens, quantized_query
):
    """Cached mixed prefills honor paged addressing and non-unit Q/K/V scales."""
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionImpl,
        TritonAttentionMetadata,
    )

    dev = torch.device("cuda")
    generator = torch.Generator(device=dev).manual_seed(19)
    lengths = (257, 513)
    cache = torch.randn(8, 1, 128, 256, device=dev, generator=generator).to(
        torch.float8_e4m3fn
    )
    table = torch.tensor([[6, 2, 7, 0, 0], [5, 1, 4, 0, 3]], device=dev)
    query = torch.randn(
        sum(query_lens), 16, 128, device=dev, generator=generator
    ).bfloat16()
    layer = torch.nn.Module()
    for name, scale in (("q", 0.75), ("k", 0.5), ("v", 0.25)):
        layer.register_buffer(f"_{name}_scale", torch.tensor(scale, device=dev))
    if quantized_query:
        query = query.to(torch.float8_e4m3fn)
    output = torch.empty(query.shape, dtype=torch.bfloat16, device=dev)
    metadata = TritonAttentionMetadata(
        num_actual_tokens=query.shape[0],
        max_query_len=max(query_lens),
        query_start_loc=torch.tensor(
            [0, query_lens[0], sum(query_lens)], device=dev, dtype=torch.int32
        ),
        max_seq_len=max(lengths),
        seq_lens=torch.tensor(lengths, device=dev, dtype=torch.int32),
        block_table=table.to(torch.int32),
        slot_mapping=torch.empty(query.shape[0], dtype=torch.int64, device=dev),
        seq_threshold_3D=256,
        num_par_softmax_segments=256,
        softmax_segm_output=torch.empty(0, device=dev),
        softmax_segm_max=torch.empty(0, device=dev),
        softmax_segm_expsum=torch.empty(0, device=dev),
        causal=True,
        use_aiter_prefill=True,
    )
    impl = TritonAttentionImpl(16, 128, 128**-0.5, 1, None, None, "fp8")
    impl.forward(layer, query, None, None, cache, metadata, output)
    offset = 0
    for seq, (query_len, length) in enumerate(zip(query_lens, lengths)):
        pages = cache[table[seq, : (length + 127) // 128]].float()
        key, value = pages.reshape(-1, 256)[:length].split(128, dim=-1)
        q = query[offset : offset + query_len].float()
        if quantized_query:
            q = (q * layer._q_scale).bfloat16().float()
        scores = torch.einsum("qhd,kd->qhk", q, key * layer._k_scale) * 128**-0.5
        positions = length - query_len + torch.arange(query_len, device=dev)
        scores.masked_fill_(
            torch.arange(length, device=dev)[None, None, :] > positions[:, None, None],
            -torch.inf,
        )
        reference = scores.softmax(-1) @ (value * layer._v_scale)
        actual = output[offset : offset + query_len].float()
        assert actual.isfinite().all()
        assert (actual - reference).norm() / reference.norm() < 0.01
        assert (
            torch.nn.functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0
            )
            > 0.9999
        )
        offset += query_len


def _metrics(got, want):
    got, want = got.float().flatten(), want.float().flatten()
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
    rel = ((got - want).norm() / want.norm()).item()
    return cos, rel


@pytest.fixture(scope="module")
def weights():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)

    def codes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev, generator=g)

    def scales(*shape):
        # e8m0 codes around 2^-7: gate/up pre-activations of a few units
        return torch.randint(
            119, 123, shape, dtype=torch.uint8, device=dev, generator=g
        )

    def param(t):
        return torch.nn.Parameter(t, requires_grad=False)

    w13, w2, s13, s2, _, _ = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        layer=torch.nn.Module(),
        w13_weight=param(codes(E, 2 * INTER, H // 2)),
        w2_weight=param(codes(E, H, INTER // 2)),
        w13_weight_scale=param(scales(E, 2 * INTER, H // 32)),
        w2_weight_scale=param(scales(E, H, INTER // 32)),
    )
    return SimpleNamespace(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        post_norm=torch.randn(H, device=dev, generator=g).mul(0.1).bfloat16(),
        router=torch.randn(128, H, device=dev, generator=g).mul(0.02),
        router_bias=torch.randn(128, device=dev, generator=g).mul(0.01),
    )


def test_full_layer_metadata_keeps_cache_mappings_separate():
    """Speculative rows retain each cache's block ids and keep padding inert."""
    from vllm.models.minimax_m3.amd.specialized_decode.mono.step import (
        expand_rows,
    )

    main = torch.arange(9, device="cuda", dtype=torch.int32).view(3, 3)
    index = torch.arange(15, device="cuda", dtype=torch.int32).view(3, 5) + 100
    lengths = torch.tensor([132, 260, 388], device="cuda", dtype=torch.int32)
    slots = torch.tensor(list(range(128, 140)) + [-1] * 4, device="cuda")
    index_slots = torch.tensor(list(range(768, 780)) + [-1] * 4, device="cuda")
    out_main = torch.empty(16, 256, device="cuda", dtype=torch.int32)
    out_index = torch.empty_like(out_main)
    out_lengths = torch.empty(16, device="cuda", dtype=torch.int32)
    out_slots, out_index_slots = torch.empty_like(slots), torch.empty_like(slots)
    batch = torch.empty_like(out_lengths)
    expand_rows[(16, 1)](
        main,
        index,
        lengths,
        slots,
        index_slots,
        out_main,
        out_index,
        out_lengths,
        out_slots,
        out_index_slots,
        batch,
        MAIN_STRIDE=3,
        INDEX_STRIDE=5,
        WIDTH=256,
        Q=4,
        REQUESTS=3,
        BLOCK=256,
    )
    request = torch.arange(16, device="cuda").div(4, rounding_mode="floor").clamp(max=2)
    assert torch.equal(out_main[:, :3], main[request])
    assert torch.equal(out_index[:, :5], index[request])
    assert not out_main[:, 3:].any() and not out_index[:, 5:].any()
    expected_lengths = (
        lengths.repeat_interleave(4) - 3 + torch.arange(12, device="cuda") % 4
    )
    assert torch.equal(out_lengths[:12], expected_lengths)
    assert torch.equal(out_slots[:12], slots[:12] // 128 * 256 + slots[:12] % 128)
    assert torch.equal(out_index_slots, index_slots)
    assert (out_lengths[12:] == 1).all() and (out_slots[12:] == -1).all()
    assert (batch[12:] == -1).all()


def _mxfp8(x):
    groups = x.float().unflatten(-1, (-1, 32))
    codes = (groups.abs().amax(-1) / 448).log2().ceil().add(127).clamp(1, 254)
    scale = torch.exp2(codes - 127)
    q = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (
        q.flatten(-2),
        codes.to(torch.int32),
        (q.float() * scale[..., None]).flatten(-2),
    )


def _plain_expert(weight, scales, expert):
    """Invert AITER's permutations before an independent torch matmul."""
    n, packed_k = weight.shape[1:]
    packed = (
        weight[expert]
        .view(torch.uint8)
        .view(n // 16, packed_k // 32, 2, 16, 16)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(n, packed_k)
    )
    cols = 2 * packed_k // 32
    codes = (
        scales.view(-1, n // 32, cols // 8, 4, 16, 2, 2)[expert]
        .permute(0, 5, 3, 1, 4, 2)
        .contiguous()
        .view(n, cols)
    )
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
        dtype=torch.float32,
    )
    indices = torch.stack((packed & 15, packed >> 4), -1).long()
    return (
        levels[indices].view(n, cols, 32) * torch.exp2(codes.float() - 127)[..., None]
    ).flatten(-2)


def _check_layer_post_reference(weights, s, padded, fuse_k1, *, blocks=1):
    """Sparse attention through MoE against independent Torch MXFP8 math.

    Uniform attention over constant values makes its exact result known.
    Distinct requests occupy interleaved K/V blocks with scalar cache scales;
    fixed routes isolate the projection, normalization, and expert arithmetic.
    Both narrow and wide task schedules must preserve that result.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.minimax_m3.amd.specialized_decode.mono.kernels.post_attn import (  # noqa: E501
        K4_ABI,
        build_post_attn_kernel,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layout import (
        SCRATCH,
        SCRATCH_BYTES,
        sym_layout,
    )

    dev = weights.w13.device
    live = s - padded
    length = 16 if blocks == 1 else blocks * 128
    selected = min(blocks, 16)
    assert blocks == 1 or fuse_k1
    g = torch.Generator(device=dev).manual_seed(42)
    values = torch.randn(s, blocks, 128, device=dev, generator=g).to(
        torch.float8_e4m3fn
    )
    if blocks > 1:
        # Basis values identify selected blocks and keep split means exact in BF16.
        ids = (
            torch.arange(blocks, device=dev)[None, :]
            + 17 * torch.arange(s, device=dev)[:, None]
        ) % 128
        basis = torch.nn.functional.one_hot(ids, 128).float() * 2
        values = basis.to(torch.float8_e4m3fn)
    cache = torch.zeros(
        s * blocks, 2, 8, 128, 16, device=dev, dtype=torch.float8_e4m3fn
    )
    cache[:, 1].copy_(values.view(-1, 128)[:, None, :, None].expand(-1, 8, 128, 16))
    expected_values = cache[:, 1].clone()
    keys = cache.view(s * blocks * 16, 1, 8, 16, 16)
    vals = cache.view(s * blocks * 16, 1, 1, 128, 16)[8:]
    scale = torch.tensor([0.5], device=dev)
    table = torch.zeros(s, max(16, blocks), dtype=torch.int32, device=dev)
    table[:, :blocks] = torch.arange(s * blocks, device=dev).view(s, blocks)
    lengths = torch.full((s,), length, dtype=torch.int32, device=dev)
    query = torch.zeros(s, 2048, dtype=torch.bfloat16, device=dev)
    residual = torch.randn(s, H, device=dev, generator=g).bfloat16()
    output = torch.empty_like(residual)
    next_residual = torch.empty_like(residual)
    wo = torch.randn(H, 2048, device=dev, generator=g).to(torch.float8_e4m3fn)
    packed_wo = rocm_aiter_ops.shuffle_weight(wo)
    wo_scale = torch.full((H,), 2048**-0.5, device=dev)
    router = torch.zeros_like(weights.router, dtype=torch.bfloat16)
    bias = torch.zeros_like(weights.router_bias)
    bias[:4] = torch.tensor([4, 3, 2, 1], device=dev)
    scratch = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=dev)
    sym = torch.zeros(sym_layout(1)["_bytes"], dtype=torch.uint8, device=dev)
    peers = torch.tensor([sym.data_ptr()], dtype=torch.int64, device=dev)
    batch_ids = torch.arange(s, dtype=torch.int32, device=dev)
    batch_ids[live:] = -1
    pointers = dict(
        h_in=residual.data_ptr(), q=query.data_ptr(), block_table=table.data_ptr(),
        seq_lens=lengths.data_ptr(), k_cache=keys.data_ptr(), v_cache=vals.data_ptr(),
        k_scale=scale.data_ptr(), v_scale=scale.data_ptr(), w_o=packed_wo.data_ptr(),
        s_o=wo_scale.data_ptr(), g_post=weights.post_norm.data_ptr(),
        w_gate=router.data_ptr(), bias=bias.data_ptr(), w13=weights.w13.data_ptr(),
        s13=weights.s13.data_ptr(), w2=weights.w2.data_ptr(), s2=weights.s2.data_ptr(),
        h_mid=next_residual.data_ptr(), ar_out=output.data_ptr(),
        scratch=scratch.data_ptr(), sym=sym.data_ptr(), peers=peers.data_ptr(),
        rank=0, layer=3, bt_width=table.shape[1], q_len=1, tl=0, k1_args=0,
        positions=0, slot_mapping=0, res=0, batch_ids=batch_ids.data_ptr(),
        index_block_table=0, index_bt_width=0, index_slot_mapping=0, ar_in=0,
        mode=1,
    )  # fmt: skip
    tokens_per_block = min(length, 128)
    total_keys = selected * tokens_per_block
    value_sum = values[:, -selected:].float().sum(1) * tokens_per_block
    if fuse_k1:
        from vllm.models.minimax_m3.amd.specialized_decode.mono.kernels.pre_attn import (  # noqa: E501
            K1_ARGS,
        )
        from vllm.models.minimax_m3.amd.specialized_decode.mono.kernels.pre_attn import (  # noqa: E501
            SCRATCH_BYTES as PRE_BYTES,
        )

        # Exact by construction: norm(.5 + .5) rounds to one; each projection
        # row selects one input, and identity RoPE leaves the normalized heads.
        hidden = torch.full_like(residual, 0.5)
        previous_residual = hidden.clone()
        residual.fill_(1)
        g_in = torch.zeros(H, dtype=torch.bfloat16, device=dev)
        g_q = torch.full((128,), -1, dtype=torch.bfloat16, device=dev)
        g_head = torch.zeros_like(g_q)
        qkv_weight = torch.zeros(2560, H, dtype=torch.bfloat16, device=dev)
        qkv_weight[torch.arange(2560), torch.arange(2560)] = 1
        qkv_packed = rocm_aiter_ops.shuffle_weight(qkv_weight.to(torch.float8_e4m3fn))
        qkv_scale = torch.ones(2560, device=dev)
        cos_sin = torch.cat((torch.ones(length, 32), torch.zeros(length, 32)), -1).to(
            device=dev, dtype=torch.bfloat16
        )
        positions = torch.full((s,), length - 1, dtype=torch.int64, device=dev)
        last_offset = (length - 1) % 128
        main_pages = table[:, blocks - 1].long()
        main_slots = main_pages * 256 + last_offset
        index_table = table.clone()
        index_table[:, :blocks] = table[:, :blocks].flip(1)
        index_slots = index_table[:, blocks - 1].long() * 128 + 31
        main_slots[live:] = -1
        index_slots[live:] = -1
        index_cache = torch.full(
            (s * blocks * 128, 128), -7, dtype=torch.float8_e4m3fn, device=dev
        )
        if blocks > 1:
            # For q=1, each key's dot product is its logical block id exactly.
            block_ids = torch.arange(blocks, device=dev)[:, None]
            dimensions = torch.arange(128, device=dev)[None, :]
            index_keys = (block_ids // 128 + (dimensions < block_ids % 128)).to(
                torch.float8_e4m3fn
            )
            index_cache.view(s, blocks, 128, 128).copy_(
                index_keys.flip(0)[None, :, None, :].expand(s, blocks, 128, 128)
            )
        expected_index = index_cache.clone()
        expected_index[index_slots[:live]] = 1
        for row in range(live):
            expected_values[main_pages[row], last_offset // 16, :, last_offset % 16] = 2
        iq = torch.empty(s, 128, dtype=torch.bfloat16, device=dev)
        pre = torch.zeros(PRE_BYTES, dtype=torch.uint8, device=dev)
        k1_values = dict(
            g_in=g_in.data_ptr(), w_qkv=qkv_packed.data_ptr(),
            s_qkv=qkv_scale.data_ptr(), g_q=g_q.data_ptr(), g_k=g_head.data_ptr(),
            g_iq=g_head.data_ptr(), g_ik=g_head.data_ptr(), cos_sin=cos_sin.data_ptr(),
            index_cache=index_cache.data_ptr(), iq_out=iq.data_ptr(),
            scratch=pre.data_ptr(),
        )  # fmt: skip
        k1 = torch.tensor(
            [k1_values[n] for n in K1_ARGS], dtype=torch.int64, device=dev
        )
        pointers.update(
            k1_args=k1.data_ptr(),
            positions=positions.data_ptr(),
            slot_mapping=main_slots.data_ptr(),
            res=previous_residual.data_ptr(),
            index_block_table=index_table.data_ptr(),
            index_bt_width=index_table.shape[1],
            index_slot_mapping=index_slots.data_ptr(),
            ar_in=hidden.data_ptr(),
        )
        # Uniform attention over the highest-scoring blocks; one V is replaced.
        value_sum[:live] += 2 - values[:live, -1].float()
    # Uniform probabilities quantize to 448. Keep the FP32 PV scale step:
    # cancelling 448 algebraically changes BF16 midpoint rounding.
    attention = (
        ((value_sum * 448) * (scale * (1.0 / 448)) / total_keys)
        .repeat(1, 16)
        .bfloat16()
    )
    _writable_flydsl_cache()
    launch = build_post_attn_kernel(
        1,
        128**-0.5,
        EPS,
        2.0,
        1.0,
        7.0,
        0,
        1,
        tokens=s,
        debug=True,
        fuse_k1=fuse_k1,
    )
    compiled = flyc.compile(launch, *K4_ABI.zeros())
    compiled(*K4_ABI.pack(pointers), torch.cuda.current_stream())

    if blocks > 1:
        from vllm.models.minimax_m3.amd.specialized_decode.mono.layout import (
            SPARSE_TABLE_WORDS,
        )

        start = SCRATCH["sparse_table"]
        selected_pages = (
            scratch[start : start + s * SPARSE_TABLE_WORDS * 4]
            .view(torch.int32)
            .view(s, SPARSE_TABLE_WORDS)[:, :128]
        )
        expected_pages = (
            table[:, blocks - selected : blocks, None] * 16
            + torch.arange(8, device=dev)
        ).flatten(1)
        assert torch.equal(selected_pages.sort(1).values, expected_pages.sort(1).values)

    if fuse_k1:
        assert torch.equal(query[:live], torch.zeros_like(query[:live]))
        assert torch.equal(iq[:live], torch.ones_like(iq[:live]))
        assert torch.equal(
            index_cache.view(torch.uint8), expected_index.view(torch.uint8)
        )
        for row in range(live):
            page = main_pages[row] * 16 + last_offset // 16
            assert (keys[page, 0, :, last_offset % 16, :].float() == 2).all()
        assert cache[:, 0].view(torch.uint8).count_nonzero().item() == live * 128
        assert torch.equal(
            cache[:, 1].view(torch.uint8), expected_values.view(torch.uint8)
        )
    attention = attention.float()
    a_scale = attention.abs().amax(-1, keepdim=True) / 448
    aq = (attention / a_scale).to(torch.float8_e4m3fn).float()
    part = ((aq @ wo.float().T) * a_scale * wo_scale).bfloat16()
    x, want_residual = gemma_fused_add_rmsnorm(part, residual, weights.post_norm, EPS)
    # ATOM sums the route-weighted experts in FP32 before rounding this TP shard.
    want = torch.zeros(s, H, dtype=torch.float32, device=dev)
    for expert in (0, 1, 2, 3, SHARED):
        upgate = _mxfp8(x)[2] @ _plain_expert(weights.w13, weights.s13, expert).T
        gate, up = upgate.chunk(2, -1)
        gate, up = gate.clamp(max=LIMIT), up.clamp(-LIMIT, LIMIT)
        mid = (gate * (ALPHA * gate).sigmoid() * (up + BETA)).bfloat16()
        term = _mxfp8(mid)[2] @ _plain_expert(weights.w2, weights.s2, expert).T
        want += term * (1.0 if expert == SHARED else 0.5)
    cos, rel = _metrics(output[:live], want[:live].bfloat16())
    residual_cos, residual_rel = _metrics(next_residual[:live], want_residual[:live])
    diagnostic = scratch[SCRATCH["diag"] :].view(torch.int32).view(256, 8)
    assert not diagnostic[:, 0].any(), diagnostic[diagnostic[:, 0] != 0]
    assert cos > 0.9999 and rel < 1e-2, (cos, rel)
    assert residual_cos > 0.9999 and residual_rel < 1e-2


@pytest.mark.parametrize(
    "s,padded",
    [(1, 0), (4, 0), (8, 0), (12, 0), (16, 0), (4, 1), (8, 1), (12, 1), (16, 1)],
)
@pytest.mark.parametrize("fuse_k1", [False, True])
def test_layer_post_matches_mxfp8_reference(weights, s, padded, fuse_k1):
    _check_layer_post_reference(weights, s, padded, fuse_k1)


@pytest.mark.parametrize("rows,blocks", [(4, 32), (4, 544), (16, 544)])
def test_layer_index_selection_uses_its_own_block_table(weights, rows, blocks):
    """Short and long selectors use the independent index cache mapping."""
    _check_layer_post_reference(weights, rows, 0, True, blocks=blocks)


def _layer_end_rank(rank, port, release_rank, results):
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer
    from vllm.models.minimax_m3.amd.specialized_decode.mono.step import (
        begin_fence,
        check_diagnostics,
    )

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    device = torch.device("cuda", rank)
    peer = PeerBuffer(256 + 4096 + 64, dist.group.WORLD, rank, 4, device)
    peer.bytes.zero_()
    addresses = peer.addresses + 256
    epoch = torch.zeros(1, dtype=torch.int64, device=device)
    diag = torch.zeros(256, 8, dtype=torch.int32, device=device)
    begin_fence(
        addresses.data_ptr(),
        rank,
        epoch.data_ptr(),
        4096,
        4,
        mode=0,
        stream=torch.cuda.current_stream(),
    )
    check_diagnostics[(1,)](diag, addresses, rank, epoch, TP=4)
    torch.accelerator.synchronize()
    dist.barrier()
    # Rank 2 cannot publish its late diagnostic until rank 0 proves it has
    # reached the fence and cannot pass it. No timing assumption orders this.
    if rank == 2:
        assert release_rank.wait(60)
        diag[0, 0] = 1
    check_diagnostics[(1,)](diag, addresses, rank, epoch, TP=4)
    before, after = torch.cuda.Event(), torch.cuda.Event()
    before.record()
    begin_fence(
        addresses.data_ptr(),
        rank,
        epoch.data_ptr(),
        4096,
        4,
        stream=torch.cuda.current_stream(),
    )
    after.record()
    if rank == 0:
        before.synchronize()
        assert not after.query(), "sampling could overtake a rank's diagnostics"
        release_rank.set()
    after.synchronize()
    results[rank] = peer.bytes[:16].view(torch.int32).tolist()
    dist.barrier()
    peer.close()
    dist.destroy_process_group()


@pytest.mark.distributed(num_gpus=4)
def test_layer_step_end_fence_orders_remote_diagnostics():
    """A rank that finishes early cannot sample before a late peer's fault
    becomes visible through the real TP4 IPC buffers.
    """
    from vllm.utils.network_utils import get_open_port

    with mp.get_context("spawn").Manager() as manager:
        results = manager.dict()
        release_rank = manager.Event()
        ranks = mp.start_processes(
            _layer_end_rank,
            args=(get_open_port(), release_rank, results),
            nprocs=4,
            join=False,
        )
        deadline = time.monotonic() + 120
        while not ranks.join(timeout=10):
            if time.monotonic() > deadline:
                for process in ranks.processes:
                    process.kill()
                pytest.fail("TP4 step-end fence did not finish")
        assert len(results) == 4
        for words in results.values():
            assert words == [1, 2, 0, 2]
