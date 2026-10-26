# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-only aiter path"
)


def _rows_from_ragged(indices: torch.Tensor, indptr: torch.Tensor) -> list[list[int]]:
    ends = indptr.tolist()
    return [indices[s:e].tolist() for s, e in zip(ends, ends[1:])]


def test_triton_sparse_mla_gate(monkeypatch) -> None:
    import vllm._aiter_ops as aiter_ops
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    warnings: list[str] = []
    monkeypatch.setattr(
        aiter_ops.logger,
        "warning_once",
        lambda msg, *args, **kwargs: warnings.append(msg % args),
    )

    def enabled(aiter: bool, flag: bool) -> bool:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", str(int(aiter)))
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", str(int(flag)))
        rocm_aiter_ops.refresh_env_variables()
        return bool(rocm_aiter_ops.is_triton_sparse_mla_enabled())

    try:
        assert not enabled(aiter=True, flag=False) and not warnings
        assert not enabled(aiter=False, flag=True)
        assert "VLLM_ROCM_USE_AITER is off" in warnings.pop()
        # On gfx950 the flag takes effect; elsewhere it warns and falls back.
        assert enabled(aiter=True, flag=True) == on_gfx950()
        if not on_gfx950():
            assert "gfx950" in warnings.pop()
    finally:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    "rope_dim",
    [0, 64],
    ids=["rope_free", "appended_rope"],  # GLM-5.3-Flash; GLM-5.1/5.2, V3.2
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@torch.inference_mode()
def test_forward_mqa_prepares_triton_sparse_mla_inputs(
    monkeypatch, rope_dim: int, kv_cache_dtype: str
) -> None:
    """What forward_mqa hands aiter: global slots built from the request-local
    top-k rows, q quantized with the layer's scale, and the cache as stored."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    kv_lora_rank, num_heads, topk, block_size = 512, 16, 128, 16
    head_size = kv_lora_rank + rope_dim
    fp8 = kv_cache_dtype == "fp8"

    # A decode row, then a short prefill.
    query_lens, seq_lens = [1, 5], [40, 20]
    num_tokens = sum(query_lens)
    block_table = torch.randperm(8, generator=gen).view(2, 4).to(device, torch.int32)
    topk_indices = torch.full((num_tokens, topk), -1, dtype=torch.int32)
    req_id_per_token: list[int] = []
    rows: list[list[int]] = []
    for req, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        for pos in range(seq_len - query_len, seq_len):
            local = torch.randperm(pos + 1, generator=gen)
            topk_indices[len(rows), : pos + 1] = local.int()
            slots = block_table[req, local // block_size].cpu() * block_size
            rows.append((slots + local % block_size).tolist())
            req_id_per_token.append(req)
    indptr = torch.tensor([0] + [len(row) for row in rows]).cumsum(0)

    impl = ROCMAiterMLASparseImpl.__new__(ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = rope_dim
    impl.scale = head_size**-0.5
    impl.sinks = torch.randn(num_heads, generator=gen).to(device)
    impl.kv_cache_dtype = kv_cache_dtype
    impl.topk_indices_buffer = topk_indices.to(device)
    impl.owns_indexer = True
    impl.index_epoch = 0
    impl.use_aiter_sparse_mla = True
    layer = SimpleNamespace(
        _q_scale=torch.tensor([0.02], device=device),
        _k_scale=torch.tensor([0.01], device=device),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        topk_tokens=topk,
        req_id_per_token=torch.tensor(req_id_per_token, dtype=torch.int32).to(device),
        block_table=block_table,
        block_size=block_size,
        paged_kv_indptr=indptr.to(device, torch.int32),
        paged_kv_indices=torch.zeros(num_tokens * topk, dtype=torch.int32).to(device),
        attn_out_dtype=torch.bfloat16,
    )
    cache_dtype = current_platform.fp8_dtype() if fp8 else torch.bfloat16
    kv_cache = torch.zeros(8, block_size, head_size, device=device).to(cache_dtype)
    # Two rows of cudagraph padding past the real tokens.
    q = torch.randn(num_tokens + 2, num_heads, head_size, generator=gen).to(
        device, torch.bfloat16
    )
    calls = []
    monkeypatch.setattr(
        rocm_aiter_ops, "triton_sparse_mla_fwd", lambda *a, **k: calls.append((a, k))
    )

    out, lse = impl.forward_mqa(q, kv_cache, metadata, layer)

    assert len(calls) == 1
    (q_in, kv_in, o, sm_scale, kv_indptr, kv_indices), kwargs = calls[0]
    assert _rows_from_ragged(kv_indices, kv_indptr) == rows
    assert q_in.shape[0] == num_tokens
    if fp8:
        assert q_in.dtype == cache_dtype
        torch.testing.assert_close(
            q_in.float() * layer._q_scale, q[:num_tokens].float(), atol=1e-3, rtol=0.07
        )
    else:
        assert q_in.data_ptr() == q.data_ptr()
    assert kv_in.shape == (8 * block_size, 1, 1, head_size)
    assert kv_in.data_ptr() == kv_cache.data_ptr()
    assert o.data_ptr() == out.data_ptr() and o.shape[0] == num_tokens
    assert sm_scale == impl.scale
    assert kwargs["kv_lora_rank"] == kv_lora_rank
    assert kwargs["qk_rope_head_dim"] == rope_dim
    assert kwargs["q_scale"] is layer._q_scale
    assert kwargs["kv_scale"] is layer._k_scale
    assert kwargs["attn_sink"] is impl.sinks
    assert lse is None


@pytest.mark.parametrize(
    ("model", "compress_ratio"),
    [("deepseek_v41", 1), ("deepseek_v41", 2), ("deepseek_v4", 4)],
)
@torch.inference_mode()
def test_paged_prefill_indices_match_gathered_prefill(
    monkeypatch, model: str, compress_ratio: int
) -> None:
    """The aiter prefill reads the paged caches through the SWA builder's rows
    and build_prefill_topk_ragged_indices, and must attend exactly the keys the
    dequantize-and-gather prefill does."""
    import importlib

    from tests.v1.attention.utils import create_vllm_config
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_prefill_topk_ragged_indices,
    )
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    rocm = importlib.import_module(f"vllm.models.{model}.amd.rocm")
    builder_cls = (
        rocm.DeepseekV41ROCMAiterSparseSWAMetadataBuilder
        if model == "deepseek_v41"
        else rocm.DeepseekV4ROCMAiterSparseSWAMetadataBuilder
    )
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(compress_ratio)
    window, width, block_size, max_model_len = 16, 32, 16, 128
    comp_block = block_size // compress_ratio
    num_compressed = max_model_len // compress_ratio

    # A decode row, then prefills with and without a prefix.
    prefix_lens, query_lens = [20, 0, 13, 40], [1, 7, 20, 9]
    seq_lens = [p + q for p, q in zip(prefix_lens, query_lens)]
    num_decodes = num_decode_tokens = 1
    num_tokens = sum(query_lens)
    qsl = torch.tensor([0] + query_lens).cumsum(0).to(torch.int32)
    positions = torch.cat([torch.arange(p, s) for p, s in zip(prefix_lens, seq_lens)])
    token_to_req = torch.repeat_interleave(
        torch.arange(len(query_lens)), torch.tensor(query_lens)
    )
    swa_bt = torch.randperm(32, generator=gen)[:16].view(4, 4).to(device, torch.int32)
    comp_bt = torch.randperm(32, generator=gen)[:16].view(4, 4).to(device, torch.int32)

    vllm_config = create_vllm_config(
        model_name="facebook/opt-125m",
        max_model_len=max_model_len,
        block_size=block_size,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        hf_config_override={
            "compress_ratios": [compress_ratio],
            "index_topk": width,
            "sliding_window": window,
        },
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=window,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    slots = swa_bt.cpu()[token_to_req, positions // block_size] * block_size + (
        positions % block_size
    )
    common = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32).to(device),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=swa_bt,
        slot_mapping=slots.to(device, torch.int64),
        positions=positions.to(device),
        causal=True,
    )

    def build(enabled: bool):
        monkeypatch.setattr(
            rocm_aiter_ops, "is_triton_sparse_mla_enabled", lambda: enabled
        )
        return builder_cls(swa_spec, ["swa"], vllm_config, device).build(0, common)

    assert build(enabled=False).prefill_swa_ragged_indices is None
    md = build(enabled=True)
    assert md.num_decode_tokens == num_decode_tokens

    # Local top-k rows. Under each row's causal cap sit a failed candidate and
    # one past the pool; past the cap, stale values neither path may read.
    num_prefill_tokens = num_tokens - num_decode_tokens
    topk = torch.full((num_prefill_tokens, width), -1, dtype=torch.int32)
    for row in range(num_prefill_tokens):
        visible = (int(positions[row + num_decode_tokens]) + 1) // compress_ratio
        cap = min(visible, width)
        topk[row, :cap] = torch.randperm(visible, generator=gen)[:cap].int()
        if cap >= 2:
            topk[row, :2] = torch.tensor([-1, num_compressed + 3])
        topk[row, cap:] = torch.randint(
            0, num_compressed, (width - cap,), generator=gen
        ).int()
    topk = topk.to(device)

    topk_indices, topk_indptr = build_prefill_topk_ragged_indices(
        topk,
        md.token_to_req_indices,
        md.query_start_loc,
        md.seq_lens,
        md.is_valid_token,
        comp_bt,
        block_size=comp_block,
        compress_ratio=compress_ratio,
        num_compressed=num_compressed,
        token_offset=num_decode_tokens,
    )
    paged_keys = [
        sorted([("s", x) for x in swa if x >= 0] + [("c", x) for x in top if x >= 0])
        for swa, top in zip(
            _rows_from_ragged(
                md.prefill_swa_ragged_indices, md.prefill_swa_ragged_indptr
            ),
            _rows_from_ragged(topk_indices, topk_indptr),
        )
    ]

    # The gathered path's combine, mapped from workspace offsets to cache slots.
    M = num_compressed + window + num_tokens
    combined, combined_lens = rocm.combine_topk_swa_indices(
        topk,
        md.query_start_loc[num_decodes:],
        md.prefill_seq_lens,
        md.prefill_gather_lens,
        window,
        compress_ratio,
        width,
        M,
        num_compressed,
    )
    gather_start = (md.prefill_seq_lens - md.prefill_gather_lens).tolist()
    swa_bt_cpu, comp_bt_cpu = swa_bt.cpu(), comp_bt.cpu()

    def slot(block_table, req, pos, size):
        return int(block_table[req, pos // size]) * size + pos % size

    gathered_keys = []
    for row, (values, n) in enumerate(zip(combined.tolist(), combined_lens.tolist())):
        req = int(token_to_req[row + num_decode_tokens])
        chunk_req = req - num_decodes
        keys = []
        for value in values[:n]:
            if value < 0:
                continue
            offset = value - chunk_req * M
            if offset < num_compressed:
                keys.append(("c", slot(comp_bt_cpu, req, offset, comp_block)))
            else:
                pos = gather_start[chunk_req] + offset - num_compressed
                keys.append(("s", slot(swa_bt_cpu, req, pos, block_size)))
        gathered_keys.append(sorted(keys))

    assert paged_keys == gathered_keys


@pytest.mark.parametrize("model", ["deepseek_v4", "deepseek_v41"])
def test_aiter_sparse_mla_stays_off_with_kv_connector(monkeypatch, model) -> None:
    import importlib

    from vllm._aiter_ops import rocm_aiter_ops

    rocm = importlib.import_module(f"vllm.models.{model}.amd.rocm")
    monkeypatch.setattr(rocm_aiter_ops, "is_triton_sparse_mla_enabled", lambda: True)
    assert rocm._aiter_sparse_mla_enabled(has_kv_transfer=False)
    assert not rocm._aiter_sparse_mla_enabled(has_kv_transfer=True)


@pytest.mark.parametrize("wo_a_fp8", [False, True])
@pytest.mark.parametrize(
    "num_heads", [64, 128], ids=["aiter", "aiter_decode_opus_prefill"]
)
@torch.inference_mode()
def test_deepseek_v4_forward_mqa_matches_reference(
    monkeypatch,
    workspace_init,
    num_heads: int,
    wo_a_fp8: bool,
) -> None:
    """forward_mqa on a mixed decode + prefill batch matches a torch reference
    whichever kernels serve it: each row comes out inverse-RoPE'd exactly once,
    by the store of the kernel that wrote it or by the standalone pass. With an
    fp8 wo_a no row is rotated here (inverse_rope_group_quant does it later)."""
    import vllm.models.deepseek_v4.amd.rocm as rocm
    from tests.v1.attention.utils import create_vllm_config
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )
    from vllm.platforms.rocm import on_gfx950
    from vllm.utils.torch_utils import set_random_seed
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    if not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    device = torch.device("cuda")
    set_random_seed(0)
    head_dim, nope_dim, rope_dim = 512, 448, 64
    window, block_size, max_model_len = 128, 64, 1024

    # Two decode rows, then prefills with and without a prefix.
    prefix_lens, query_lens = [300, 700, 100, 0], [1, 1, 40, 50]
    seq_lens = [p + q for p, q in zip(prefix_lens, query_lens)]
    num_reqs, num_tokens = len(query_lens), sum(query_lens)
    num_blocks = num_reqs * max_model_len // block_size
    block_table = torch.randperm(num_blocks, device=device).view(num_reqs, -1).int()
    positions = torch.cat([torch.arange(p, s) for p, s in zip(prefix_lens, seq_lens)])
    token_to_req = torch.repeat_interleave(
        torch.arange(num_reqs), torch.tensor(query_lens)
    )
    positions = positions.to(device)

    def slots(req, pos):
        block = block_table[req, pos // block_size].long()
        return block * block_size + pos % block_size

    # The SWA cache holds every token so far, this step's included.
    swa_cache = torch.zeros(
        (num_blocks, block_size, 584), dtype=torch.uint8, device=device
    )
    for req, seq_len in enumerate(seq_lens):
        k = torch.randn(seq_len, head_dim, device=device) * 0.5
        pos = torch.arange(seq_len, device=device)
        quantize_and_insert_k_cache(
            k.bfloat16(), swa_cache, slots(req, pos), block_size=block_size
        )

    vllm_config = create_vllm_config(
        model_name="facebook/opt-125m",
        max_model_len=max_model_len,
        block_size=block_size,
        max_num_seqs=num_reqs,
        max_num_batched_tokens=256,
        hf_config_override={"compress_ratios": [1], "sliding_window": window},
    )
    vllm_config.model_config.model_arch_config.total_num_attention_heads = num_heads
    swa_spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_dim,
        dtype=torch.bfloat16,
        sliding_window=window,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
    )
    qsl = torch.tensor([0] + query_lens).cumsum(0).to(torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens_t.to(device),
        seq_lens_cpu_upper_bound=seq_lens_t,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=slots(token_to_req.to(device), positions),
        positions=positions,
        causal=True,
    )
    monkeypatch.setattr(rocm_aiter_ops, "is_triton_sparse_mla_enabled", lambda: True)
    swa_builder = rocm.DeepseekV4ROCMAiterSparseSWAMetadataBuilder(
        swa_spec, ["swa"], vllm_config, device
    )
    swa_metadata = swa_builder.build(0, common)
    monkeypatch.setattr(
        rocm,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"swa": swa_metadata}),
    )

    inv_freq = 1.0 / 10000 ** (
        torch.arange(0, rope_dim, 2, device=device).float() / rope_dim
    )
    freqs = torch.outer(torch.arange(max_model_len, device=device).float(), inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    # An SWA-only layer, with the attributes the decode and prefill paths read.
    attn = rocm.DeepseekV4ROCMAiterMLAAttention.__new__(
        rocm.DeepseekV4ROCMAiterMLAAttention
    )
    torch.nn.Module.__init__(attn)
    attn.prefix = "attn"
    attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=swa_cache)
    attn.kv_cache = None
    attn.compress_ratio = 1
    attn.window_size = window
    attn.max_image_tokens = 0
    attn.topk_indices_buffer = torch.full(
        (num_tokens, 1), -1, dtype=torch.int32, device=device
    )
    attn.n_local_heads = num_heads
    attn.head_dim, attn.nope_head_dim, attn.rope_head_dim = head_dim, nope_dim, rope_dim
    attn.scale = head_dim**-0.5
    attn.attn_sink = torch.randn(num_heads, device=device)
    attn.kv_cache_dtype = "fp8_ds_mla"
    attn._has_kv_transfer = False
    attn.rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin_cache)
    attn._wo_a_fp8_weight = torch.empty(0) if wo_a_fp8 else None
    attn._use_aiter_sparse_mla_decode = rocm._aiter_sparse_mla_enabled(False)
    attn._use_aiter_sparse_mla_prefill = swa_builder._use_aiter_sparse_mla_prefill

    q = (torch.randn(num_tokens, num_heads, head_dim, device=device) * 0.5).bfloat16()
    output = torch.empty_like(q)
    attn.forward_mqa(q, None, positions, output)

    # Reference: attention with the sink over each token's window of the
    # dequantized cache, then the inverse GPT-J RoPE on the last 64 lanes.
    keys = torch.empty(
        num_reqs, max(seq_lens), head_dim, dtype=torch.bfloat16, device=device
    )
    dequantize_and_gather_k_cache(
        keys,
        swa_cache,
        seq_lens=seq_lens_t.to(device),
        gather_lens=None,
        block_table=block_table,
        block_size=block_size,
        offset=0,
        use_fnuz=current_platform.is_fp8_fnuz(),
    )
    expected = torch.empty(num_tokens, num_heads, head_dim, device=device)
    for row in range(num_tokens):
        pos = int(positions[row])
        k = keys[token_to_req[row], max(0, pos - window + 1) : pos + 1].float()
        scores = q[row].float() @ k.T * attn.scale
        m = torch.maximum(scores.amax(-1), attn.attn_sink)
        p = torch.exp(scores - m[:, None])
        expected[row] = p @ k / (p.sum(-1) + torch.exp(attn.attn_sink - m))[:, None]
    if not wo_a_fp8:
        cos, sin = cos_sin_cache[positions].unsqueeze(1).chunk(2, dim=-1)
        even = expected[..., nope_dim::2].clone()
        odd = expected[..., nope_dim + 1 :: 2].clone()
        expected[..., nope_dim::2] = even * cos + odd * sin
        expected[..., nope_dim + 1 :: 2] = odd * cos - even * sin

    # Per row, so a single unrotated or twice-rotated row cannot hide.
    err = (output.float() - expected).flatten(1).norm(dim=1)
    rel = err / expected.flatten(1).norm(dim=1)
    assert rel.max() < 1e-2, rel


@pytest.mark.parametrize("output", ["bf16", "mxfp8"])
@torch.inference_mode()
def test_sparse_mla_store_epilogue_matches_standalone_passes(output: str) -> None:
    """The inverse RoPE (and MXFP8 quantization) aiter applies in its store
    matches running the kernel without it and then the standalone passes it
    replaces: rocm_inverse_rope_rows_ / rocm_inverse_rope_mxfp8_rows."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )
    from vllm.platforms.rocm import on_gfx950
    from vllm.utils.torch_utils import set_random_seed
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_inverse_rope_mxfp8_rows,
        rocm_inverse_rope_rows_,
    )

    if not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    device = torch.device("cuda")
    set_random_seed(0)
    num_tokens, num_heads, head_dim, rope_dim, block_size = 10, 64, 512, 64, 64

    def fp8_ds_mla_cache(num_rows: int) -> torch.Tensor:
        cache = torch.zeros(
            (num_rows // block_size, block_size, 584), dtype=torch.uint8, device=device
        )
        k = torch.randn(num_rows, head_dim, device=device) * 0.5
        slots = torch.arange(num_rows, device=device)
        quantize_and_insert_k_cache(k.bfloat16(), cache, slots, block_size=block_size)
        return cache

    def ragged(keys_per_row: int, num_rows: int):
        indices = torch.cat(
            [
                torch.randperm(num_rows, device=device)[:keys_per_row]
                for _ in range(num_tokens)
            ]
        )
        indptr = torch.arange(0, (num_tokens + 1) * keys_per_row, keys_per_row)
        return indptr.to(device, torch.int32), indices.int()

    swa_cache, compressed_cache = fp8_ds_mla_cache(256), fp8_ds_mla_cache(128)
    swa_indptr, swa_indices = ragged(32, 256)
    topk_indptr, topk_indices = ragged(48, 128)
    q = (torch.randn(num_tokens, num_heads, head_dim, device=device) * 0.5).bfloat16()
    sink = torch.randn(num_heads, device=device)
    positions = torch.randint(0, 4096, (num_tokens,), device=device)
    inv_freq = 1.0 / 10000 ** (
        torch.arange(0, rope_dim, 2, device=device).float() / rope_dim
    )
    freqs = torch.outer(torch.arange(4096, device=device).float(), inv_freq)
    cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    def sparse_mla(o, rotate: bool, out_mxfp8=None) -> None:
        rocm_aiter_ops.triton_sparse_mla_fwd(
            q,
            swa_cache,
            o,
            head_dim**-0.5,
            swa_indptr,
            swa_indices,
            attn_sink=sink,
            extra_kv_buffer=compressed_cache,
            extra_kv_indptr=topk_indptr,
            extra_kv_indices=topk_indices,
            inv_rope_positions=positions if rotate else None,
            inv_rope_cos_sin_cache=cos_sin_cache if rotate else None,
            out_mxfp8=out_mxfp8,
        )

    unrotated = torch.empty_like(q)
    sparse_mla(unrotated, rotate=False)
    expected = unrotated.clone()
    rocm_inverse_rope_rows_(expected, positions, cos_sin_cache, rope_dim)

    if output == "bf16":
        fused = torch.empty_like(q)
        sparse_mla(fused, rotate=True)
        torch.testing.assert_close(fused, expected)
        return

    def mxfp8_buffers():
        return (
            torch.empty(
                num_tokens,
                num_heads * head_dim,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.empty(
                num_tokens, num_heads * head_dim // 32, dtype=torch.uint8, device=device
            ),
        )

    def dequantize(data, scale):
        blocks = data.float().view(num_tokens, -1, 32)
        blocks = blocks * torch.exp2(scale.float() - 127).unsqueeze(-1)
        return blocks.view_as(q)

    fused_mxfp8 = mxfp8_buffers()
    sparse_mla(None, rotate=True, out_mxfp8=fused_mxfp8)
    standalone_mxfp8 = mxfp8_buffers()
    rocm_inverse_rope_mxfp8_rows(
        unrotated, positions, cos_sin_cache, rope_dim, *standalone_mxfp8
    )
    # MXFP8 rounding alone puts both about 0.03 rel-L2 off the bf16 rows
    for data, scale in (fused_mxfp8, standalone_mxfp8):
        err = (dequantize(data, scale) - expected.float()).norm()
        assert err / expected.float().norm() < 0.05
