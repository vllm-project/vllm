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
        # Without the kernel the flag warns and falls back, whatever the arch.
        monkeypatch.setattr(aiter_ops, "_has_aiter_triton_sparse_mla", lambda: False)
        assert not enabled(aiter=True, flag=True)
        assert ("sparse_mla" if on_gfx950() else "gfx950") in warnings.pop()
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
        return rocm.DeepseekV4ROCMAiterSparseSWAMetadataBuilder(
            swa_spec, ["swa"], vllm_config, device
        ).build(0, common)

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
