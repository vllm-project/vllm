# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.platforms import current_platform

FLASHINFER_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024

if not current_platform.is_cuda() or not current_platform.has_device_capability(90):
    pytest.skip(
        reason="FlashInfer MLA requires CUDA compute capability 9.0 or above.",
        allow_module_level=True,
    )
else:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

requires_sm90 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(90),
    reason="This test requires an SM90 GPU.",
)
requires_sm10x = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="This test requires an SM10x GPU.",
)

# Deepseek R1 MLA config.
NUM_HEADS = 128
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
SCALE = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5


def _make_decode_inputs(bs: int, block_size: int, dtype: torch.dtype):
    """Build valid trtllm MLA decode inputs on the current CUDA device."""
    max_seq_len_cap = 1024
    seq_lens = [torch.randint(2, max_seq_len_cap, (1,)).item() for _ in range(bs)]
    seq_lens[-1] = max_seq_len_cap
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    # Generate block tables with random but unique block IDs
    # From https://github.com/flashinfer-ai/flashinfer/pull/1222
    blocks_per_seq = (seq_lens_tensor + block_size - 1) // block_size
    max_num_blocks_per_seq = max(blocks_per_seq.max().item(), 4)
    total_blocks_needed = int(sum(blocks_per_seq))
    all_block_ids = torch.randperm(total_blocks_needed)

    block_tables = torch.zeros((bs, max_num_blocks_per_seq), dtype=torch.int32)
    block_id = 0
    for i in range(bs):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    kv_cache = torch.randn(block_tables.numel(), block_size, QK_HEAD_DIM).to(dtype)
    q = torch.randn(bs, NUM_HEADS, QK_HEAD_DIM).to(dtype)
    return q, kv_cache, block_tables, seq_lens_tensor, max_seq_len


def ref_mla(
    out: Tensor,  # (bs, num_heads, v_head_dim)
    query: Tensor,  # (bs, num_heads, head_dim)
    kv_cache: Tensor,  # (num_blocks, block_size, head_dim)
    scale: float,
    block_tables: Tensor,  # (bs, max_num_blocks)
    seq_lens: Tensor,  # (bs,)
):
    bs, num_heads, v_head_dim = out.shape
    head_dim = query.shape[2]

    for i in range(bs):
        # gather and flatten KV-cache
        kv = kv_cache[block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(1, -1, head_dim)[:, : seq_lens[i]]  # (1, seq_len, head_dim)
        v = kv[:, :, :v_head_dim]

        q = query[i].view(num_heads, 1, head_dim)
        o = F.scaled_dot_product_attention(q, kv, v, scale=scale, enable_gqa=True)
        out[i] = o.view(num_heads, v_head_dim)

    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bs", [1, 2, 4, 16])
@pytest.mark.parametrize("block_size", [32, 64])
@requires_sm10x
def test_flashinfer_mla_decode(dtype: torch.dtype, bs: int, block_size: int):
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    q, kv_cache, block_tables, seq_lens_tensor, max_seq_len = _make_decode_inputs(
        bs, block_size, dtype
    )

    out_ref = q.new_zeros(bs, NUM_HEADS, KV_LORA_RANK)
    ref_mla(out_ref, q, kv_cache, SCALE, block_tables, seq_lens_tensor)

    workspace_buffer = torch.zeros(
        FLASHINFER_WORKSPACE_BUFFER_SIZE,
        dtype=torch.uint8,
        device=q.device,
    )
    # Flashinfer MLA expects the query to be of shape
    # (bs, q_len_per_request, num_heads, qk_head_dim),
    # where q_len_per_request is the MTP query length (=1 without MTP)
    q = q.unsqueeze(1)

    out_ans = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=SCALE,
    )
    out_ans = out_ans.squeeze(1)
    torch.testing.assert_close(out_ans, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("mode", ["tp", "dcp", "hisparse"])
@requires_sm10x
def test_flashinfer_trtllm_sparse_mla_decode_without_rope(dtype, mode, monkeypatch):
    """TP skips preparation; empty DCP contributions and HiSparse outputs vanish."""
    import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as backend
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

    torch.set_default_device("cuda")
    torch.manual_seed(42)
    batch_size, num_heads, block_size, num_blocks, sparse_topk = 3, 8, 64, 4, 2176
    valid_lens = [1 if mode == "tp" else 0, 17, 73]
    counts = torch.tensor(valid_lens, dtype=torch.int32)
    query = torch.randn(batch_size, 1, num_heads, KV_LORA_RANK).to(dtype)
    kv_cache = torch.randn(num_blocks, block_size, KV_LORA_RANK).to(dtype)
    num_slots = num_blocks * block_size
    indices = torch.full((batch_size, sparse_topk), -1, dtype=torch.int32)
    for token, length in enumerate(valid_lens):
        indices[token, :length] = torch.randperm(num_slots)[:length].int()
    empty = counts == 0
    reference_indices = indices.clone()
    reference_indices[:, 0].masked_fill_(empty, 0)
    workspace = torch.empty(FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.int8)

    impl = object.__new__(backend.FlashInferMLASparseImpl)
    impl.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = 0
    impl.index_group = (
        object.__new__(HiSparseMLAIndexGroup) if mode == "hisparse" else None
    )
    impl._workspace_buffer = workspace
    impl.bmm1_scale = QK_NOPE_HEAD_DIM**-0.5
    impl.bmm2_scale = 1.0
    impl.is_nope_mla = True
    impl.need_to_return_lse_for_decode = mode == "dcp"
    if mode == "tp":

        def unexpected_helper(*args, **kwargs):
            pytest.fail("TP must not launch empty-query preparation")

        monkeypatch.setattr(
            backend, "prepare_sparse_mla_safe_lengths", unexpected_helper
        )

    out, lse = impl._run_mqa_kernel(query.squeeze(1), kv_cache, indices, counts)
    reference = trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=0,
        block_tables=reference_indices.unsqueeze(1),
        seq_lens=counts,
        max_seq_len=sparse_topk,
        sparse_mla_top_k=sparse_topk,
        sparse_mla_top_k_lens=counts.clamp(min=1),
        bmm1_scale=QK_NOPE_HEAD_DIM**-0.5,
        bmm2_scale=1.0,
        return_lse=mode == "dcp",
    )
    if mode == "dcp":
        reference, reference_lse = reference
        reference_lse = reference_lse.reshape(batch_size, num_heads)
        reference_lse.masked_fill_(empty[:, None], float("-inf"))
        torch.testing.assert_close(lse, reference_lse, atol=0, rtol=0)

        from vllm.v1.attention.ops.dcp import CPTritonContext, correct_attn_out

        # The peer owns only our empty selections. Check our contribution after
        # the production merge, without zeroing the local attention output.
        peer_lse = torch.full_like(lse, float("-inf")).masked_fill(empty[:, None], 0)
        out, _ = correct_attn_out(
            out, torch.stack((lse, peer_lse)), cp_rank=0, ctx=CPTritonContext()
        )
    else:
        assert lse is None
    reference = reference.squeeze(1).masked_fill(empty[:, None, None], 0)
    torch.testing.assert_close(out, reference, atol=0, rtol=0)

    if dtype == torch.bfloat16:
        flat_cache = kv_cache.view(num_slots, KV_LORA_RANK).float()
        refs = []
        for token, length in enumerate(valid_lens):
            selected_kv = flat_cache[reference_indices[token, :length].long()]
            scores = torch.einsum("hd,kd->hk", query[token, 0].float(), selected_kv)
            probs = torch.softmax(scores * QK_NOPE_HEAD_DIM**-0.5, dim=-1)
            refs.append(torch.einsum("hk,kd->hd", probs, selected_kv))
        ref = torch.stack(refs).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@requires_sm10x
def test_sparse_mla_safe_lengths():
    from vllm.v1.attention.backends.mla.sparse_utils import (
        prepare_sparse_mla_safe_lengths,
    )

    device = "cuda"
    num_tokens = 259
    # Include a partial tile and expose accidental writes into adjacent storage.
    counts_storage = torch.full((num_tokens, 2), 17, dtype=torch.int32, device=device)
    counts = counts_storage[:, 0]
    counts[::3] = 0
    original_counts = counts_storage.clone()
    empty = counts == 0
    indices_storage = torch.full(
        (num_tokens, 256), 42, dtype=torch.int32, device=device
    )
    indices = indices_storage[:, ::2]
    expected_indices = indices_storage.clone()
    expected_indices[:, 0].masked_fill_(empty, 0)
    safe = prepare_sparse_mla_safe_lengths(indices, counts)
    torch.testing.assert_close(safe, counts.clamp(min=1), rtol=0, atol=0)
    torch.testing.assert_close(indices_storage, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(counts_storage, original_counts, rtol=0, atol=0)


@requires_sm90
def test_flashinfer_sm90_fp8_mla_decode_without_rope():
    """Hopper FA3 supports BF16 queries over an FP8 cache without KPE."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    batch_size = 2
    num_heads = 16
    page_size = 16
    num_pages = 6

    q_nope = torch.randn(
        batch_size,
        num_heads,
        KV_LORA_RANK,
        dtype=torch.bfloat16,
        device=device,
    )
    q_pe = torch.empty(
        batch_size,
        num_heads,
        0,
        dtype=torch.bfloat16,
        device=device,
    )

    ckv = torch.randn(
        num_pages,
        page_size,
        KV_LORA_RANK,
        device=device,
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    ckv_scale = ckv.abs().max().item() / fp8_max
    ckv_fp8 = (ckv / ckv_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    scale_bf16 = torch.tensor(ckv_scale, dtype=torch.bfloat16, device=device)
    ckv_ref = ckv_fp8.to(torch.bfloat16) * scale_bf16
    kpe_fp8 = torch.empty(
        num_pages,
        page_size,
        0,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    kpe_ref = torch.empty(
        num_pages,
        page_size,
        0,
        dtype=torch.bfloat16,
        device=device,
    )

    qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([4, 1, 3, 0, 5], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([45, 29], dtype=torch.int32, device=device)
    sm_scale = QK_NOPE_HEAD_DIM**-0.5

    def run(
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        workspace = torch.empty(
            FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa3")
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            num_heads,
            KV_LORA_RANK,
            0,
            page_size,
            False,
            sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=ckv_cache.dtype,
        )
        return wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, **kwargs)

    out_ref = run(ckv_ref, kpe_ref)
    out = run(ckv_fp8, kpe_fp8, ckv_scale=ckv_scale, kpe_scale=1.0)
    torch.testing.assert_close(out, out_ref, atol=2e-2, rtol=2e-2)


@requires_sm10x
def test_flashinfer_mla_decode_workspace_supports_autotune():
    """VLLM's FlashInfer MLA decode workspace must be int8 for autotuning.

    Model Runner V2's warmup autotunes ``trtllm_batch_decode_mla``, which makes
    the FlashInfer autotuner enumerate the CuteDSL tactic. That tactic asserts
    ``workspace_buffer.dtype == torch.int8``; the trtllm-gen path (used for
    normal, non-autotuned inference) instead views the buffer as uint8, so a
    uint8 workspace only fails once the autotuner tries CuteDSL. That regressed
    every DeepSeek MLA test on Blackwell under V2 with
    ``workspace_buffer must be torch.int8`` (vllm-project/vllm#46646).
    """
    from flashinfer.autotuner import autotune

    from vllm.v1.attention.backends.mla.flashinfer_mla import _get_workspace_buffer

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    workspace_buffer = _get_workspace_buffer(return_lse=False)
    assert workspace_buffer.dtype == torch.int8

    q, kv_cache, block_tables, seq_lens_tensor, max_seq_len = _make_decode_inputs(
        bs=1, block_size=64, dtype=torch.bfloat16
    )

    # Under the autotuner the CuteDSL tactic is instantiated with our workspace;
    # a uint8 buffer raises AssertionError here, an int8 buffer succeeds.
    with torch.inference_mode(), autotune(True):
        trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_seq_len=max_seq_len,
            bmm1_scale=SCALE,
        )
