# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity: tokenspeed_mla_decode vs flashinfer trtllm_batch_decode_with_kv_cache_mla."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="tokenspeed_mla / TRT-LLM MLA decode require Blackwell (SM100+).",
        allow_module_level=True,
    )

try:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
except ImportError:
    pytest.skip(reason="flashinfer not installed", allow_module_level=True)

try:
    from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode
except ImportError:
    pytest.skip(reason="tokenspeed_mla not installed", allow_module_level=True)


FLASHINFER_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_TS_MAX_Q_LEN = 8


def _ts_workspace(device, num_heads, kv_lora_rank):
    needed = get_num_sm(device) * num_heads * _TS_MAX_Q_LEN * (kv_lora_rank + 1) * 4
    return torch.empty(needed, dtype=torch.int8, device=device)


@pytest.mark.parametrize("bs", [1, 2, 4, 16])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("q_len_per_request", [1, 2, 4])
def test_tokenspeed_vs_trtllm_decode(bs: int, block_size: int, q_len_per_request: int):
    """Match tokenspeed_mla_decode against TRT-LLM batch decode MLA.

    Both kernels consume the same FP8 KV cache, paged block table, and
    seq_lens. The only structural difference is rank: TRT-LLM expects 4D
    (`unsqueeze(1)` for the kv-head dim) while tokenspeed expects 3D. We
    pass each kernel its preferred shape from the same underlying tensor.
    """
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    # Deepseek R1 dims — both kernels are R1-shape-specialized.
    num_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

    MAX_SEQ_LEN = 1024

    seq_lens = [torch.randint(2, MAX_SEQ_LEN, (1,)).item() for _ in range(bs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)

    blocks_per_seq = (seq_lens_tensor + block_size - 1) // block_size
    max_num_blocks_per_seq = max(blocks_per_seq.max().item(), 4)
    total_blocks_needed = sum(blocks_per_seq).item()
    all_block_ids = torch.randperm(total_blocks_needed, dtype=torch.int32)

    block_tables = torch.zeros((bs, max_num_blocks_per_seq), dtype=torch.int32)
    block_id = 0
    for i in range(bs):
        n = blocks_per_seq[i].item()
        block_tables[i, :n] = all_block_ids[block_id : block_id + n]
        block_id += n

    # KV cache: build in BF16 then cast once to FP8 so both kernels see the
    # exact same quantized values. Shape (num_blocks, block_size, qk_head_dim).
    kv_cache_bf16 = torch.randn(
        block_tables.numel(), block_size, qk_head_dim, dtype=torch.bfloat16
    )
    kv_cache = kv_cache_bf16.to(torch.float8_e4m3fn)

    # Query: (bs, q_len_per_request, num_heads, qk_head_dim) — same layout as
    # FlashInferMLAImpl.forward_mqa. Cast to FP8 to match KV.
    q = torch.randn(
        bs, q_len_per_request, num_heads, qk_head_dim, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)

    # --- TRT-LLM reference ---
    fi_workspace = torch.zeros(FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8)
    out_ref = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=fi_workspace,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale,
    )

    # --- TokenSpeed candidate ---
    ts_workspace = _ts_workspace(q.device, num_heads, kv_lora_rank)
    out_ts = tokenspeed_mla_decode(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ts_workspace,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        softmax_scale=scale,
    )

    # Both kernels output v_head_dim=kv_lora_rank=512 per head.
    # Output dtypes can differ; compare in float32.
    out_ref_f = out_ref.to(torch.float32)
    out_ts_f = out_ts.to(torch.float32)
    assert out_ref_f.shape == out_ts_f.shape, (
        f"shape mismatch: trtllm={tuple(out_ref_f.shape)} "
        f"tokenspeed={tuple(out_ts_f.shape)}"
    )

    torch.testing.assert_close(out_ts_f, out_ref_f, atol=2e-2, rtol=2e-2)
