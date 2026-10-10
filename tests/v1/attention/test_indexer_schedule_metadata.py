# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA indexer decode schedule must match DeepGEMM's current SM count.

Dual-batch overlap lowers it (`deep_gemm.set_num_sms`) while microbatches run,
after the metadata builder planned `schedule_metadata` for the whole GPU, and
`fp8_fp4_paged_mqa_logits` asserts the schedule matches the count in effect.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    native_next_n_supported,
    set_num_sms,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.v1.attention.backends.mla.indexer import DeepSeekV32IndexerDecodeMetadata


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_decode_schedule_follows_deepgemm_num_sms(next_n: int):
    if not native_next_n_supported(next_n):
        pytest.skip(f"next_n={next_n} has no native kernel on this architecture")
    torch.manual_seed(0)
    batch_size, heads, head_dim, block_size, max_model_len = 4, 32, 128, 64, 2048
    num_blocks = batch_size * max_model_len // block_size

    q = torch.randn((batch_size, next_n, heads, head_dim), device="cuda")
    weights = torch.randn((batch_size * next_n, heads), device="cuda")
    # Fused FP8 cache: per block, the FP8 values then one fp32 scale per token.
    kv = torch.randn((num_blocks, block_size, head_dim), device="cuda")
    kv_scale = kv.abs().amax(dim=-1, keepdim=True).clamp(1e-4) / 448.0
    kv_cache = torch.cat(
        [
            (kv / kv_scale).to(torch.float8_e4m3fn).view(torch.uint8).flatten(1),
            kv_scale.view(torch.uint8).flatten(1),
        ],
        dim=1,
    ).view(num_blocks, block_size, 1, head_dim + 4)
    block_table = torch.randperm(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, -1
    )
    context_lens = torch.randint(
        next_n, max_model_len, (batch_size, 1), device="cuda", dtype=torch.int32
    )
    seq_lens = context_lens - next_n + 1 + torch.arange(next_n, device="cuda")
    seq_lens = seq_lens.to(torch.int32).contiguous()

    full_sms = get_num_sms()
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        decode_lens=torch.full((batch_size,), next_n, device="cuda"),
        requires_padding=False,
        schedule_metadata=get_paged_mqa_logits_metadata(seq_lens, block_size, full_sms),
        schedule_num_sms=full_sms,
        schedule_block_size=block_size,
    )

    def paged_mqa_logits(schedule_metadata: torch.Tensor) -> torch.Tensor:
        return fp8_fp4_paged_mqa_logits(
            (q.to(torch.float8_e4m3fn), None),
            kv_cache,
            weights,
            seq_lens,
            block_table,
            schedule_metadata,
            max_model_len,
            clean_logits=False,
        )

    assert decode.get_schedule_metadata() is decode.schedule_metadata
    ref_logits = paged_mqa_logits(decode.schedule_metadata)

    # What `create_sm_control_context` does while DBO microbatches run.
    set_num_sms(full_sms - 20)
    try:
        schedule_metadata = decode.get_schedule_metadata()
        assert schedule_metadata is not decode.schedule_metadata
        # Later indexer layers of the same forward reuse it.
        assert decode.get_schedule_metadata() is schedule_metadata
        logits = paged_mqa_logits(schedule_metadata)
    finally:
        set_num_sms(full_sms)
    assert decode.get_schedule_metadata() is decode.schedule_metadata

    # Columns past each row's context length are never written.
    valid = torch.arange(max_model_len, device="cuda") < seq_lens.view(-1, 1)
    torch.testing.assert_close(
        logits.masked_fill(~valid, 0), ref_logits.masked_fill(~valid, 0)
    )
