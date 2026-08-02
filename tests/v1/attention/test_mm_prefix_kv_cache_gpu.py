# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 mm_prefix through the metadata builder and the paged KV-cache path.

The pooling deployment that validated this feature runs with
``disable_pooling_kv_cache=True``, so the ``flash_attn_varlen_func`` call that
takes a ``block_table`` has never run with a ``mask_mod`` attached. This test
drives the real ``FlashAttentionMetadataBuilder`` over a batch that mixes a
context-carrying prefill chunk with decode rows, which is also the only place
where ``seq_lens_cpu_upper_bound`` and the packed query offsets interact.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FA4 mm_prefix is CUDA only", allow_module_level=True)

from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.v1.attention.backends.fa_utils import (  # noqa: E402
    is_fa_version_supported,
)

if not is_fa_version_supported(4):
    pytest.skip("FA4 not supported on this device", allow_module_level=True)

from tests.v1.attention.test_attention_backends import (  # noqa: E402
    create_and_prepopulate_kv_cache,
    run_attention_backend,
)
from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402

# The production Gemma4 checkpoint: text_config.use_bidirectional_attention is
# "vision", so is_mm_prefix_lm is True and the builder allocates its buffers.
# Only config.json / tokenizer are read; weights are never loaded.
MODEL = "google/gemma-4-E2B-it"

BLOCK_SIZE = 16
SLIDING_WINDOW = 256

# Request 0 is a prefill chunk whose range starts mid-context; requests 1-3 are
# decodes whose generated token sits outside every range.
BATCH = BatchSpec(
    seq_lens=[352, 513, 200, 97],
    query_lens=[96, 1, 1, 1],
    name="mm_prefix_mixed",
)
MM_RANGES = {
    0: [(224, 287), (300, 351)],
    1: [(4, 259)],
    2: [(0, 63), (80, 143)],
    3: [(1, 64)],
}


def _reference(
    q,
    k_fulls,
    v_fulls,
    query_lens,
    seq_lens,
    mm_ranges,
    sliding_window,
    scale,
    num_heads,
    num_kv_heads,
    device,
):
    out = torch.empty_like(q, dtype=torch.float32)
    q_off = 0
    for req_idx, (q_len, k_len) in enumerate(zip(query_lens, seq_lens)):
        ctx = k_len - q_len
        q_pos = torch.arange(q_len, device=device) + ctx
        k_pos = torch.arange(k_len, device=device)
        delta = q_pos[:, None] - k_pos[None, :]

        keep = (delta >= 0) & (delta < sliding_window)
        for start, end in mm_ranges.get(req_idx, []):
            q_in = (q_pos >= start) & (q_pos <= end)
            k_in = (k_pos >= start) & (k_pos <= end)
            keep |= q_in[:, None] & k_in[None, :]

        q_i = q[q_off : q_off + q_len].float().transpose(0, 1)
        k_i = k_fulls[req_idx].float().transpose(0, 1)
        v_i = v_fulls[req_idx].float().transpose(0, 1)
        repeats = num_heads // num_kv_heads
        if repeats > 1:
            k_i = k_i.repeat_interleave(repeats, dim=0)
            v_i = v_i.repeat_interleave(repeats, dim=0)

        scores = (q_i @ k_i.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~keep[None], float("-inf"))
        out[q_off : q_off + q_len] = (scores.softmax(-1) @ v_i).transpose(0, 1)
        q_off += q_len
    return out


def test_decode_only_batch_reports_no_ranges():
    """Decode-only steps must carry no mm_prefix metadata at all.

    This is what makes FULL CUDA graph capture consistent for this feature.
    Capture runs through ``_dummy_run`` with no multimodal requests, so no
    mask_mod is attached to the captured graph; a decode-only replay must
    therefore also want no mask_mod, or the captured graph would silently drop
    it. Indexing by query token gives that for free, because a generated token
    is always past every prompt range.

    The range-id scheme keyed off the request having ranges at all, so it
    attached a mask_mod to decode steps and did not have this property.
    """
    vllm_config = create_vllm_config(
        model_name=MODEL,
        max_model_len=1024,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=256,
    )
    device = torch.device("cuda:0")
    decode_batch = BatchSpec(
        seq_lens=[513, 200, 97], query_lens=[1, 1, 1], name="decode_only"
    )
    common = create_common_attn_metadata(decode_batch, BLOCK_SIZE, device)
    common.mm_req_doc_ranges = {0: [(4, 259)], 1: [(0, 63)], 2: [(1, 64)]}

    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

    builder = FlashAttentionMetadataBuilder(
        create_standard_kv_cache_spec(vllm_config),
        ["model.layers.0.self_attn.attn"],
        vllm_config,
        device,
    )
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert md.mm_prefix_query_range_tensor is None


@pytest.mark.parametrize("with_mm_ranges", [True, False])
def test_mm_prefix_kv_cache_path(with_mm_ranges):
    """FA4 + mask_mod + block_table matches a dense reference.

    Also run with the ranges removed so the two references differ; that guards
    against the mask_mod being silently skipped, which would make the
    mm_prefix assertion pass for the wrong reason.
    """
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    vllm_config = create_vllm_config(
        model_name=MODEL,
        max_model_len=max(BATCH.seq_lens),
        block_size=BLOCK_SIZE,
        num_gpu_blocks=2048,
    )
    assert vllm_config.model_config.is_mm_prefix_lm
    assert vllm_config.attention_config.flash_attn_version == 4

    mc = vllm_config.model_config
    num_heads = mc.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = mc.get_num_kv_heads(vllm_config.parallel_config)
    head_size = mc.get_head_size()
    dtype = torch.bfloat16
    scale = head_size**-0.5

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    common = create_common_attn_metadata(BATCH, BLOCK_SIZE, device)
    common.mm_req_doc_ranges = MM_RANGES if with_mm_ranges else {}

    qs, new_ks, new_vs, k_fulls, v_fulls, k_ctxs, v_ctxs = [], [], [], [], [], [], []
    for q_len, s_len in zip(BATCH.query_lens, BATCH.seq_lens):
        ctx = s_len - q_len
        qs.append(torch.randn(q_len, num_heads, head_size, dtype=dtype, device=device))
        k_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        k_fulls.append(k_full)
        v_fulls.append(v_full)
        k_ctxs.append(k_full[:ctx])
        v_ctxs.append(v_full[:ctx])
        new_ks.append(k_full[ctx:])
        new_vs.append(v_full[ctx:])

    query = torch.cat(qs)
    key = torch.cat(new_ks)
    value = torch.cat(new_vs)

    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_ctxs,
        v_contexts=v_ctxs,
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=2048,
        common_attn_metadata=common,
        randomize_blocks=True,
    )

    with set_current_vllm_config(vllm_config):
        actual = run_attention_backend(
            AttentionBackendEnum.FLASH_ATTN,
            kv_cache_spec,
            ["model.layers.0.self_attn.attn"],
            vllm_config,
            device,
            common,
            query,
            key,
            value,
            kv_cache,
            sliding_window=SLIDING_WINDOW,
        )

    expected = _reference(
        query,
        k_fulls,
        v_fulls,
        BATCH.query_lens,
        BATCH.seq_lens,
        MM_RANGES if with_mm_ranges else {},
        SLIDING_WINDOW,
        scale,
        num_heads,
        num_kv_heads,
        device,
    )
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)

    if with_mm_ranges:
        causal_only = _reference(
            query,
            k_fulls,
            v_fulls,
            BATCH.query_lens,
            BATCH.seq_lens,
            {},
            SLIDING_WINDOW,
            scale,
            num_heads,
            num_kv_heads,
            device,
        )
        assert not torch.allclose(expected, causal_only, atol=2e-2, rtol=2e-2), (
            "test batch does not actually exercise the mm_prefix branch"
        )
