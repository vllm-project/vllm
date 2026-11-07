# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer DiffKV prefill (TRITON_FLASHINFER_DIFFKV) vs an exact reference.

Drives the backend's own ``plan_paged_prefill`` / ``run_paged_prefill`` on the
DiffKV packed cache ``[num_blocks, num_kv_heads, block_size, hqk + hv]`` with a
scattered block table, and checks the output against an FP32 reference.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skip_global_cleanup

requires_sm12x_flashinfer = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(120)
    or not has_flashinfer(),
    reason="TRITON_FLASHINFER_DIFFKV is gated to sm_12x with FlashInfer",
)

HEAD_SIZE_QK, HEAD_SIZE_V = 192, 128
# (num_query_heads, num_kv_heads): MiMo-V2 full attention at TP2, and one KV head.
NUM_HEADS = [(32, 2), (16, 1)]
# (query_len, kv_len) per request.
SEQ_LENS = [
    [(2048, 2048)],  # fresh prompt
    [(2048, 2048 + 8192)],  # chunked prefill over an 8K prefix
    [(256, 256 + 32768)],  # short chunk over a 32K prefix
    [(17, 17 + 8192), (1, 300)],  # just above the routing threshold
    [(1, 1328), (37, 500), (1024, 1024 + 4096), (1, 77)],  # mixed with decode rows
]
LAYOUTS = ["HND", "NHD", "HND_interleaved", "NHD_interleaved"]


def _packed_cache(layout, num_blocks, num_kv_heads, block_size, dtype):
    """Logical (B, H, N, C) cache with the physical strides of ``layout``.

    ``*_interleaved`` places a second layer's block between consecutive blocks,
    as the block-compact layer-interleaved layouts do.
    """
    content = HEAD_SIZE_QK + HEAD_SIZE_V
    head_major = layout.startswith("HND")
    inner = (
        (num_kv_heads, block_size, content)
        if head_major
        else (block_size, num_kv_heads, content)
    )
    if layout.endswith("interleaved"):
        cache = torch.randn(num_blocks, 2, *inner, dtype=dtype)[:, 1]
    else:
        cache = torch.randn(num_blocks, *inner, dtype=dtype)
    return cache if head_major else cache.transpose(1, 2)


def _reference(query, cache, block_table, query_lens, kv_lens, scale):
    """FP32 causal attention over the HND packed cache, gathered per request."""
    _, num_kv_heads, block_size, _ = cache.shape
    num_heads = query.shape[1]
    out = torch.empty(query.shape[0], num_heads, HEAD_SIZE_V, dtype=torch.float32)
    start = 0
    for i, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        blocks = block_table[i, : cdiv(kv_len, block_size)].long()
        seq = (
            cache[blocks].permute(0, 2, 1, 3).reshape(-1, num_kv_heads, cache.shape[-1])
        )
        seq = seq[:kv_len].float()
        k = seq[..., :HEAD_SIZE_QK].repeat_interleave(num_heads // num_kv_heads, 1)
        v = seq[..., HEAD_SIZE_QK:].repeat_interleave(num_heads // num_kv_heads, 1)
        q = query[start : start + q_len].float()
        q_pos = torch.arange(kv_len - q_len, kv_len)
        k_pos = torch.arange(kv_len)
        for h in range(0, num_heads, 8):  # bound the score matrix
            s = torch.einsum("qhd,khd->hqk", q[:, h : h + 8], k[:, h : h + 8]) * scale
            s.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], float("-inf"))
            out[start : start + q_len, h : h + 8] = torch.einsum(
                "hqk,khd->qhd", s.softmax(-1), v[:, h : h + 8]
            )
        start += q_len
    return out


@requires_sm12x_flashinfer
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("layout", LAYOUTS)
@torch.inference_mode()
def test_flashinfer_diffkv_prefill_vs_reference(
    seq_lens, num_heads, block_size, layout
):
    _check_prefill(seq_lens, num_heads, block_size, layout, torch.bfloat16)


@requires_sm12x_flashinfer
@pytest.mark.parametrize("seq_lens", SEQ_LENS[1:])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@torch.inference_mode()
def test_flashinfer_diffkv_prefill_fp16(seq_lens, layout):
    _check_prefill(seq_lens, NUM_HEADS[0], 16, layout, torch.float16)


def _check_prefill(seq_lens, num_heads, block_size, layout, dtype):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    from vllm.v1.attention.backends.triton_flashinfer_diffkv import (
        plan_paged_prefill,
        run_paged_prefill,
    )
    from vllm.v1.utils import CpuGpuBuffer

    torch.set_default_device("cuda")
    set_random_seed(0)
    num_query_heads, num_kv_heads = num_heads
    query_lens = [q for q, _ in seq_lens]
    kv_lens = [k for _, k in seq_lens]
    num_reqs = len(seq_lens)
    scale = HEAD_SIZE_QK**-0.5

    # Every request's blocks are distinct and scattered across the cache,
    # with spare blocks so indices do not simply run 0..n.
    pages = [cdiv(k, block_size) for k in kv_lens]
    num_blocks = 2 * sum(pages) + 7
    perm = torch.randperm(num_blocks, dtype=torch.int32)
    block_table = torch.zeros(num_reqs, max(pages), dtype=torch.int32)
    offset = 0
    for i, n in enumerate(pages):
        block_table[i, :n] = perm[offset : offset + n]
        offset += n

    cache = _packed_cache(layout, num_blocks, num_kv_heads, block_size, dtype)
    query = torch.randn(sum(query_lens), num_query_heads, HEAD_SIZE_QK, dtype=dtype)

    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        torch.zeros(512 << 20, dtype=torch.uint8), "HND"
    )
    plan_paged_prefill(
        wrapper,
        qo_indptr_cpu=torch.tensor([0] + query_lens, device="cpu").cumsum(
            0, dtype=torch.int32
        ),
        seq_lens_cpu=torch.tensor(kv_lens, dtype=torch.int32, device="cpu"),
        block_table=block_table,
        page_size=block_size,
        paged_kv_indptr=CpuGpuBuffer(
            num_reqs + 1, dtype=torch.int32, device="cuda", pin_memory=False
        ),
        paged_kv_indices=torch.zeros(sum(pages), dtype=torch.int32),
        paged_kv_last_page_len=CpuGpuBuffer(
            num_reqs, dtype=torch.int32, device="cuda", pin_memory=False
        ),
        num_qo_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size_qk=HEAD_SIZE_QK,
        head_size_v=HEAD_SIZE_V,
        sm_scale=scale,
        q_dtype=dtype,
        kv_dtype=dtype,
    )
    out = torch.full(
        (sum(query_lens), num_query_heads, HEAD_SIZE_V), float("nan"), dtype=dtype
    )
    run_paged_prefill(wrapper, query, cache, out, HEAD_SIZE_QK, HEAD_SIZE_V)

    ref = _reference(query, cache, block_table, query_lens, kv_lens, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    assert (out.float() - ref).norm() / ref.norm() < 1e-2


def _spec_config(num_speculative_tokens):
    spec = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens, parallel_drafting=False
        )
    )
    return SimpleNamespace(speculative_config=spec, use_v2_model_runner=True)


@pytest.mark.parametrize("num_speculative_tokens", [None, 3, 31])
def test_routing_keeps_graph_capturable_batches_on_triton(num_speculative_tokens):
    """Capture uses Triton, so replayed (uniform) batches must select it too."""
    from vllm.v1.attention.backends.triton_flashinfer_diffkv import (
        DiffKVPrefillRouting,
    )

    routing = DiffKVPrefillRouting.__new__(DiffKVPrefillRouting)
    routing.enabled = True
    routing.min_query_len = DiffKVPrefillRouting.flashinfer_min_query_len(
        _spec_config(num_speculative_tokens)
    )

    widest_uniform = 1 + (num_speculative_tokens or 0)
    assert DiffKVPrefillRouting.capture_variant == 0
    for query_len in (1, widest_uniform):
        metadata = SimpleNamespace(max_query_len=query_len, mm_req_doc_ranges=None)
        assert routing.select(metadata) == 0
    prefill = SimpleNamespace(max_query_len=4096, mm_req_doc_ranges=None)
    assert routing.select(prefill) == 1


@pytest.mark.parametrize(
    "kwargs,supported",
    [
        ({}, True),
        ({"sliding_window": 128}, False),
        ({"sinks": torch.zeros(8)}, False),
        ({"kv_cache_dtype": "fp8"}, False),
        ({"logits_soft_cap": 30.0}, False),
    ],
)
def test_layer_eligibility(kwargs, supported):
    """Layers FlashInfer does not cover keep their whole group on Triton."""
    from vllm.v1.attention.backends.triton_flashinfer_diffkv import (
        FlashInferDiffKVImpl,
    )

    args = dict(
        num_heads=8,
        head_size=HEAD_SIZE_QK,
        scale=HEAD_SIZE_QK**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    impl = FlashInferDiffKVImpl(**(args | kwargs))
    assert impl.flashinfer_supported is supported


def _batch(query_lens, upper, exact):
    """Metadata whose GPU ``seq_lens`` hold the exact lengths."""
    return SimpleNamespace(
        num_reqs=len(query_lens),
        query_start_loc_cpu=torch.tensor([0, *query_lens]).cumsum(0),
        seq_lens_cpu_upper_bound=torch.tensor(upper),
        seq_lens=torch.tensor(exact),
    )


def test_host_seq_lens_pure_prefill_skips_readback():
    from vllm.v1.attention.backends.triton_flashinfer_diffkv import host_seq_lens

    batch = _batch([2048, 512], upper=[4096, 512], exact=[4096, 512])
    batch.seq_lens = None  # any GPU readback would fail
    assert host_seq_lens(batch, max_decode_query_len=8).tolist() == [4096, 512]


def test_host_seq_lens_mixed_spec_batch_uses_exact_lengths():
    """Spec-decode rows' upper bound assumes every draft was accepted."""
    from vllm.v1.attention.backends.triton_flashinfer_diffkv import host_seq_lens

    batch = _batch([8, 2048], upper=[1000, 4096], exact=[995, 4096])
    assert host_seq_lens(batch, max_decode_query_len=8).tolist() == [995, 4096]
    assert host_seq_lens(batch, max_decode_query_len=1).tolist() == [1000, 4096]


@requires_sm12x_flashinfer
def test_auto_selection_and_opt_out(monkeypatch):
    from vllm.v1.attention.backends.triton_flashinfer_diffkv import (
        TritonFlashInferDiffKVBackend as backend,
    )

    full = dict(
        head_size=HEAD_SIZE_QK,
        head_size_v=HEAD_SIZE_V,
        has_sinks=False,
        sliding_window=None,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
    )
    assert backend.is_supported_on_current_device(**full)
    assert backend.is_supported_on_current_device(**(full | {"dtype": torch.float16}))
    for unsupported in (
        {"has_sinks": True},
        {"sliding_window": 128},
        {"kv_cache_dtype": "fp8"},
        {"head_size": 128, "head_size_v": 128},
    ):
        assert not backend.is_supported_on_current_device(**(full | unsupported))
    monkeypatch.setenv("VLLM_DIFFKV_FLASHINFER_PREFILL", "0")
    assert not backend.is_supported_on_current_device(**full)
