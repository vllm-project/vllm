# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec-driven heterogeneous-TP KV transfer (RFC #42082).

A connector that only calls ``slice_for_tp_transfer`` and
``jointly_contiguous_chunks`` must assemble the same bytes a decode rank would
have written itself, for prefill/decode engines with different TP sizes and
layouts.
"""

import pytest
import torch

from tests.v1.kv_connector.unit.utils import transfer_kv
from vllm.distributed.kv_transfer.kv_connector.utils import mamba_tp_transfer_chunks
from vllm.v1.attention.backends.gdn_attn import GDNSpec
from vllm.v1.attention.backends.mamba1_attn import Mamba1Spec
from vllm.v1.attention.backends.mamba2_attn import Mamba2Spec
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

pytestmark = pytest.mark.cpu_test

NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE = 3, 4, 2
TP_PAIRS = [(1, 2), (2, 1), (1, 4), (4, 1), (2, 4), (4, 2), (2, 2), (8, 2), (2, 8)]


def _make_cache(
    kv: torch.Tensor, heads: range, hnd: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """(raw bytes, logical [B, H, N, 2 * D] view) of one rank's packed K|V
    cache, built from global K/V of shape [2, H, B, N, D]."""
    page = torch.cat(
        [kv[0, heads.start : heads.stop], kv[1, heads.start : heads.stop]], -1
    )
    page = page.transpose(0, 1)
    physical = page if hnd else page.transpose(1, 2)
    raw = physical.contiguous().flatten().view(torch.uint8)
    view = raw.view(kv.dtype).view(physical.shape)
    return raw, view if hnd else view.transpose(1, 2)


def _spec(total_heads: int, tp_size: int) -> AttentionSpec:
    return AttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=max(1, total_heads // tp_size),
        head_size=HEAD_SIZE,
        dtype=torch.float32,
        max_tp_shards=total_heads,
    )


def _heads(spec: AttentionSpec, tp_size: int, tp_rank: int) -> range:
    total = spec.max_tp_shards
    start = tp_rank * min(tp_size, total) // tp_size * spec.num_kv_heads
    return range(start, start + spec.num_kv_heads)


@pytest.mark.parametrize("p_tp,d_tp", TP_PAIRS)
@pytest.mark.parametrize("p_hnd,d_hnd", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("total_heads", [4, 2, 1])
def test_hetero_tp_transfer_assembles_decode_cache(
    p_tp, d_tp, p_hnd, d_hnd, total_heads
):
    shape = (2, total_heads, NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE)
    kv = torch.arange(1, 1 + torch.Size(shape).numel(), dtype=torch.float32)
    kv = kv.view(shape)
    p_spec, d_spec = _spec(total_heads, p_tp), _spec(total_heads, d_tp)
    p_caches = [_make_cache(kv, _heads(p_spec, p_tp, r), p_hnd) for r in range(p_tp)]
    for d_rank in range(d_tp):
        d_heads = _heads(d_spec, d_tp, d_rank)
        expected, _ = _make_cache(kv, d_heads, d_hnd)
        d_cache = _make_cache(torch.zeros_like(kv), d_heads, d_hnd)
        transfer_kv(p_spec, p_caches, p_tp, d_spec, d_cache, d_tp, d_rank)
        torch.testing.assert_close(d_cache[0], expected, rtol=0, atol=0)


def test_receivers_spread_reads_across_replicas():
    """Decode ranks sharing a KV head read it from different prefill replicas."""
    p_spec, d_spec = _spec(2, 8), _spec(2, 4)
    cache = torch.empty(NUM_BLOCKS, 1, BLOCK_SIZE, 2 * HEAD_SIZE, device="meta")
    sources = [
        [
            p_rank
            for p_rank in range(8)
            if d_spec.slice_for_tp_transfer(
                cache, 4, d_rank, p_spec, 8, p_rank, receiving=True
            )
        ]
        for d_rank in range(4)
    ]
    assert sources == [[0], [2], [4], [6]]


def test_replicated_heads_require_max_tp_shards():
    """With one head per rank on both sides, the total head count is unknown."""
    spec = AttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=HEAD_SIZE, dtype=torch.float32
    )
    cache = torch.empty(NUM_BLOCKS, 1, BLOCK_SIZE, 2 * HEAD_SIZE, device="meta")
    with pytest.raises(ValueError, match="max_tp_shards"):
        spec.slice_for_tp_transfer(cache, 4, 0, spec, 2, 0, receiving=True)


def test_custom_head_slots_transfer_only_whole_pages():
    """A plain spec cannot separate heads packed into custom slots."""

    def spec(num_kv_heads: int) -> AttentionSpec:
        return AttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_kv_heads,
            head_size=HEAD_SIZE,
            dtype=torch.float32,
            num_head_slots=2,
            state_content_bytes=num_kv_heads * HEAD_SIZE * 4,
        )

    cache = torch.empty(NUM_BLOCKS, 2, BLOCK_SIZE, 4 * HEAD_SIZE, device="meta")
    slices = spec(4).slice_for_tp_transfer(cache, 1, 0, spec(4), 1, 0, receiving=True)
    assert slices == [cache]
    with pytest.raises(NotImplementedError):
        spec(4).slice_for_tp_transfer(cache, 1, 0, spec(2), 2, 0, receiving=True)


CONV_ROWS = 3
# Per Mamba spec: the global conv sub-projection sizes, each sharded across TP,
# and the global temporal state shape, sharded along its first dim.
MAMBA_LAYOUTS = {
    Mamba1Spec: ((16,), (16, 4)),
    Mamba2Spec: ((16, 8, 8), (8, 2, 4)),
    GDNSpec: ((8, 8, 16), (8, 2, 4)),
}


def _make_mamba_page(
    spec_cls: type[MambaSpec],
    conv: list[torch.Tensor],
    temporal: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> tuple[MambaSpec, tuple[torch.Tensor, torch.Tensor]]:
    """One rank's spec and (raw bytes, ``[B, 1, 1, C]`` int8 view) pages, from
    global per-block conv sub-projections and temporal state."""

    def shard(state: torch.Tensor) -> torch.Tensor:
        rows = state.shape[1] // tp_size
        return state[:, tp_rank * rows : (tp_rank + 1) * rows]

    local_conv = torch.cat([shard(proj) for proj in conv], dim=1)
    local_temporal = shard(temporal)
    spec = spec_cls(
        block_size=BLOCK_SIZE,
        shapes=(tuple(local_conv.shape[1:]), tuple(local_temporal.shape[1:])),
        dtypes=(local_conv.dtype, local_temporal.dtype),
    )
    raw = torch.cat(
        [
            local_conv.flatten(1).view(torch.uint8),
            local_temporal.flatten(1).view(torch.uint8),
        ],
        dim=1,
    ).flatten()
    return spec, (raw, raw.view(torch.int8).view(NUM_BLOCKS, 1, 1, -1))


@pytest.mark.parametrize("p_tp,d_tp", TP_PAIRS)
@pytest.mark.parametrize("spec_cls", list(MAMBA_LAYOUTS))
def test_hetero_tp_transfer_assembles_decode_mamba_state(p_tp, d_tp, spec_cls):
    """Each conv sub-projection and the temporal state are resharded on their own."""
    conv_sizes, temporal_shape = MAMBA_LAYOUTS[spec_cls]
    values = iter(range(1, 1 << 20))

    def state(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        numel = torch.Size(shape).numel()
        data = [next(values) for _ in range(numel)]
        return torch.tensor(data, dtype=dtype).view(shape)

    conv = [
        state(NUM_BLOCKS, size, CONV_ROWS, dtype=torch.float16) for size in conv_sizes
    ]
    temporal = state(NUM_BLOCKS, *temporal_shape, dtype=torch.float32)
    p_pages = [_make_mamba_page(spec_cls, conv, temporal, p_tp, r) for r in range(p_tp)]
    p_spec = p_pages[0][0]
    for d_rank in range(d_tp):
        d_spec, (expected, _) = _make_mamba_page(spec_cls, conv, temporal, d_tp, d_rank)
        zeros = [torch.zeros_like(proj) for proj in conv]
        _, d_page = _make_mamba_page(
            spec_cls, zeros, torch.zeros_like(temporal), d_tp, d_rank
        )
        transfer_kv(
            p_spec, [page for _, page in p_pages], p_tp, d_spec, d_page, d_tp, d_rank
        )
        assert torch.equal(d_page[0], expected)


def test_mamba_spec_without_tp_layout_transfers_only_whole_pages():
    """A plain MambaSpec does not know how its states shard."""
    spec = MambaSpec(
        block_size=BLOCK_SIZE, shapes=((8, 3), (8, 4)), dtypes=(torch.float32,) * 2
    )
    page = torch.empty(NUM_BLOCKS, 1, 1, spec.page_size_bytes, device="meta")
    assert spec.slice_for_tp_transfer(page, 2, 1, spec, 2, 1, receiving=True) == [page]
    with pytest.raises(NotImplementedError):
        spec.slice_for_tp_transfer(page, 2, 0, spec, 1, 0, receiving=True)


def test_mamba2_replicated_groups_fail_closed():
    """With fewer groups than ranks, B and C are replicated rather than sharded,
    so they cannot be resliced by TP."""
    # One group of 8: TP1 holds it once, each TP2 rank holds a copy.
    tp1 = Mamba2Spec(
        block_size=BLOCK_SIZE,
        shapes=((16 + 2 * 8, 3), (8, 2, 4)),
        dtypes=(torch.float16, torch.float32),
    )
    tp2 = Mamba2Spec(
        block_size=BLOCK_SIZE,
        shapes=((8 + 2 * 8, 3), (4, 2, 4)),
        dtypes=(torch.float16, torch.float32),
    )
    page = torch.empty(NUM_BLOCKS, 1, 1, tp2.page_size_bytes, device="meta")
    with pytest.raises(NotImplementedError, match="does not shard evenly"):
        tp2.slice_for_tp_transfer(page, 2, 0, tp1, 1, 0, receiving=True)


@pytest.mark.parametrize(
    "mamba_type,local_tp,conv_dim_local,conv_rows,temporal_shape,expected_proj_dims",
    [
        # nvidia/Nemotron-H-8B-Base-8K (Mamba2)
        # mamba_num_heads=128, head_dim=64, n_groups=8, ssm_state_size=128
        pytest.param(
            "mamba2",
            1,
            10240,
            3,
            (128, 64, 128),
            (8192, 1024, 1024),
            id="nemotron_h_8b_tp1",
        ),
        pytest.param(
            "mamba2",
            4,
            2560,
            3,
            (32, 64, 128),
            (2048, 256, 256),
            id="nemotron_h_8b_tp4",
        ),
        # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (Mamba2)
        # mamba_num_heads=64, head_dim=64, n_groups=8, ssm_state_size=128
        pytest.param(
            "mamba2",
            1,
            6144,
            3,
            (64, 64, 128),
            (4096, 1024, 1024),
            id="nemotron_nano_30b_tp1",
        ),
        # Qwen/Qwen3.5-0.8B (GDN, symmetric: num_v=num_k=16)
        # key_dim=2048, value_dim=2048, conv_dim=6144
        pytest.param(
            "gdn_attention",
            1,
            6144,
            3,
            (16, 128, 128),
            (2048, 2048, 2048),
            id="qwen35_08b_tp1",
        ),
        pytest.param(
            "gdn_attention",
            4,
            1536,
            3,
            (4, 128, 128),
            (512, 512, 512),
            id="qwen35_08b_tp4",
        ),
        # Qwen/Qwen3.5-4B (GDN, asymmetric: num_v=32, num_k=16, K:V=1:2)
        # key_dim=2048, value_dim=4096, conv_dim=8192
        pytest.param(
            "gdn_attention",
            1,
            8192,
            3,
            (32, 128, 128),
            (2048, 2048, 4096),
            id="qwen35_4b_tp1",
        ),
        # Qwen/Qwen3.5-27B (GDN, asymmetric: num_v=48, num_k=16, K:V=1:3)
        # key_dim=2048, value_dim=6144, conv_dim=10240
        pytest.param(
            "gdn_attention",
            1,
            10240,
            3,
            (48, 128, 128),
            (2048, 2048, 6144),
            id="qwen35_27b_tp1",
        ),
        pytest.param(
            "gdn_attention",
            8,
            1280,
            3,
            (6, 128, 128),
            (256, 256, 768),
            id="qwen35_27b_tp8",
        ),
        # ai21labs/AI21-Jamba2-Mini (Mamba1)
        # mamba d_inner = mamba_expand(2) * hidden_size(4096) = 8192
        # mamba_d_state=16, mamba_d_conv=4 → conv_rows=3.
        # Conv state holds only x: a single contiguous sub-projection.
        pytest.param(
            "mamba1",
            1,
            8192,
            3,
            (8192, 16),
            (8192,),
            id="jamba_mini_tp1",
        ),
        pytest.param(
            "mamba1",
            4,
            2048,
            3,
            (2048, 16),
            (2048,),
            id="jamba_mini_tp4",
        ),
        pytest.param(
            "mamba1",
            8,
            1024,
            3,
            (1024, 16),
            (1024,),
            id="jamba_mini_tp8",
        ),
    ],
)
def test_mamba_specs_split_real_model_conv_states(
    mamba_type, local_tp, conv_dim_local, conv_rows, temporal_shape, expected_proj_dims
):
    """Backend specs cut real models' conv states into their sub-projections."""
    spec_cls = {"mamba1": Mamba1Spec, "mamba2": Mamba2Spec, "gdn_attention": GDNSpec}
    spec = spec_cls[mamba_type](
        block_size=64,
        shapes=((conv_dim_local, conv_rows), temporal_shape),
        dtypes=(torch.bfloat16, torch.bfloat16),
    )
    runs = mamba_tp_transfer_chunks(spec, local_tp, 0, local_tp, 0, receiving=True)
    conv_runs = [size // (conv_rows * 2) for _, _, size in runs[:-1]]
    assert conv_runs == list(expected_proj_dims)
