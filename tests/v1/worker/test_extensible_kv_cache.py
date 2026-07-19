# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU integration tests for the extensible KV cache allocation paths.

Drives `GPUModelRunner._allocate_kv_cache_tensors` / `_reshape_kv_cache_tensors`
/ `extend_kv_cache` directly with fake attention backends, covering the buffer
layouts the extensible flow supports: block-major (one committed prefix),
K/V-split (one prefix per half), Mamba (block-major per layer), and hybrid
attention + Mamba (attention re-strided to block-major). Buffer sizes exceed
the CUDA VMM allocation granularity so touching a block that the commit logic
missed would fault instead of silently passing.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.gpu.attn_utils import (
    _allocate_extensible_kv_cache,
    _kv_cache_num_segments_by_layer,
    _reshape_kv_cache,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

BLOCK_SIZE = 16
NUM_BLOCKS = 256


class _SplitKVBackend(AttentionBackend):
    """Fake backend with a K/V-split layout, like FlashAttention."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class _BlockMajorBackend(AttentionBackend):
    """Fake backend with a num-blocks-first layout, like FlashInfer."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


class _StrideOrderBackend(AttentionBackend):
    """Fake backend whose stride order makes a kv-first shape block-major."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (1, 0, 2, 3, 4)


def _full_attention_spec() -> FullAttentionSpec:
    # page_size_bytes = 2 (K+V) * 16 * 8 * 128 * 2 bytes = 64 KiB; 256 blocks
    # = 16 MiB, several VMM granules per buffer.
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _mamba_spec() -> MambaSpec:
    # page_size_bytes = (8*128 + 16*64) * 4 bytes = 8 KiB per block per layer.
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((8, 128), (16, 64)),
        dtypes=(torch.float32, torch.float32),
    )


def _make_runner(kv_cache_config: KVCacheConfig, attn_groups) -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cuda:0")
    runner.kv_cache_config = kv_cache_config
    runner.attn_groups = attn_groups
    runner.runner_only_attn_layers = set()
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    return runner


def _attention_config(spec: FullAttentionSpec, backend) -> tuple[KVCacheConfig, list]:
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=NUM_BLOCKS * spec.page_size_bytes, shared_by=["layer.0"])
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )
    attn_groups = [
        [
            AttentionGroup(
                backend=backend,
                layer_names=["layer.0"],
                kv_cache_spec=spec,
                kv_cache_group_id=0,
            )
        ]
    ]
    return kv_cache_config, attn_groups


def _free_buffers(runner: GPUModelRunner) -> None:
    for buffer, _ in getattr(runner, "_extensible_kv_cache_buffers", []):
        buffer.free()


def test_kv_cache_num_segments_by_layer() -> None:
    """Segment counts follow the physical layout of each layer's backend."""
    spec = _full_attention_spec()
    for backend, expected in (
        (_SplitKVBackend, 2),
        (_BlockMajorBackend, 1),
        # kv-first logical shape but block-major physical order -> 1 segment.
        (_StrideOrderBackend, 1),
    ):
        kv_cache_config, attn_groups = _attention_config(spec, backend)
        runner = _make_runner(kv_cache_config, attn_groups)
        assert runner._kv_cache_num_segments_by_layer() == {"layer.0": expected}


def test_extensible_split_layout_grows_both_halves() -> None:
    """A K/V-split layer keeps its natural layout and both halves grow in
    lockstep."""
    spec = _full_attention_spec()
    kv_cache_config, attn_groups = _attention_config(spec, _SplitKVBackend)
    runner = _make_runner(kv_cache_config, attn_groups)
    try:
        raw_tensors = runner._allocate_kv_cache_tensors(
            kv_cache_config, extensible=True
        )
        kv_caches = runner._reshape_kv_cache_tensors(raw_tensors, [BLOCK_SIZE])
        kv_cache = kv_caches["layer.0"]
        assert kv_cache.shape == (2, NUM_BLOCKS, BLOCK_SIZE, 8, 128)
        [(buffer, bytes_per_block_per_segment)] = runner._extensible_kv_cache_buffers
        assert buffer.num_segments == 2
        assert bytes_per_block_per_segment == spec.page_size_bytes // 2

        # Only block 0 is committed -- in each half.
        kv_cache[0, 0].fill_(1)  # K, block 0
        kv_cache[1, 0].fill_(2)  # V, block 0
        torch.cuda.synchronize()

        runner.extend_kv_cache(NUM_BLOCKS)
        # Old data survives the grow; new blocks are usable in both halves and
        # zeroed.
        assert torch.all(kv_cache[0, 0] == 1)
        assert torch.all(kv_cache[1, 0] == 2)
        kv_cache[0, NUM_BLOCKS - 1].fill_(3)
        kv_cache[1, NUM_BLOCKS - 1].fill_(4)
        torch.cuda.synchronize()
        assert torch.all(kv_cache[0, NUM_BLOCKS - 1] == 3)
        assert torch.all(kv_cache[1, NUM_BLOCKS - 1] == 4)
        assert torch.count_nonzero(kv_cache[:, 1 : NUM_BLOCKS - 1]) == 0
    finally:
        _free_buffers(runner)


def test_extensible_block_major_layout() -> None:
    """A layer whose physical layout is block-major uses a single segment."""
    spec = _full_attention_spec()
    kv_cache_config, attn_groups = _attention_config(spec, _BlockMajorBackend)
    runner = _make_runner(kv_cache_config, attn_groups)
    try:
        raw_tensors = runner._allocate_kv_cache_tensors(
            kv_cache_config, extensible=True
        )
        kv_caches = runner._reshape_kv_cache_tensors(raw_tensors, [BLOCK_SIZE])
        kv_cache = kv_caches["layer.0"]
        assert kv_cache.shape == (NUM_BLOCKS, 2, BLOCK_SIZE, 8, 128)
        [(buffer, bytes_per_block_per_segment)] = runner._extensible_kv_cache_buffers
        assert buffer.num_segments == 1
        assert bytes_per_block_per_segment == spec.page_size_bytes

        kv_cache[0].fill_(1)
        runner.extend_kv_cache(NUM_BLOCKS)
        kv_cache[NUM_BLOCKS - 1].fill_(2)
        torch.cuda.synchronize()
        assert torch.all(kv_cache[0] == 1)
        assert torch.all(kv_cache[NUM_BLOCKS - 1] == 2)
        assert torch.count_nonzero(kv_cache[1 : NUM_BLOCKS - 1]) == 0
    finally:
        _free_buffers(runner)


def test_legacy_split_layout_commits_everything() -> None:
    """Without `extensible`, the full buffer is committed up front."""
    spec = _full_attention_spec()
    kv_cache_config, attn_groups = _attention_config(spec, _SplitKVBackend)
    runner = _make_runner(kv_cache_config, attn_groups)
    raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config, extensible=False)
    kv_caches = runner._reshape_kv_cache_tensors(raw_tensors, [BLOCK_SIZE])
    kv_cache = kv_caches["layer.0"]
    kv_cache[0, NUM_BLOCKS - 1].fill_(1)
    kv_cache[1, NUM_BLOCKS - 1].fill_(2)
    torch.cuda.synchronize()
    assert torch.all(kv_cache[0, NUM_BLOCKS - 1] == 1)
    assert torch.all(kv_cache[1, NUM_BLOCKS - 1] == 2)
    with pytest.raises(RuntimeError, match="extensible"):
        runner.extend_kv_cache(NUM_BLOCKS)


def test_extensible_mamba_grows_per_layer() -> None:
    """Mamba per-layer buffers are block-major and grow with the KV cache."""
    spec = _mamba_spec()
    num_blocks = 512
    layer_names = ["mamba.0", "mamba.1"]
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[name])
            for name in layer_names
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)],
    )
    attn_groups = [
        [
            AttentionGroup(
                backend=_BlockMajorBackend,
                layer_names=layer_names,
                kv_cache_spec=spec,
                kv_cache_group_id=0,
            )
        ]
    ]
    runner = _make_runner(kv_cache_config, attn_groups)
    try:
        raw_tensors = runner._allocate_kv_cache_tensors(
            kv_cache_config, extensible=True
        )
        kv_caches = runner._reshape_kv_cache_tensors(raw_tensors, [BLOCK_SIZE])
        assert set(kv_caches) == set(layer_names)
        assert len(runner._extensible_kv_cache_buffers) == len(layer_names)
        for buffer, bytes_per_block_per_segment in runner._extensible_kv_cache_buffers:
            assert buffer.num_segments == 1
            assert bytes_per_block_per_segment == spec.page_size_bytes

        # Write block 0 of every state of every layer (the committed
        # prefixes), then grow.
        for name in layer_names:
            for state_tensor in kv_caches[name]:
                state_tensor[0].fill_(1)
        torch.cuda.synchronize()
        runner.extend_kv_cache(num_blocks)
        for name in layer_names:
            for state_tensor in kv_caches[name]:
                state_tensor[num_blocks - 1].fill_(2)
        torch.cuda.synchronize()
        for name in layer_names:
            for state_tensor in kv_caches[name]:
                assert torch.all(state_tensor[0] == 1)
                assert torch.all(state_tensor[num_blocks - 1] == 2)
                assert torch.count_nonzero(state_tensor[1 : num_blocks - 1]) == 0
    finally:
        _free_buffers(runner)


def test_extensible_hybrid_attention_mamba() -> None:
    """In hybrid models the attention cache is re-strided to block-major, so
    its buffer must use a single segment."""
    attn_spec = _full_attention_spec()
    mamba_spec = _mamba_spec()
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=NUM_BLOCKS * attn_spec.page_size_bytes, shared_by=["attn.0"]
            ),
            KVCacheTensor(
                size=NUM_BLOCKS * mamba_spec.page_size_bytes, shared_by=["mamba.0"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"], kv_cache_spec=mamba_spec),
        ],
    )
    attn_groups = [
        [
            AttentionGroup(
                backend=_SplitKVBackend,
                layer_names=["attn.0"],
                kv_cache_spec=attn_spec,
                kv_cache_group_id=0,
            )
        ],
        [
            AttentionGroup(
                backend=_BlockMajorBackend,
                layer_names=["mamba.0"],
                kv_cache_spec=mamba_spec,
                kv_cache_group_id=1,
            )
        ],
    ]
    runner = _make_runner(kv_cache_config, attn_groups)
    try:
        # The K/V-split attention layer is forced to one segment by the hybrid
        # block-major re-stride.
        assert runner._kv_cache_num_segments_by_layer() == {"attn.0": 1, "mamba.0": 1}

        raw_tensors = runner._allocate_kv_cache_tensors(
            kv_cache_config, extensible=True
        )
        kv_caches = runner._reshape_kv_cache_tensors(
            raw_tensors, [BLOCK_SIZE, BLOCK_SIZE]
        )
        attn_cache = kv_caches["attn.0"]
        # `_update_hybrid_attention_mamba_layout` re-strides to interleave K/V
        # per block: block b spans one contiguous page.
        hidden_size = attn_cache.shape[2:].numel()
        assert attn_cache.stride()[:2] == (hidden_size, 2 * hidden_size)

        attn_cache[0, 0].fill_(1)  # K, block 0
        attn_cache[1, 0].fill_(2)  # V, block 0
        for state_tensor in kv_caches["mamba.0"]:
            state_tensor[0].fill_(3)
        torch.cuda.synchronize()

        runner.extend_kv_cache(NUM_BLOCKS)
        attn_cache[0, NUM_BLOCKS - 1].fill_(4)
        attn_cache[1, NUM_BLOCKS - 1].fill_(5)
        for state_tensor in kv_caches["mamba.0"]:
            state_tensor[NUM_BLOCKS - 1].fill_(6)
        torch.cuda.synchronize()
        assert torch.all(attn_cache[0, 0] == 1)
        assert torch.all(attn_cache[1, 0] == 2)
        assert torch.all(attn_cache[0, NUM_BLOCKS - 1] == 4)
        assert torch.all(attn_cache[1, NUM_BLOCKS - 1] == 5)
        assert torch.count_nonzero(attn_cache[:, 1 : NUM_BLOCKS - 1]) == 0
        for state_tensor in kv_caches["mamba.0"]:
            assert torch.all(state_tensor[0] == 3)
            assert torch.all(state_tensor[NUM_BLOCKS - 1] == 6)
            assert torch.count_nonzero(state_tensor[1 : NUM_BLOCKS - 1]) == 0
    finally:
        _free_buffers(runner)


# ---------------------------------------------------------------------------
# V2 model runner (vllm.v1.worker.gpu) extensible allocation
# ---------------------------------------------------------------------------


def _v2_allocate(kv_cache_config, attn_groups, kernel_block_sizes):
    flat_groups = [g for groups in attn_groups for g in groups]
    raw_tensors, buffers = _allocate_extensible_kv_cache(
        kv_cache_config,
        {},
        torch.device("cuda:0"),
        flat_groups,
        kernel_block_sizes,
        "auto",
    )
    kv_caches = _reshape_kv_cache(
        attn_groups=flat_groups,
        kv_cache_raw_tensors=raw_tensors,
        cache_dtype="auto",
        kernel_block_sizes=kernel_block_sizes,
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    return kv_caches, buffers


def test_v2_num_segments_by_layer() -> None:
    """V2 segment counts follow the layer's physical layout, and hybrid
    models force block-major (one segment)."""
    spec = _full_attention_spec()
    for backend, expected in (
        (_SplitKVBackend, 2),
        (_BlockMajorBackend, 1),
        (_StrideOrderBackend, 1),
    ):
        _, attn_groups = _attention_config(spec, backend)
        flat_groups = [g for groups in attn_groups for g in groups]
        assert _kv_cache_num_segments_by_layer(
            flat_groups, [BLOCK_SIZE], "auto", has_mamba=False
        ) == {"layer.0": expected}
        assert _kv_cache_num_segments_by_layer(
            flat_groups, [BLOCK_SIZE], "auto", has_mamba=True
        ) == {"layer.0": 1}


def test_v2_extensible_split_layout_grows_incrementally() -> None:
    """A K/V-split layer grows both halves in lockstep through the staged
    commits the V2 flow performs (init -> warmup prefix -> final size)."""
    spec = _full_attention_spec()
    kv_cache_config, attn_groups = _attention_config(spec, _SplitKVBackend)
    kv_caches, buffers = _v2_allocate(kv_cache_config, attn_groups, [BLOCK_SIZE])
    try:
        kv_cache = kv_caches["layer.0"]
        assert kv_cache.shape == (2, NUM_BLOCKS, BLOCK_SIZE, 8, 128)
        assert buffers.num_blocks_committed == 1

        kv_cache[0, 0].fill_(1)  # K, block 0
        kv_cache[1, 0].fill_(2)  # V, block 0
        torch.cuda.synchronize()

        # Warmup-style prefix commit, then the final post-warmup commit.
        buffers.commit(8)
        kv_cache[0, 7].fill_(3)
        torch.cuda.synchronize()
        buffers.commit(NUM_BLOCKS)
        # Shrink requests are ignored.
        buffers.commit(1)
        assert buffers.num_blocks_committed == NUM_BLOCKS

        kv_cache[1, NUM_BLOCKS - 1].fill_(4)
        torch.cuda.synchronize()
        assert torch.all(kv_cache[0, 0] == 1)
        assert torch.all(kv_cache[1, 0] == 2)
        assert torch.all(kv_cache[0, 7] == 3)
        assert torch.all(kv_cache[1, NUM_BLOCKS - 1] == 4)
        assert torch.count_nonzero(kv_cache[:, 1:7]) == 0
        assert torch.count_nonzero(kv_cache[:, 8 : NUM_BLOCKS - 1]) == 0
        assert buffers.physical_bytes >= NUM_BLOCKS * spec.page_size_bytes
    finally:
        buffers.free()


def test_v2_extensible_hybrid_attention_mamba() -> None:
    """V2 hybrid models re-stride attention to block-major; both the
    attention and Mamba buffers grow as single-segment prefixes."""
    attn_spec = _full_attention_spec()
    mamba_spec = _mamba_spec()
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=NUM_BLOCKS * attn_spec.page_size_bytes, shared_by=["attn.0"]
            ),
            KVCacheTensor(
                size=NUM_BLOCKS * mamba_spec.page_size_bytes, shared_by=["mamba.0"]
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mamba.0"], kv_cache_spec=mamba_spec),
        ],
    )
    attn_groups = [
        [
            AttentionGroup(
                backend=_SplitKVBackend,
                layer_names=["attn.0"],
                kv_cache_spec=attn_spec,
                kv_cache_group_id=0,
            )
        ],
        [
            AttentionGroup(
                backend=_BlockMajorBackend,
                layer_names=["mamba.0"],
                kv_cache_spec=mamba_spec,
                kv_cache_group_id=1,
            )
        ],
    ]
    kv_caches, buffers = _v2_allocate(
        kv_cache_config, attn_groups, [BLOCK_SIZE, BLOCK_SIZE]
    )
    try:
        attn_cache = kv_caches["attn.0"]
        # Re-strided to interleave K/V per block: block b spans one page.
        hidden_size = attn_cache.shape[2:].numel()
        assert attn_cache.stride()[:2] == (hidden_size, 2 * hidden_size)

        attn_cache[0, 0].fill_(1)
        attn_cache[1, 0].fill_(2)
        for state_tensor in kv_caches["mamba.0"]:
            state_tensor[0].fill_(3)
        torch.cuda.synchronize()

        buffers.commit(NUM_BLOCKS)
        attn_cache[0, NUM_BLOCKS - 1].fill_(4)
        attn_cache[1, NUM_BLOCKS - 1].fill_(5)
        for state_tensor in kv_caches["mamba.0"]:
            state_tensor[NUM_BLOCKS - 1].fill_(6)
        torch.cuda.synchronize()
        assert torch.all(attn_cache[0, 0] == 1)
        assert torch.all(attn_cache[1, 0] == 2)
        assert torch.all(attn_cache[0, NUM_BLOCKS - 1] == 4)
        assert torch.all(attn_cache[1, NUM_BLOCKS - 1] == 5)
        assert torch.count_nonzero(attn_cache[:, 1 : NUM_BLOCKS - 1]) == 0
        for state_tensor in kv_caches["mamba.0"]:
            assert torch.all(state_tensor[0] == 3)
            assert torch.all(state_tensor[NUM_BLOCKS - 1] == 6)
            assert torch.count_nonzero(state_tensor[1 : NUM_BLOCKS - 1]) == 0
    finally:
        buffers.free()
