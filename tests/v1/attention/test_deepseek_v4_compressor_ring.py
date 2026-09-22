# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonJitKey
from vllm.models.deepseek_v4.compressor import (
    CompressorMetadataBuilder,
    CompressorStateCache,
    _build_c128_ring_metadata,
    _c128_ring_capacity,
    build_c128_ring_metadata,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import CircularBufferSpec, SlidingWindowMLASpec

pytestmark = pytest.mark.skip_global_cleanup


def _state_cache(compress_ratio: int) -> CompressorStateCache:
    cache = object.__new__(CompressorStateCache)
    cache.compress_ratio = compress_ratio
    cache.state_dim = 1024
    cache.dtype = torch.float32
    cache.block_size = 4 if compress_ratio == 4 else 8
    cache.sliding_window = (2 if compress_ratio == 4 else 1) * compress_ratio
    return cache


def _config(num_speculative_tokens: int = 0):
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


@pytest.mark.parametrize("is_cuda", [True, False])
def test_c128_spec_is_circular_only_on_cuda(monkeypatch, is_cuda: bool) -> None:
    from vllm.models.deepseek_v4 import compressor

    monkeypatch.setattr(compressor.current_platform, "is_cuda", lambda: is_cuda)
    c128_spec = _state_cache(128).get_kv_cache_spec(_config(5))
    if is_cuda:
        assert isinstance(c128_spec, CircularBufferSpec)
        assert c128_spec.block_size == 256
        assert c128_spec.prefix_cacheable is False
        assert c128_spec.uses_slot_mapping is False
    else:
        assert isinstance(c128_spec, SlidingWindowMLASpec)
        assert c128_spec.block_size == 8
        assert c128_spec.sliding_window == 128

    c4_spec = _state_cache(4).get_kv_cache_spec(_config(5))
    assert isinstance(c4_spec, SlidingWindowMLASpec)
    assert c4_spec.block_size == 4
    assert c4_spec.sliding_window == 8


def test_c128_capacity_covers_one_speculative_step() -> None:
    assert [_c128_ring_capacity(n) for n in (0, 1, 127, 128, 129)] == [
        128,
        256,
        256,
        256,
        384,
    ]


@pytest.mark.parametrize("group_phase", range(128))
def test_c128_ring_survives_one_speculative_step(group_phase: int) -> None:
    """Draft writes must not overwrite committed rows in the open group."""
    num_spec = 5
    capacity = _c128_ring_capacity(num_spec)
    query_len = num_spec + 1
    positions = torch.arange(group_phase, group_phase + query_len)
    slots, _ = build_c128_ring_metadata(
        torch.full((query_len,), -1, dtype=torch.int64),
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([0, query_len], dtype=torch.int32),
        positions,
        query_len,
        1,
        capacity,
    )

    committed = torch.arange(group_phase - group_phase % 128, group_phase)
    assert set(slots.tolist()).isdisjoint((committed % capacity).tolist())


def test_c128_metadata_warmup_has_one_shape_independent_key(monkeypatch) -> None:
    from vllm.model_executor.warmup import jit_warmup_triton_helper

    prepared = []

    def fake_key_deriver(kernel):
        def derive(kwargs):
            prepared.append(kwargs)
            return {TritonJitKey(id(kernel), "fake", 0, kwargs["CAPACITY"])}

        return derive

    monkeypatch.setattr(
        jit_warmup_triton_helper, "_triton_key_deriver", fake_key_deriver
    )
    kernel = _build_c128_ring_metadata.kernel
    kernel_fn = getattr(kernel, "func", kernel)
    monkeypatch.setitem(
        _build_c128_ring_metadata.__dict__,
        "_kernel_arg_names",
        tuple(inspect.signature(kernel_fn).parameters),
    )
    keys = _build_c128_ring_metadata.get_warmup_keys(capacity=256)

    assert len(keys) == 1
    assert prepared[0]["CAPACITY"] == 256
    assert prepared[0]["BLOCK"] == 256
    assert "num_reqs" not in prepared[0]


@pytest.mark.parametrize(
    ("is_cuda", "is_rocm", "expected"),
    [(True, False, [{"capacity": 256}]), (False, True, [])],
)
def test_c128_model_registers_ring_metadata_warmup_only_on_cuda(
    monkeypatch, is_cuda: bool, is_rocm: bool, expected: list[dict]
) -> None:
    from vllm.models.deepseek_v4 import compressor
    from vllm.models.deepseek_v4.common.ops import fused_compress_quant_cache

    registered = []
    monkeypatch.setattr(
        compressor,
        "current_platform",
        SimpleNamespace(
            device_type="cpu",
            is_cuda=lambda: is_cuda,
            is_rocm=lambda: is_rocm,
            is_xpu=lambda: False,
        ),
    )
    monkeypatch.setattr(
        compressor._build_c128_ring_metadata,
        "register_warmup",
        lambda **kwargs: registered.append(kwargs),
    )
    monkeypatch.setattr(
        compressor._SAVE_PARTIAL_STATES_KERNEL, "register_warmup", lambda **kwargs: None
    )
    indexer_kernel_name = "_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL"
    indexer_kernel = getattr(fused_compress_quant_cache, indexer_kernel_name)
    monkeypatch.setattr(indexer_kernel, "register_warmup", lambda: None)
    monkeypatch.setattr(
        compressor,
        "MergedColumnParallelLinear",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        compressor, "RMSNorm", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        compressor, "CompressorStateCache", lambda *args, **kwargs: torch.nn.Identity()
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(qk_rope_head_dim=64, rms_norm_eps=1e-6),
            max_model_len=1024,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=128,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
        num_speculative_tokens=5,
    )

    compressor.DeepseekCompressor(config, 128, 128, 128)

    assert registered == expected


def test_only_circular_c128_allocates_ring_metadata_buffers(monkeypatch) -> None:
    from vllm.models.deepseek_v4 import compressor

    monkeypatch.setattr(compressor.current_platform, "is_cuda", lambda: True)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=127,
            async_scheduling=False,
        ),
        speculative_config=None,
    )

    def build(spec):
        builder = CompressorMetadataBuilder(
            spec, ["state"], config, torch.device("cpu")
        )
        query_start_loc = torch.tensor([0, 127], dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=torch.tensor([127], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([127], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=127,
            max_query_len=127,
            max_seq_len=127,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.full((127,), -1, dtype=torch.int64),
            positions=torch.arange(127),
        )
        common._token_to_req_indices_cache = torch.zeros(127, dtype=torch.int32)
        return builder, builder.build(0, common)

    circular = _state_cache(128).get_kv_cache_spec(_config())
    circular_builder, circular_metadata = build(circular)
    assert circular_builder.slot_mapping_buffer is not None
    assert circular_builder.tail_slot_mapping_buffer is not None
    assert circular_builder.slot_mapping_buffer.shape == (127,)
    assert circular_builder.tail_slot_mapping_buffer.shape == (127,)
    assert circular_builder.slot_mapping_buffer.data_ptr() % 16 == 0
    assert circular_builder.tail_slot_mapping_buffer.data_ptr() % 16 == 0
    assert circular_metadata.c128_boundary is False

    c4 = _state_cache(4).get_kv_cache_spec(_config())
    c4_builder, c4_metadata = build(c4)
    assert c4_builder.slot_mapping_buffer is None
    assert c4_builder.tail_slot_mapping_buffer is None
    assert c4_metadata.c128_boundary is None

    monkeypatch.setattr(compressor.current_platform, "is_cuda", lambda: False)
    paged_c128 = _state_cache(128).get_kv_cache_spec(_config())
    paged_c128_builder, paged_c128_metadata = build(paged_c128)
    assert paged_c128_builder.slot_mapping_buffer is None
    assert paged_c128_builder.tail_slot_mapping_buffer is None
    assert paged_c128_metadata.c128_boundary is None


def test_c128_ring_mapping_and_tail_for_nonuniform_batch() -> None:
    per_req = [torch.arange(120, 400), torch.arange(250, 390)]
    positions = torch.cat(per_req)
    query_start_loc = torch.tensor([0, len(per_req[0]), positions.numel()])
    num_actual = positions.numel()
    padded = num_actual + 7
    common_slots = torch.full((padded,), -1, dtype=torch.int64)
    block_table = torch.tensor([[11, -1], [3, -1]], dtype=torch.int32)

    slots, tail_slots = build_c128_ring_metadata(
        common_slots,
        block_table,
        query_start_loc,
        positions,
        num_actual,
        2,
        256,
    )

    expected_slots = torch.cat(
        [11 * 256 + per_req[0] % 256, 3 * 256 + per_req[1] % 256]
    )
    assert torch.equal(slots[:num_actual], expected_slots)
    assert slots[num_actual:].tolist() == [-1] * 7
    assert tail_slots[:24].tolist() == [-1] * 24
    assert torch.equal(tail_slots[24 : len(per_req[0])], expected_slots[24:280])
    assert torch.equal(tail_slots[280:num_actual], expected_slots[280:num_actual])
    assert tail_slots[num_actual:].tolist() == [-1] * 7


def test_c128_ring_mapping_masks_padded_actual_tokens() -> None:
    slots, tail_slots = build_c128_ring_metadata(
        torch.full((4,), -1, dtype=torch.int64),
        torch.tensor([[2]], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([10, 11, 12, 0]),
        num_actual_tokens=4,
        num_reqs=1,
        capacity=128,
        token_to_req_indices=torch.zeros(4, dtype=torch.int32),
    )

    assert slots.tolist() == [266, 267, 268, -1]
    assert tail_slots.tolist() == [266, 267, 268, -1]


def test_c128_reads_the_batch_before_saving_the_wrapped_tail() -> None:
    old_positions = range(128, 250)
    new_positions = torch.arange(250, 521)
    query_start_loc = torch.tensor([0, new_positions.numel()])
    common_slots = torch.full_like(new_positions, -1)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    capacity = 256
    slots, tail_slots = build_c128_ring_metadata(
        common_slots,
        block_table,
        query_start_loc,
        new_positions,
        new_positions.numel(),
        1,
        capacity,
    )

    ring = {pos % capacity: pos for pos in old_positions}
    boundaries = [255, 383, 511]
    chunk_start = int(new_positions[0])
    for position in boundaries:
        gathered = [
            (
                int(new_positions[pos - chunk_start])
                if pos >= chunk_start
                else ring[pos % capacity]
            )
            for pos in range(position - 127, position + 1)
        ]
        assert gathered == list(range(position - 127, position + 1))

    for row in torch.nonzero(tail_slots >= 0).flatten().tolist():
        ring[int(slots[row]) % capacity] = int(new_positions[row])
    assert boundaries == [255, 383, 511]
    assert sorted(ring.values()) == list(range(265, 521))


def test_c128_null_block_keeps_all_slots_invalid() -> None:
    positions = torch.arange(120, 140)
    common_slots = torch.full((24,), -1, dtype=torch.int64)
    slots, tail_slots = build_c128_ring_metadata(
        common_slots,
        torch.tensor([[-1]], dtype=torch.int32),
        torch.tensor([0, 20]),
        positions,
        20,
        1,
        256,
    )
    assert slots.tolist() == [-1] * 24
    assert tail_slots.tolist() == [-1] * 24
