# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonJitKey
from vllm.models.deepseek_v4.compressor import (
    _BUILD_C128_RING_METADATA_KERNEL,
    CompressorMetadataBuilder,
    CompressorStateCache,
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
    kernel = _BUILD_C128_RING_METADATA_KERNEL.kernel
    kernel_fn = getattr(kernel, "func", kernel)
    monkeypatch.setitem(
        _BUILD_C128_RING_METADATA_KERNEL.__dict__,
        "_kernel_arg_names",
        tuple(inspect.signature(kernel_fn).parameters),
    )
    keys = _BUILD_C128_RING_METADATA_KERNEL.get_warmup_keys(capacity=256)

    assert len(keys) == 1
    assert prepared[0]["CAPACITY"] == 256
    assert prepared[0]["BLOCK"] == 256
    assert "num_reqs" not in prepared[0]


def test_c128_model_registers_ring_metadata_warmup(monkeypatch) -> None:
    from vllm.models.deepseek_v4 import compressor
    from vllm.models.deepseek_v4.common.ops import fused_compress_quant_cache

    registered = []
    monkeypatch.setattr(
        compressor,
        "current_platform",
        SimpleNamespace(
            device_type="cpu",
            is_cuda=lambda: True,
            is_rocm=lambda: False,
            is_xpu=lambda: False,
        ),
    )
    monkeypatch.setattr(
        compressor._BUILD_C128_RING_METADATA_KERNEL,
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

    assert registered == [{"capacity": 256}]


def test_only_circular_c128_builds_no_boundary_fast_path_metadata(monkeypatch) -> None:
    from vllm.models.deepseek_v4 import compressor

    monkeypatch.setattr(compressor.current_platform, "is_cuda", lambda: True)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=128,
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
        return builder.build(0, common)

    circular = _state_cache(128).get_kv_cache_spec(_config())
    assert build(circular).c128_boundary is False

    c4 = _state_cache(4).get_kv_cache_spec(_config())
    assert build(c4).c128_boundary is None


def test_non_circular_two_stage_uses_paged_prefill_path(monkeypatch) -> None:
    from vllm.models.deepseek_v4.common.ops import fused_compress_quant_cache

    prefill_calls = []
    decode_calls = []
    monkeypatch.setattr(
        fused_compress_quant_cache,
        "_launch_two_stage_sparse_attn_compressor",
        lambda **kwargs: prefill_calls.append(kwargs),
    )
    monkeypatch.setattr(
        fused_compress_quant_cache,
        "compress_norm_rope_store_triton",
        lambda **kwargs: decode_calls.append(kwargs),
    )
    token_to_req = torch.zeros(3, dtype=torch.int32)
    positions = torch.tensor([0, 127, 255])
    slot_mapping = torch.arange(3)
    block_table = torch.arange(32).reshape(1, 32)

    fused_compress_quant_cache.compress_norm_rope_store_two_stage_triton(
        state_cache=torch.empty(32, 8, 1024),
        kv=torch.empty(3, 512),
        score=torch.empty(3, 512),
        ape=torch.empty(128, 512),
        num_actual=3,
        token_to_req_indices=token_to_req,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=8,
        query_start_loc=torch.tensor([0, 3]),
        is_circular=False,
        state_width=512,
        cos_sin_cache=torch.empty(256, 64),
        kv_cache=torch.empty(1, 1),
        k_cache_metadata=SimpleNamespace(slot_mapping=torch.arange(3)),
        pdl_kwargs={},
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=128,
        overlap=False,
        use_fp4_cache=False,
        rms_norm_weight=torch.empty(512),
        rms_norm_eps=1e-6,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
        num_decode_tokens=1,
        compress_scratch=torch.empty(3, 512),
    )

    assert len(prefill_calls) == 1
    assert prefill_calls[0]["block_table"] is block_table
    assert prefill_calls[0]["block_size"] == 8
    assert torch.equal(prefill_calls[0]["positions"], positions[1:])
    assert (
        not {
            "kv",
            "score",
            "ape",
            "query_start_loc",
            "token_offset",
        }
        & prefill_calls[0].keys()
    )
    assert len(decode_calls) == 1
    assert decode_calls[0]["num_actual"] == 1
    assert decode_calls[0]["is_circular"] is False


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
