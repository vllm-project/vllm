# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU cache-adapter restoration and independent-slot integration."""

import importlib
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression.adapter import (
    build_memory_plan,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
runtime = (
    importlib.import_module("kvserve_v1.compression.runtime")
    if os.environ.get("VLLM_TEST_REQUIRE_MOONCAKE_COMPRESSION") == "1"
    else pytest.importorskip("kvserve_v1.compression.runtime")
)
CompressionRuntime = runtime.CompressionRuntime
CompressionConfig = runtime.CompressionConfig
cpp_backend = importlib.import_module("kvserve_v1.compression.runtime.cpp_backend")
LAYER_NAME = "model.layers.0.self_attn"


@pytest.fixture
def make_manager(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.compression import (
        adapter,
    )

    monkeypatch.setattr(adapter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(adapter, "get_current_attn_backends", lambda config: [])
    monkeypatch.setattr(
        adapter,
        "get_attn_backends_and_specs_for_layers",
        lambda config, specs, backends: (backends, [None] * len(backends)),
    )
    torch.manual_seed(0)
    created = []

    def factory(
        compression=None,
        num_layers=1,
        raw_aggregate=None,
        blocks_per_logical=1,
        block_size=16,
        head_size=8,
    ):
        extra = {
            "compression": {
                "enable-compression-pipeline": True,
                "split_type": "layer",
                **(compression or {}),
            }
        }
        if raw_aggregate is not None:
            extra["raw_aggregate"] = raw_aggregate
        config = CompressionConfig.from_extra_config(extra)
        dependencies = []
        if config.uses_codec:
            dependencies.append("nvidia.nvcomp")
        if config.uses_transformer and not cpp_backend.enabled():
            dependencies.append("humming.ops")
        for dependency in dependencies:
            if os.environ.get("VLLM_TEST_REQUIRE_MOONCAKE_COMPRESSION") == "1":
                importlib.import_module(dependency)
            else:
                pytest.importorskip(dependency)
        monkeypatch.setattr(
            adapter,
            "select_common_block_size",
            lambda size, backends, specs: size // blocks_per_logical,
        )
        vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            cache_config=SimpleNamespace(block_size=block_size),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_hidden_layers=num_layers)
            ),
        )
        specs = {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=2,
                head_size=head_size,
                dtype=torch.bfloat16,
            )
            for i in range(num_layers)
        }
        plan = build_memory_plan(vllm_config, specs)
        assert plan is not None
        manager = CompressionRuntime(
            plan, "LBHNC", blocks_per_logical, torch.device("cuda", 0)
        )
        created.append(manager)
        caches = {
            name: torch.rand(
                4 * blocks_per_logical,
                2,
                block_size // blocks_per_logical,
                2 * head_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for name in specs
        }
        manager.bind_kv_caches(caches, dict.fromkeys(specs, 0))
        return manager, caches[LAYER_NAME]

    yield factory
    for manager in reversed(created):
        manager.shutdown()


def _copy_wire(producer, consumer, encoded, part, tokens):
    source, target = producer.slots[part], consumer.slots[1 - part]
    target.arena_a[: encoded.codec_bytes].copy_(source.arena_a[: encoded.codec_bytes])
    aux = producer.aux_bytes(part, {0: tokens})
    target.aux[:aux].copy_(source.aux[:aux])
    target.raw_tail[: encoded.raw_tail_bytes].copy_(
        source.raw_tail[: encoded.raw_tail_bytes]
    )
    torch.accelerator.synchronize()


def _assert_restored(producer, consumer, part, tokens, atol=0.06):
    factor = producer.pipelines[0].blocks_per_logical
    block_size = producer.layout.layer_parts[0].block_size
    count = (tokens + block_size - 1) // block_size
    source_ids = [b * factor + i for b in [2, 0][:count] for i in range(factor)]
    target_ids = [b * factor + i for b in [1, 3][:count] for i in range(factor)]
    untouched = sorted(set(range(4 * factor)) - set(target_ids))
    for name, source in producer.pipelines[0].kv_caches.items():
        target = consumer.pipelines[0].kv_caches[name]
        head_size = source.shape[-1] // 2
        section = slice(part * head_size, (part + 1) * head_size)
        expected = source[source_ids, :, :, section].permute(0, 2, 1, 3).flatten(0, 1)
        actual = target[target_ids, :, :, section].permute(0, 2, 1, 3).flatten(0, 1)
        stages = producer.config.pipeline
        tolerance = atol if "quantizer" in stages or "transformer" in stages else 0
        torch.testing.assert_close(
            actual[:tokens], expected[:tokens], atol=tolerance, rtol=0
        )
        if "aggregate" not in stages:
            tail = tokens // block_size * block_size
            assert torch.equal(actual[tail:tokens], expected[tail:tokens])
        assert (target[untouched] == -9).all()


@pytest.mark.parametrize(
    "stages,tokens,factor,block_size",
    [
        (None, 31, 2, 16),
        (["codec"], 17, 2, 16),
    ],
)
def test_peer_buffers_restore_layers_blocks_and_partial_tail(
    make_manager, stages, tokens, factor, block_size
):
    config = {"logical_blocks_per_chunk": 2}
    if stages is not None:
        config.update(
            pipeline=stages,
            **dict.fromkeys(
                (
                    "high_key_num_levels",
                    "low_key_num_levels",
                    "high_value_num_levels",
                    "low_value_num_levels",
                ),
                256,
            ),
        )
    num_layers = 1 if stages is None else 3
    producer, _ = make_manager(
        config, num_layers=num_layers, blocks_per_logical=factor, block_size=block_size
    )
    consumer, _ = make_manager(
        config, num_layers=num_layers, blocks_per_logical=factor, block_size=block_size
    )
    for target in consumer.pipelines[0].kv_caches.values():
        target.fill_(-9)
    torch.accelerator.synchronize()
    count = (tokens + block_size - 1) // block_size
    for part in (0, 1):
        encoded = producer.encode(
            part, [[2, 0][:count]], part, valid_tokens_by_group={0: tokens}
        )
        encoded.event.synchronize()
        _copy_wire(producer, consumer, encoded, part, tokens)
        consumer.decode(
            1 - part,
            [[1, 3][:count]],
            part,
            encoded.codec_bytes,
            encoded.u8_bytes,
            valid_tokens_by_group={0: tokens},
        ).synchronize()
        _assert_restored(
            producer, consumer, part, tokens, atol=0.3 if stages is None else 0.06
        )


def test_concurrent_slots_do_not_cross_request_writes(make_manager):
    manager, cache = make_manager({"pipeline": ["codec"], "slot_count": 2})
    expected = cache.clone()
    torch.accelerator.synchronize()

    def encode(slot):
        with torch.accelerator.device_index(0):
            return manager.encode(slot, [[slot]], 0, valid_tokens_by_group={0: 16})

    def decode(slot, encoded):
        with torch.accelerator.device_index(0):
            return manager.decode(
                slot,
                [[slot]],
                0,
                encoded.codec_bytes,
                encoded.u8_bytes,
                valid_tokens_by_group={0: 16},
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        chunks = list(pool.map(encode, (0, 1)))
        for chunk in chunks:
            chunk.event.synchronize()
        cache[:2, :, :, :8].zero_()
        torch.accelerator.synchronize()
        for event in pool.map(decode, (0, 1), chunks):
            event.synchronize()
    assert torch.equal(cache, expected)
