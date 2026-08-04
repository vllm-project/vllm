# SPDX-License-Identifier: Apache-2.0
"""End-to-end benchmark: CacheGen encode/decode on GPU (CUDA or XPU).

Measures wall-clock to serialize and deserialize a synthetic KV-cache
blob, and records the compression ratio against the uncompressed baseline.

Run with: ``pytest tests/benchmarks/test_cachegen.py --benchmark-only``
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata

# CacheGen has hand-written accelerator kernels and requires the selected
# torch runtime to be available.
if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


def _generate_kv(num_tokens, device):
    num_layers, num_heads, head_size = 32, 8, 128
    shape = [num_tokens, num_heads, head_size]
    pairs = []
    for _ in range(num_layers):
        k = torch.rand(shape, dtype=torch.bfloat16, device=device)
        v = torch.rand(shape, dtype=torch.bfloat16, device=device)
        pairs.append((k, v))
    return torch.stack([torch.stack(p, dim=0) for p in pairs], dim=0)


def _make_serde(chunk_size):
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    return (
        CacheGenSerializer(config, metadata),
        CacheGenDeserializer(config, metadata, torch.bfloat16),
    )


@pytest.mark.benchmark(group="cachegen_encode")
@pytest.mark.parametrize("chunk_size", [64, 128, 256, 768])
def test_cachegen_encoder_bench(benchmark, chunk_size):
    serializer, _ = _make_serde(chunk_size)
    kv = _generate_kv(chunk_size, torch_device_type)

    def run():
        return serializer.to_bytes(kv)

    out = benchmark(run)
    raw_bytes = kv.element_size() * kv.numel()
    print(
        f"\n[chunk={chunk_size}] raw={raw_bytes} compressed={len(out)} "
        f"ratio={raw_bytes / max(len(out), 1):.2f}x"
    )


@pytest.mark.benchmark(group="cachegen_decode")
@pytest.mark.parametrize("chunk_size", [64, 128, 256, 768])
def test_cachegen_decoder_bench(benchmark, chunk_size):
    serializer, deserializer = _make_serde(chunk_size)
    kv = _generate_kv(chunk_size, torch_device_type)
    payload = serializer.to_bytes(kv)

    benchmark(deserializer.from_bytes, payload)
