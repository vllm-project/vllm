# SPDX-License-Identifier: Apache-2.0
# Standard
from functools import partial
import os
import random
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.xpu_connectors import VLLMPagedMemXPUConnectorV2
from tests.v1.utils import (
    dumb_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
)

if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip(
        "Requires available xpu runtime",
        allow_module_level=True,
    )

pytestmark = pytest.mark.xpu
BACKENDS = ["cpu", "disk"]
DEVICE = torch.device(torch_device_type)
# Optional override for tempfile root; see tests/v1/test_cache_engine.py
# for rationale.
_TEST_TMPDIR = os.environ.get("LMCACHE_TEST_TMPDIR") or None


def generate_random_slot_mapping(
    num_blocks: int,
    block_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    return torch.tensor(slot_mapping, device=device)


def get_expected_count(token_len: int, save_unfull_chunk: bool, chunk_size: int) -> int:
    if save_unfull_chunk:
        return token_len
    return (token_len // chunk_size) * chunk_size


def _wait_for_store(
    engine,
    tokens: list[torch.Tensor],
    expected: int,
    timeout: int = 60,
) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if all(engine.lookup(t) == expected for t in tokens):
            return
        time.sleep(0.1)
    raise TimeoutError(f"Store operation has not finished in {timeout} seconds")


def _create_connector(
    device: torch.device,
    hidden_dim: int,
    num_layers: int,
    *,
    use_gpu: bool,
    chunk_size: int,
    dtype: torch.dtype,
    use_mla: bool = False,
):
    return VLLMPagedMemXPUConnectorV2(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        use_mla=use_mla,
        device=device,
    )


def _v2_store_vllm_contract(
    engine,
    list_token_ids,
    list_slot_mappings,
    kvcaches,
    *,
    chunk_size: int,
    save_unfull_chunk: bool,
):
    """Mimic vLLM non-layerwise store contract for V2 connector."""
    for token_ids, slot_mapping in zip(
        list_token_ids, list_slot_mappings, strict=False
    ):
        tokens = token_ids
        slots = slot_mapping.to(kvcaches[0][0].device)

        if save_unfull_chunk:
            aligned_len = len(tokens)
        else:
            aligned_len = (len(tokens) // chunk_size) * chunk_size
            tokens = tokens[:aligned_len]
            slots = slots[:aligned_len]

        if aligned_len == 0:
            continue

        engine.store(tokens, kvcaches=kvcaches, slot_mapping=slots)


def _v2_retrieve_vllm_contract(
    engine,
    list_token_ids,
    list_slot_mappings,
    kvcaches,
    *,
    chunk_size: int,
    save_unfull_chunk: bool,
):
    """Mimic vLLM non-layerwise retrieve contract for V2 connector."""
    for token_ids, slot_mapping in zip(
        list_token_ids, list_slot_mappings, strict=False
    ):
        tokens = token_ids
        slots = slot_mapping.to(kvcaches[0][0].device)

        if save_unfull_chunk:
            aligned_len = len(tokens)
        else:
            aligned_len = (len(tokens) // chunk_size) * chunk_size
            tokens = tokens[:aligned_len]
            slots = slots[:aligned_len]

        if aligned_len == 0:
            continue

        engine.retrieve(tokens, kvcaches=kvcaches, slot_mapping=slots)


@pytest.fixture
def create_config():
    def make_config(
        backend: str, size: float, save_unfull_chunk: bool = True, **kwargs
    ):
        common = dict(
            save_unfull_chunk=save_unfull_chunk,
            extra_config={"force_store_wait": True},
            use_layerwise=False,
        )

        match backend:
            case "cpu":
                return LMCacheEngineConfig.from_defaults(
                    local_cpu=True,
                    max_local_cpu_size=size,
                    **common,
                )
            case "disk":
                assert "path" in kwargs, "'path' is missing for disk backend"
                return LMCacheEngineConfig.from_defaults(
                    local_cpu=False,
                    max_local_cpu_size=size,
                    local_disk=kwargs["path"],
                    max_local_disk_size=size,
                    **common,
                )
            case _:
                raise ValueError(f"Unknown backend: {backend}")

    with tempfile.TemporaryDirectory(
        dir=_TEST_TMPDIR, ignore_cleanup_errors=True
    ) as temp_dir:
        yield partial(make_config, path=temp_dir)


def _build_engine(
    *,
    name: str,
    cfg,
    connector,
    num_layers: int,
    chunk_size: int,
    num_heads: int,
    head_dim: int,
    autorelease_v1,
):
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)
    return autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            name,
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )


@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="store-v2")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_store_1gb_v2(
    benchmark,
    backend,
    use_gpu,
    save_unfull_chunk,
    create_config,
    autorelease_v1,
):
    num_heads = 8
    head_dim = 128
    num_layers = 32
    dtype = torch.bfloat16

    num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    chunk_size = 256

    num_requests = 4
    num_repeats = 10

    connector = _create_connector(
        DEVICE,
        hidden_dim=num_heads * head_dim,
        num_layers=num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        use_mla=False,
    )

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        DEVICE,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_dim,
    )

    cfg = create_config(backend, 1.5, save_unfull_chunk=save_unfull_chunk)
    engine = _build_engine(
        name="test",
        cfg=cfg,
        connector=connector,
        num_layers=num_layers,
        chunk_size=chunk_size,
        num_heads=num_heads,
        head_dim=head_dim,
        autorelease_v1=autorelease_v1,
    )

    expected = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)

    def run_func(tokens, slot_mappings):
        _v2_store_vllm_contract(
            engine,
            tokens,
            slot_mappings,
            kv_cache,
            chunk_size=chunk_size,
            save_unfull_chunk=save_unfull_chunk,
        )
        _wait_for_store(engine, tokens, expected)

    def setup():
        list_tokens = [generate_tokens(num_tokens, DEVICE) for _ in range(num_requests)]
        list_slot_mappings = [
            generate_random_slot_mapping(num_blocks, block_size, num_tokens, DEVICE)
            for _ in range(num_requests)
        ]
        return (list_tokens, list_slot_mappings), {}

    warm_tokens, warm_slot_mappings = setup()[0]
    run_func(warm_tokens, warm_slot_mappings)
    benchmark.pedantic(run_func, setup=setup, rounds=num_repeats, iterations=1)


@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="retrieve-v2")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_retrieve_1gb_allhit_v2(
    benchmark,
    backend,
    use_gpu,
    save_unfull_chunk,
    create_config,
    autorelease_v1,
):
    num_heads = 8
    head_dim = 128
    num_layers = 32
    dtype = torch.bfloat16

    num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    chunk_size = 256

    num_requests = 4
    num_repeats = 10

    connector = _create_connector(
        DEVICE,
        hidden_dim=num_heads * head_dim,
        num_layers=num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        use_mla=False,
    )

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        DEVICE,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_dim,
    )

    list_tokens = [generate_tokens(num_tokens, DEVICE) for _ in range(num_requests)]
    list_slot_mappings = [
        generate_random_slot_mapping(num_blocks, block_size, num_tokens, DEVICE)
        for _ in range(num_requests)
    ]

    cfg = create_config(backend, 1.5, save_unfull_chunk=save_unfull_chunk)
    engine = _build_engine(
        name="test",
        cfg=cfg,
        connector=connector,
        num_layers=num_layers,
        chunk_size=chunk_size,
        num_heads=num_heads,
        head_dim=head_dim,
        autorelease_v1=autorelease_v1,
    )

    _v2_store_vllm_contract(
        engine,
        list_tokens,
        list_slot_mappings,
        kv_cache,
        chunk_size=chunk_size,
        save_unfull_chunk=save_unfull_chunk,
    )
    expected = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    _wait_for_store(engine, list_tokens, expected)

    def run_func():
        _v2_retrieve_vllm_contract(
            engine,
            list_tokens,
            list_slot_mappings,
            kv_cache,
            chunk_size=chunk_size,
            save_unfull_chunk=save_unfull_chunk,
        )

    run_func()
    benchmark.pedantic(run_func, rounds=num_repeats, iterations=1)


@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="lookup-v2")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_lookup_20k_tokens_v2(
    benchmark,
    backend,
    use_gpu,
    save_unfull_chunk,
    create_config,
    autorelease_v1,
):
    num_heads = 8
    head_dim = 128
    num_layers = 32
    dtype = torch.bfloat16

    num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    chunk_size = 256

    # Match layerwise benchmark's request count cap for use_gpu=True.
    num_requests = 8 if use_gpu else 10
    num_repeats = 10

    connector = _create_connector(
        DEVICE,
        hidden_dim=num_heads * head_dim,
        num_layers=num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        use_mla=False,
    )

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        DEVICE,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_dim,
    )

    list_tokens = [generate_tokens(num_tokens, DEVICE) for _ in range(num_requests)]
    list_slot_mappings = [
        generate_random_slot_mapping(num_blocks, block_size, num_tokens, DEVICE)
        for _ in range(num_requests)
    ]

    cfg = create_config(backend, 3.0, save_unfull_chunk=save_unfull_chunk)
    engine = _build_engine(
        name="test",
        cfg=cfg,
        connector=connector,
        num_layers=num_layers,
        chunk_size=chunk_size,
        num_heads=num_heads,
        head_dim=head_dim,
        autorelease_v1=autorelease_v1,
    )

    _v2_store_vllm_contract(
        engine,
        list_tokens,
        list_slot_mappings,
        kv_cache,
        chunk_size=chunk_size,
        save_unfull_chunk=save_unfull_chunk,
    )
    expected = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    _wait_for_store(engine, list_tokens, expected)

    def run_func():
        for t in list_tokens:
            engine.lookup(t)

    benchmark.pedantic(run_func, iterations=1, rounds=num_repeats)
