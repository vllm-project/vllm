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
from lmcache.v1.gpu_connector.xpu_connectors import VLLMPagedMemLayerwiseXPUConnector
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

BACKENDS = ["cpu", "disk"]

pytestmark = pytest.mark.xpu

# Optional override for tempfile root; see tests/v1/test_cache_engine.py
# for rationale.
_TEST_TMPDIR = os.environ.get("LMCACHE_TEST_TMPDIR") or None


def generate_random_slot_mapping(num_blocks, block_size, num_tokens, device):
    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    return torch.tensor(slot_mapping, device=device)


def get_expected_count(token_len, save_unfull_chunk, chunk_size):
    # expected tokens stored (not chunks)
    if save_unfull_chunk:
        return token_len
    return (token_len // chunk_size) * chunk_size


def _wait_for_store(engine, tokens, expected, timeout=60):
    start = time.time()
    last_hits = []
    while time.time() - start < timeout:
        last_hits = [engine.lookup(t) for t in tokens]
        ready = all(hit == expected for hit in last_hits)
        if ready:
            return
        time.sleep(0.1)
    raise TimeoutError(
        "Store operation hasn't finished in "
        f"{timeout} seconds. expected={expected}, hits={last_hits}"
    )


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
    return VLLMPagedMemLayerwiseXPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )


def _layerwise_store_vllm_contract(
    engine,
    list_token_ids,  # list[Tensor] or list[list[int]]
    list_slot_mappings,  # list[Tensor]
    kvcaches,
    *,
    num_layers: int,
    chunk_size: int,
    save_unfull_chunk: bool,
):
    """
    Mimic vLLM layerwise store contract:

      - create one store_layer generator per request
        (first one sync=True, rest sync=False)
      - tick generators once per layer (num_layers times)
      - tick generators one final time (like wait_for_save)

    IMPORTANT:
      - store_layer expects token_ids as list[int] (like vLLM adapter)
      - for save_unfull_chunk=False, truncate to aligned_len
        instead of trailing False mask
      - mask False indicates skipped LEADING tokens only (we use skip=0 here)
    """
    storers = []
    is_first = True

    for token_ids, slot_mapping in zip(
        list_token_ids, list_slot_mappings, strict=False
    ):
        # token_ids: Tensor -> list[int]
        token_ids_list = (
            token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        )

        # keep slot_mapping on same device as kvcaches (xpu)
        slot_mapping = slot_mapping.to(kvcaches[0].device)

        if save_unfull_chunk:
            aligned_len = len(token_ids_list)
        else:
            aligned_len = (len(token_ids_list) // chunk_size) * chunk_size
            token_ids_list = token_ids_list[:aligned_len]
            slot_mapping = slot_mapping[:aligned_len]

        if aligned_len == 0:
            continue

        skip_leading_tokens = 0
        # put mask on same device (avoid implicit device moves)
        mask = torch.ones(
            len(token_ids_list), dtype=torch.bool, device=slot_mapping.device
        )
        mask[:skip_leading_tokens] = False

        st = engine.store_layer(
            token_ids_list,
            mask=mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            sync=is_first,
        )
        storers.append(st)
        is_first = False

    # Tick once per layer
    for _ in range(num_layers):
        for st in storers:
            next(st)

    # Finalize tick (equivalent of wait_for_save)
    for st in storers:
        next(st)


def _layerwise_retrieve_vllm_contract(
    engine,
    list_token_ids,
    list_slot_mappings,
    kvcaches,
    *,
    num_layers: int,
    chunk_size: int,
    save_unfull_chunk: bool,
):
    """Mimic vLLM layerwise retrieve contract.

    For layerwise mode, benchmark should call retrieve_layer with the same
    slot mapping + KV cache context used by GPU connector.
    """
    retrievers = []
    is_first = True

    for token_ids, slot_mapping in zip(
        list_token_ids, list_slot_mappings, strict=False
    ):
        if isinstance(token_ids, torch.Tensor):
            tokens = token_ids
        else:
            tokens = torch.tensor(
                token_ids, dtype=torch.long, device=slot_mapping.device
            )

        slot_mapping = slot_mapping.to(kvcaches[0].device)

        if save_unfull_chunk:
            aligned_len = len(tokens)
        else:
            aligned_len = (len(tokens) // chunk_size) * chunk_size
            tokens = tokens[:aligned_len]
            slot_mapping = slot_mapping[:aligned_len]

        if aligned_len == 0:
            continue

        mask = torch.ones(aligned_len, dtype=torch.bool, device=slot_mapping.device)
        retriever = engine.retrieve_layer(
            tokens,
            mask=mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            sync=is_first,
        )
        retrievers.append(retriever)
        is_first = False

    # Tick once per layer to overlap multi-request loading/copying.
    for _ in range(num_layers):
        for retriever in retrievers:
            next(retriever)

    # Finalize connector sync stage.
    for retriever in retrievers:
        next(retriever)

    # Consume final ret_mask yield.
    for retriever in retrievers:
        next(retriever)


@pytest.fixture
def create_config():
    def make_config(backend, size, save_unfull_chunk=True, **kwargs):
        # NOTE: use_layerwise=True because we use store_layer contract
        common = dict(
            save_unfull_chunk=save_unfull_chunk,
            extra_config={"force_store_wait": True},  # deterministic for benchmarks
            use_layerwise=True,
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
    # metadata shape used by tests
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


# --------------------------
# Store benchmarks
# --------------------------
@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="store")
@pytest.mark.parametrize("backend", ["cpu", "disk"])
@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_store_1GB(
    benchmark,
    backend,
    use_gpu,
    save_unfull_chunk,
    create_config,
    autorelease_v1,
):
    """
    Store benchmark for XPU layerwise connector.

    Reduces volatility by:
      - warming up once outside timing
      - reusing the same engine/backend state
      - shuffling request order per round
    """
    # model-related metadata
    num_heads = 8
    head_dim = 128
    num_layers = 32
    dtype = torch.bfloat16

    # lmcache / vllm configs
    device = torch.device(torch_device_type)

    num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    chunk_size = 256
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    # benchmark configs
    num_requests = 4
    num_repeats = 10

    connector = _create_connector(
        device,
        hidden_dim=num_heads * head_dim,
        num_layers=num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        use_mla=False,
    )

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_dim,
    )

    cache_size = 1.5
    cfg = create_config(backend, cache_size, save_unfull_chunk=save_unfull_chunk)

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "test",
            cfg,
            dumb_metadata(kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    expected = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)

    def run_func(tokens, slot_mappings):
        _layerwise_store_vllm_contract(
            engine,
            tokens,
            slot_mappings,
            kv_cache,
            num_layers=num_layers,
            chunk_size=chunk_size,
            save_unfull_chunk=save_unfull_chunk,
        )
        _wait_for_store(engine, tokens, expected)

    def setup():
        list_tokens = [generate_tokens(num_tokens, device) for _ in range(num_requests)]
        list_slot_mappings = [
            generate_random_slot_mapping(num_blocks, block_size, num_tokens, device)
            for _ in range(num_requests)
        ]
        return (
            list_tokens,
            list_slot_mappings,
        ), {}

    # Warm up once outside timing to absorb first-touch XPU/backend overhead.
    warm_tokens, warm_slot_mappings = setup()[0]
    run_func(warm_tokens, warm_slot_mappings)
    benchmark.pedantic(
        run_func,
        setup=setup,
        rounds=num_repeats,
        iterations=1,
    )


# --------------------------
# Retrieve benchmarks (100% hit)
# --------------------------
@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="retrieve")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_retrieve_1GB_allhit(
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

    device = torch.device(torch_device_type)

    num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    chunk_size = 256

    num_requests = 4
    num_repeats = 10

    connector = _create_connector(
        device,
        hidden_dim=num_heads * head_dim,
        num_layers=num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        use_mla=False,
    )

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_dim,
    )
    kvcaches = kv_cache

    list_tokens = [generate_tokens(num_tokens, device) for _ in range(num_requests)]
    list_slot_mappings = [
        generate_random_slot_mapping(num_blocks, block_size, num_tokens, device)
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

    # Pre-populate cache once (not timed)
    _layerwise_store_vllm_contract(
        engine,
        list_tokens,
        list_slot_mappings,
        kvcaches,
        num_layers=num_layers,
        chunk_size=chunk_size,
        save_unfull_chunk=save_unfull_chunk,
    )
    expected = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    _wait_for_store(engine, list_tokens, expected)

    def run_func():
        _layerwise_retrieve_vllm_contract(
            engine,
            list_tokens,
            list_slot_mappings,
            kvcaches,
            num_layers=num_layers,
            chunk_size=chunk_size,
            save_unfull_chunk=save_unfull_chunk,
        )

    # Warm up once outside timing to absorb first-touch retrieve overhead.
    run_func()

    benchmark.pedantic(run_func, rounds=num_repeats, iterations=1)


# --------------------------
# Lookup benchmarks (100% hit)
# 10 rounds, 1 iteration (requested)
# --------------------------
@pytest.mark.no_shared_allocator
@pytest.mark.benchmark(group="lookup")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_lookup_20K_tokens(
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

    device = torch.device(torch_device_type)

    num_tokens = 2000
    num_blocks = 1000
    block_size = 16
    chunk_size = 256

    # use_gpu=True stages per-request chunks on XPU during layerwise store.
    # Keeping 10 requests can exceed the default staging pool in pre-population.
    num_requests = 8 if use_gpu else 10
    num_repeats = 10

    connector = _create_connector(
        device,
        hidden_dim=num_heads * head_dim,
        num_layers=num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        use_mla=False,
    )

    kv_cache = generate_kv_cache_paged_list_tensors(
        num_blocks,
        device,
        block_size,
        dtype,
        num_layers=num_layers,
        head_size=head_dim,
    )
    kvcaches = kv_cache

    list_tokens = [generate_tokens(num_tokens, device) for _ in range(num_requests)]
    list_slot_mappings = [
        generate_random_slot_mapping(num_blocks, block_size, num_tokens, device)
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

    # Pre-populate once (not timed)
    _layerwise_store_vllm_contract(
        engine,
        list_tokens,
        list_slot_mappings,
        kvcaches,
        num_layers=num_layers,
        chunk_size=chunk_size,
        save_unfull_chunk=save_unfull_chunk,
    )
    expected = get_expected_count(num_tokens, save_unfull_chunk, chunk_size)
    _wait_for_store(engine, list_tokens, expected)

    def run_func():
        # 1 iteration per round, 10 rounds (requested)
        for t in list_tokens:
            engine.lookup(t)

    benchmark.pedantic(run_func, iterations=1, rounds=num_repeats)
