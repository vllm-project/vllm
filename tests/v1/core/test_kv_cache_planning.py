# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import mmap
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.core.kv_cache_planning as kv_cache_planning
import vllm.v1.core.kv_cache_utils as kv_cache_utils
import vllm.v1.hisparse.runtime as hisparse_runtime_module
from tests.v1.core.utils import (
    new_kv_cache_spec,
    new_mamba_spec,
    new_mla_spec,
    new_sliding_window_spec,
)
from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.attention import HiSparseConfig
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_planning import (
    check_enough_kv_cache_memory,
    estimate_max_model_len,
    get_kv_cache_configs,
    get_kv_cache_groups,
    is_kv_cache_spec_uniform,
    kv_cache_groups_tp_replicas,
)
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
)
from vllm.v1.hisparse.layout import (
    get_hisparse_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("gpu_block_size", [32, 64])
@pytest.mark.parametrize("shared_host_pool", [False, True])
def test_hisparse_hma_uses_resolved_gpu_block_size(
    monkeypatch, gpu_block_size, shared_host_pool
):
    monkeypatch.setattr(
        hisparse_runtime_module.current_platform, "is_cuda_alike", lambda: True
    )
    specs = {
        "model.layers.0.self_attn": MLAAttentionSpec(
            block_size=gpu_block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_index_group_leader=True,
        ),
        "model.layers.0.self_attn.indexer": MLAAttentionSpec(
            block_size=gpu_block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        ),
    }
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=HiSparseConfig()),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=128),
            max_model_len=gpu_block_size,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2 if shared_host_pool else 1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            world_size=2 if shared_host_pool else 1,
            distributed_executor_backend="mp",
            nnodes_within_dp=1,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=7,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
        ),
    )
    monkeypatch.setattr(
        kv_cache_planning, "get_hisparse_host_pool_bytes", lambda _: 2**30
    )
    cache_config = kv_cache_planning.get_kv_cache_config_from_groups(
        config, get_hisparse_kv_cache_groups(config, specs), available_memory=2**30
    )
    assert cache_config.num_blocks == 7
    assert cache_config.hisparse_host_num_blocks is not None
    assert cache_config.hisparse_host_num_blocks > 7

    host_group, indexer_group, *auxiliary_groups = cache_config.kv_cache_groups
    assert cache_config.hisparse_shared_host_pool is shared_host_pool
    host_page = host_group.kv_cache_spec.page_size_bytes
    alignment = mmap.PAGESIZE if shared_host_pool else 1
    expected_host_stride = (host_page + alignment - 1) // alignment * alignment
    assert cache_config.hisparse_host_block_stride == expected_host_stride
    assert cache_config.hisparse_host_num_blocks == 2**30 // expected_host_stride
    assert host_group.host_resident
    assert not any(group.host_resident for group in [indexer_group, *auxiliary_groups])
    host_layers = set(host_group.layer_names)
    for tensor in cache_config.kv_cache_tensors:
        assert all(
            (name in host_layers) == tensor.host_resident for name in tensor.layers
        )
    host_tensor = next(t for t in cache_config.kv_cache_tensors if t.host_resident)
    host_spec = host_group.kv_cache_spec.kv_cache_specs[host_tensor.layers[0]]
    assert host_tensor.block_stride == host_spec.page_size_bytes
    assert host_tensor.layer_stride == (
        host_spec.page_size_bytes * cache_config.hisparse_host_num_blocks
    )
    assert host_group.kv_cache_spec.block_size == gpu_block_size
    assert indexer_group.kv_cache_spec.block_size == gpu_block_size
    host_specs = host_group.kv_cache_spec.kv_cache_specs
    gpu_indexer_specs = indexer_group.kv_cache_spec.kv_cache_specs
    assert set(host_specs) == {"model.layers.0.self_attn"}
    assert set(gpu_indexer_specs) == {"model.layers.0.self_attn.indexer"}
    assert indexer_group.kv_cache_spec.prefix_cacheable
    assert host_group.enable_kv_transfer
    auxiliary_specs = [group.kv_cache_spec for group in auxiliary_groups]
    assert any(isinstance(spec, HiSparseResidentSpec) for spec in auxiliary_specs)
    assert any(isinstance(spec, HiSparseHotSpec) for spec in auxiliary_specs)
    assert all(
        spec.block_size == gpu_block_size
        for spec in auxiliary_specs
        if isinstance(spec, (HiSparseResidentSpec, HiSparseHotSpec))
    )
    scheduler_block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        cache_config,
        SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=16,
                enable_prefix_caching=True,
                prefix_match_unit=None,
            ),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            kv_transfer_config=object(),
        ),
    )
    assert scheduler_block_size == hash_block_size == gpu_block_size


@pytest.mark.parametrize(
    "max_model_len,num_gpu_blocks,ok",
    [
        (8192, 160, False),
        (8192, 1000, True),
        (32768, 656, False),
        (32768, 657, True),
        (-1, 657, True),
    ],
)
def test_hisparse_pool_must_fit_max_model_len(
    monkeypatch, max_model_len, num_gpu_blocks, ok
):
    """Each resident group takes its own blocks from HiSparse's shared GPU pool,
    so a pool that fits only the indexer pages must be rejected at startup.
    Resident pages are capped at the in-flight window: at 32768 tokens with 2048
    in flight, one request needs 512 indexer + 4 x 32 resident + 16 hot blocks,
    plus the null block. Auto-fit (-1) must pick a length that passes the same
    check."""
    monkeypatch.setattr(
        hisparse_runtime_module.current_platform, "is_cuda_alike", lambda: True
    )
    monkeypatch.setattr(
        kv_cache_planning, "get_hisparse_host_pool_bytes", lambda _: 2**30
    )
    specs: dict[str, KVCacheSpec] = {}
    for i in range(4):
        specs[f"model.layers.{i}.self_attn"] = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_index_group_leader=True,
        )
        specs[f"model.layers.{i}.self_attn.indexer"] = MLAAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        )
    config = VllmConfig(model_config=ModelConfig(max_model_len=max_model_len))
    config.scheduler_config.max_num_batched_tokens = 2048
    # Async scheduling doubles the in-flight window, and with it the bound.
    config.scheduler_config.async_scheduling = False
    config.attention_config.hisparse_config = HiSparseConfig()
    config.model_config.hf_config.index_topk = 128
    config.cache_config.num_gpu_blocks_override = num_gpu_blocks
    config.cache_config.kv_cache_layout = "BLHNC"
    # Workers report the resident/hot caches HiSparse derives.
    specs |= {
        name: group.kv_cache_spec
        for group in get_hisparse_kv_cache_groups(config, specs) or []
        if isinstance(group.kv_cache_spec, (HiSparseResidentSpec, HiSparseHotSpec))
        for name in group.layer_names
    }

    if ok:
        kv_cache_planning.get_kv_cache_configs(config, [specs], [2**34])
    else:
        with pytest.raises(ValueError, match="max seq len"):
            kv_cache_planning.get_kv_cache_configs(config, [specs], [2**34])


def new_chunked_local_attention_spec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.float32,
    page_size_padded=None,
    attention_chunk_size=4,
):
    return ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_size_padded,
        attention_chunk_size=attention_chunk_size,
    )


def test_get_kv_cache_configs_multiple_workers():
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.prefix_cache_retention_interval = None

    ref_kv_cache_spec = new_kv_cache_spec()
    same_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
    ]

    # Basic case. All things are the same.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    expected = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                layers=["layer1", "layer2"],
                layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                block_stride=ref_kv_cache_spec.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
        ],
    )
    assert kv_cache_configs == [expected, expected]

    # Different available memory. This is the case for TP.
    # Use the smallest memory available.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        same_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 20,
        ],
    )
    assert kv_cache_configs == [expected, expected]

    # Different KV cache specs. This is the case for PP.
    different_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer2": new_kv_cache_spec(),
            "layer3": new_kv_cache_spec(),
        },
    ]

    # Different workers have different layers.
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer1"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], new_kv_cache_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                    layers=["layer2", "layer3"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer2", "layer3"], new_kv_cache_spec()),
            ],
        ),
    ]

    # Some layers are the same, some are different. This is the case for TP+PP
    tp_pp_kv_cache_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
        {
            "layer3": new_kv_cache_spec(),
        },
    ]

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        tp_pp_kv_cache_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    expected_12 = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                layers=["layer1", "layer2"],
                layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                block_stride=ref_kv_cache_spec.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
        ],
    )
    expected_3 = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=ref_kv_cache_spec.page_size_bytes * 10,
                layers=["layer3"],
                layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                block_stride=ref_kv_cache_spec.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer3"], ref_kv_cache_spec),
        ],
    )
    assert kv_cache_configs == [
        expected_12,
        expected_12,
        expected_3,
        expected_3,
    ]

    # Different workers have different types of layers. This is the case for
    # hybrid models + PP.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_kv_cache_spec(),
        },
        {
            "layer3": new_sliding_window_spec(),
            "layer4": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ref_kv_cache_spec.page_size_bytes * 2 * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                    layers=["layer1", "layer2"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1", "layer2"], ref_kv_cache_spec),
                KVCacheGroupSpec([], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10 * 2,
                    layers=["layer3", "layer4"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec([], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer3", "layer4"], new_sliding_window_spec()),
            ],
        ),
    ]

    # When divided into multiple KVCacheGroups, need to ensure the number of
    # layers per group is similar.
    different_type_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
            "layer2": new_sliding_window_spec(),
            "layer3": new_sliding_window_spec(),
        },
        {
            "layer4": new_kv_cache_spec(),
            "layer5": new_sliding_window_spec(),
            "layer6": new_sliding_window_spec(),
        },
    ]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        different_type_layer_specs,
        [
            ref_kv_cache_spec.page_size_bytes * 10,
            ref_kv_cache_spec.page_size_bytes * 10,
        ],
    )
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer1"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer2"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer3"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer1"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer2"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer3"], new_sliding_window_spec()),
            ],
        ),
        KVCacheConfig(
            num_blocks=10,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer4"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer5"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * 10,
                    layers=["layer6"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes * 10,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer4"], ref_kv_cache_spec),
                KVCacheGroupSpec(["layer5"], new_sliding_window_spec()),
                KVCacheGroupSpec(["layer6"], new_sliding_window_spec()),
            ],
        ),
    ]

    # Have conflicting layers. Need to raise an error.
    conflicting_layer_specs = [
        {
            "layer1": new_kv_cache_spec(),
        },
        {
            "layer1": new_sliding_window_spec(),
        },
    ]
    with pytest.raises(AssertionError):
        get_kv_cache_configs(
            vllm_config,
            conflicting_layer_specs,
            [
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
                ref_kv_cache_spec.page_size_bytes * 2 * 10,
            ],
        )


@pytest.mark.parametrize(
    "asymmetric_memory",
    [False, True],
    ids=["symmetric", "asymmetric"],
)
def test_get_kv_cache_configs_pp_sharding(asymmetric_memory):
    model_config = ModelConfig(max_model_len=512)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.prefix_cache_retention_interval = None

    ref_kv_cache_spec = new_kv_cache_spec()
    pp_kv_cache_specs = [
        {"layer1": ref_kv_cache_spec},
        {"layer2": ref_kv_cache_spec},
    ]

    expected_num_blocks = model_config.max_model_len // ref_kv_cache_spec.block_size + 1
    avail_memory = ref_kv_cache_spec.page_size_bytes * expected_num_blocks

    # With per-worker validation, each worker only needs memory for its own
    # layers. Worker 2 having more memory shouldn't affect worker 1's config.
    available_memory = (
        [avail_memory, avail_memory * 2] if asymmetric_memory else [avail_memory] * 2
    )

    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        pp_kv_cache_specs,
        available_memory,
    )

    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    layers=["layer1"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes
                    * expected_num_blocks,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer1"], ref_kv_cache_spec)],
        ),
        KVCacheConfig(
            num_blocks=expected_num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=ref_kv_cache_spec.page_size_bytes * expected_num_blocks,
                    layers=["layer2"],
                    layer_stride=ref_kv_cache_spec.page_size_bytes
                    * expected_num_blocks,
                    block_stride=ref_kv_cache_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer2"], ref_kv_cache_spec)],
        ),
    ]


def test_project_kv_cache_groups_to_worker():
    spec_a = new_kv_cache_spec()
    spec_b = new_kv_cache_spec(num_kv_heads=4)

    global_groups = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], spec_a),
    ]
    worker_spec = {"layer1": spec_a, "layer2": spec_a}
    projected = kv_cache_planning._project_kv_cache_groups_to_worker(
        global_groups, worker_spec
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer2"]
    assert projected[0].kv_cache_spec is spec_a

    projected = kv_cache_planning._project_kv_cache_groups_to_worker(
        global_groups, {"layer4": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == []
    assert projected[0].kv_cache_spec is spec_a

    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer1": spec_a, "layer2": spec_b, "layer3": spec_a},
    )
    global_groups_uniform = [
        KVCacheGroupSpec(["layer1", "layer2", "layer3"], uniform_spec),
    ]
    projected = kv_cache_planning._project_kv_cache_groups_to_worker(
        global_groups_uniform, {"layer1": spec_a, "layer3": spec_a}
    )
    assert len(projected) == 1
    assert projected[0].layer_names == ["layer1", "layer3"]
    proj_spec = projected[0].kv_cache_spec
    assert isinstance(proj_spec, UniformTypeKVCacheSpecs)
    assert set(proj_spec.kv_cache_specs.keys()) == {"layer1", "layer3"}


@pytest.mark.parametrize("with_draft", [False, True])
@pytest.mark.parametrize("indexer_alignment", [None, 512])
def test_sparse_mla_preserves_physical_row_addressing(with_draft, indexer_alignment):
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
    from vllm.v1.worker.utils import allocate_kv_cache

    config = VllmConfig(model_config=ModelConfig(max_model_len=1024))
    config.cache_config.kv_cache_layout = "BLHNC"
    config.parallel_config.decode_context_parallel_size = 4
    common = dict(block_size=64, num_kv_heads=1, dtype=torch.uint8)
    specs = {
        "target": MLAAttentionSpec(
            **common,
            head_size=576,
            state_content_bytes=656,
            block_stride_alignment=656,
            cache_dtype_str="fp8_ds_mla",
            is_index_group_leader=True,
        ),
        "indexer": MLAAttentionSpec(
            **common,
            head_size=132,
            cache_role=SparseCacheRole.INDEXER,
            block_stride_alignment=indexer_alignment,
        ),
        "draft": SlidingWindowSpec(
            block_size=64,
            num_kv_heads=64,
            head_size=64,
            dtype=torch.bfloat16,
            sliding_window=2048,
            dcp_sharded=False,
        ),
    }
    if not with_draft:
        del specs["draft"]
    groups = get_kv_cache_groups(config, specs)
    cache_config = kv_cache_planning.get_kv_cache_config_from_groups(
        config, groups, available_memory=8 * 1024 * 1024
    )
    caches = allocate_kv_cache(
        cache_config, torch.device("cpu"), KVCacheLayout.BLHNC, [64] * len(groups)
    )
    cache = caches["target"].squeeze(1)
    if indexer_alignment is not None:
        assert caches["indexer"].stride(0) % indexer_alignment == 0
    rows, stride_rows = flat_kv_row_view(cache, 64)
    cache[1, 0].fill_(7)
    torch.testing.assert_close(rows[stride_rows], cache[1, 0])
    assert torch.all(rows[stride_rows] == 7)


def test_is_kv_cache_spec_uniform():
    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_kv_cache_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_kv_cache_spec(num_kv_heads=32),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
    }
    assert is_kv_cache_spec_uniform(kv_cache_spec)

    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(num_kv_heads=32, sliding_window=1),
        "layer_2": new_sliding_window_spec(num_kv_heads=32, sliding_window=2),
    }
    assert not is_kv_cache_spec_uniform(kv_cache_spec)

    script = """
import sys

from vllm.v1.core.kv_cache_planning import is_kv_cache_spec_uniform
from vllm.v1.kv_cache_interface import KVCacheSpec

if sys.flags.optimize < 1:
    raise RuntimeError("subprocess is not running with optimization enabled")
specs = {
    "a": KVCacheSpec(block_size=1),
    "b": KVCacheSpec(block_size=2),
}
if is_kv_cache_spec_uniform(specs):
    raise RuntimeError("different specs were treated as uniform")
"""
    subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "want_estimated_max_len"),
    [
        ("Qwen/Qwen1.5-7B", 16385, 16384),
        ("Qwen/Qwen1.5-7B", 16383, 16383),
    ],
)
def test_estimate_max_model_len(model_id, max_model_len, want_estimated_max_len):
    # Create a VllmConfig
    model_config = ModelConfig(
        model_id,
        runner="generate",
        dtype="float16",
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=32768,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )

    # Create KV cache specs
    kv_cache_spec = {}
    for i in range(32):
        layer_name = f"layer_{i}"
        kv_cache_spec[layer_name] = FullAttentionSpec(
            block_size=16,
            num_kv_heads=32,
            head_size=128,
            dtype=torch.float16,
        )
    # Estimate the maximum model length, 16384 model_len need 8GB
    estimated_max_len = estimate_max_model_len(
        vllm_config, kv_cache_spec, 8 * GiB_bytes
    )
    assert estimated_max_len == want_estimated_max_len


def test_get_kv_cache_config_one_worker():
    # pass max_model_len to pass check_enough_kv_cache_memory
    model_config = ModelConfig(max_model_len=16)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.prefix_cache_retention_interval = None

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    # all layers are full attention -> single group
    kv_cache_specs_full = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    kv_cache_config_full = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    print(kv_cache_config_full)
    assert kv_cache_config_full == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )

    # all layers are sliding window -> single group
    kv_cache_specs_sliding = {
        "layer_1": new_sliding_window_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_sliding = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_sliding], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_sliding == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_sliding_window_spec())
        ],
    )

    # full + sliding, but disable_hybrid_kv_cache_manager
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"], new_kv_cache_spec(sliding_window=1)
            ),
        ],
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False

    # full + sliding, with hybrid_kv_cache_manager
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 64,
                layers=["layer_1"],
                layer_stride=mem_per_block_per_layer * 64,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 64,
                layers=["layer_2"],
                layer_stride=mem_per_block_per_layer * 64,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_2"], new_sliding_window_spec()),
        ],
    )

    # 2 full + 4 sliding, 2 layers per group
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_sliding_window_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_3", "layer_5"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 2,
                layers=["layer_4", "layer_6"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec()),
            KVCacheGroupSpec(["layer_3", "layer_5"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_4", "layer_6"], new_sliding_window_spec()),
        ],
    )

    # 3 full + 7 sliding, pad to 3 full + 9 sliding
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_sliding_window_spec(),
        "layer_5": new_sliding_window_spec(),
        "layer_6": new_sliding_window_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_1", "layer_2", "layer_3"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_4", "layer_7", "layer_10"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_5", "layer_8"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_6", "layer_9"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1", "layer_2", "layer_3"], new_kv_cache_spec()),
            KVCacheGroupSpec(
                ["layer_4", "layer_7", "layer_10"], new_sliding_window_spec()
            ),
            KVCacheGroupSpec(["layer_5", "layer_8"], new_sliding_window_spec()),
            KVCacheGroupSpec(["layer_6", "layer_9"], new_sliding_window_spec()),
        ],
    )

    # 6 full + 5 sliding, pad to 6 full + 6 sliding. This is a typical case for gpt-oss
    # eagle where there is only one more full attention layer than sliding window layers
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
        "layer_3": new_kv_cache_spec(),
        "layer_4": new_kv_cache_spec(),
        "layer_5": new_kv_cache_spec(),
        "layer_6": new_kv_cache_spec(),
        "layer_7": new_sliding_window_spec(),
        "layer_8": new_sliding_window_spec(),
        "layer_9": new_sliding_window_spec(),
        "layer_10": new_sliding_window_spec(),
        "layer_11": new_sliding_window_spec(),
    }

    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 6 * 32]
    )[0]
    print(kv_cache_config_hybrid)
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 6,
                layers=[
                    "layer_1",
                    "layer_2",
                    "layer_3",
                    "layer_4",
                    "layer_5",
                    "layer_6",
                ],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 6,
                layers=["layer_7", "layer_8", "layer_9", "layer_10", "layer_11"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "layer_6"],
                new_kv_cache_spec(),
            ),
            KVCacheGroupSpec(
                ["layer_7", "layer_8", "layer_9", "layer_10", "layer_11"],
                new_sliding_window_spec(),
            ),
        ],
    )

    # different hidden size but same type, use UniformTypeKVCacheSpecs
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=128),
        "layer_2": new_kv_cache_spec(head_size=64),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 3 * 32]
    )[0]
    # Layers of different page sizes pack densely into one allocation: the
    # 2x-larger layer_1 takes the first region, layer_2 the next.
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_1"],
                layer_stride=mem_per_block_per_layer * 2 * 32,
                block_stride=mem_per_block_per_layer * 2,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32 * 3,
                layers=["layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
                offset=mem_per_block_per_layer * 2 * 32,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1", "layer_2"],
                UniformTypeKVCacheSpecs(
                    block_size=16, kv_cache_specs=kv_cache_specs_hybrid
                ),
            )
        ],
    )

    # Different hidden size and different type, align by different block size
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": new_sliding_window_spec(head_size=32),
    }
    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 32]
    )[0]
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                layers=["layer_1"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
            KVCacheTensor(
                size=mem_per_block_per_layer * 32,
                layers=["layer_2"],
                layer_stride=mem_per_block_per_layer * 32,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer_1"], new_kv_cache_spec(head_size=64)),
            KVCacheGroupSpec(
                ["layer_2"], new_sliding_window_spec(head_size=32, block_size=32)
            ),
        ],
    )

    # different hidden size that cannot be aligned by using different block size,
    # but can be aligned by padding the smaller physical page.
    swa_spec = new_sliding_window_spec(head_size=96)
    kv_cache_specs_hybrid = {
        "layer_1": new_kv_cache_spec(head_size=64),
        "layer_2": swa_spec,
    }

    kv_cache_config_hybrid = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_hybrid], [mem_per_block_per_layer * 2 * 32]
    )[0]
    padded_page_size = swa_spec.page_size_bytes
    assert kv_cache_config_hybrid == KVCacheConfig(
        num_blocks=42,
        kv_cache_tensors=[
            KVCacheTensor(
                size=padded_page_size * 42,
                layers=["layer_1"],
                layer_stride=padded_page_size * 42,
                block_stride=padded_page_size,
            ),
            KVCacheTensor(
                size=padded_page_size * 42,
                layers=["layer_2"],
                layer_stride=padded_page_size * 42,
                block_stride=padded_page_size,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer_1"],
                new_kv_cache_spec(
                    head_size=64,
                    page_size_padded=padded_page_size,
                ),
            ),
            KVCacheGroupSpec(
                ["layer_2"],
                new_sliding_window_spec(head_size=96),
            ),
        ],
    )

    # Test num_gpu_blocks_override
    vllm_config.cache_config.num_gpu_blocks_override = 16
    kv_cache_config_override_blocks = get_kv_cache_configs(
        vllm_config, [kv_cache_specs_full], [mem_per_block_per_layer * 2 * 32]
    )[0]
    assert kv_cache_config_override_blocks == KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mem_per_block_per_layer * 16 * 2,
                layers=["layer_1", "layer_2"],
                layer_stride=mem_per_block_per_layer * 16,
                block_stride=mem_per_block_per_layer,
            ),
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer_1", "layer_2"], new_kv_cache_spec())],
    )


def test_get_kv_cache_configs_attention_free():
    kv_cache_specs: dict[str, KVCacheSpec] = {}
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    vllm_config.cache_config.prefix_cache_retention_interval = None
    kv_cache_configs = get_kv_cache_configs(vllm_config, [kv_cache_specs], [0])
    assert kv_cache_configs == [
        KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        )
    ]


def test_packed_groups_glm5_like_hybrid():
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=8192))
    vllm_config.cache_config.kv_cache_layout = "BLHNC"
    kpool, block_size = 4, 1024
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    for i in range(11):
        kv_cache_spec[f"layers.{i}.attn"] = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            block_stride_alignment=32 * 576 * 2,
        )
        kv_cache_spec[f"layers.{i}.indexer"] = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=132,
            dtype=torch.uint8,
            tokens_per_state=kpool,
            block_stride_alignment=64 * 132,
        )
        kv_cache_spec[f"layers.{i}.tail"] = CircularBufferSpec(
            block_size=kpool,
            num_kv_heads=2,
            head_size=128,
            head_size_v=0,
            dtype=torch.bfloat16,
        )
    for i in range(34):
        kv_cache_spec[f"layers.{i}.kda"] = MambaSpec(
            shapes=((3, 6144), (16, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            block_size=block_size,
        )

    groups = kv_cache_planning.get_kv_cache_groups(vllm_config, kv_cache_spec)
    attn = [
        g
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        and isinstance(g.kv_cache_spec.first_spec, MLAAttentionSpec)
    ]
    rings = [
        g
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        and isinstance(g.kv_cache_spec.first_spec, CircularBufferSpec)
    ]
    mamba = [
        g
        for g in groups
        if isinstance(
            getattr(g.kv_cache_spec, "first_spec", g.kv_cache_spec), MambaSpec
        )
    ]
    assert len(attn) == 1 and len(attn[0].layer_names) == 22
    assert len(rings) == 1 and len(rings[0].layer_names) == 11
    # 11 * (1.18 MB MLA + 34 KB indexer) per block fits 12 fp32-SSM KDA states.
    assert len(mamba) == 3
    assert sum(len(g.layer_names) for g in mamba) == 34

    kv_cache_config = kv_cache_planning.get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory=2 * 1024**3
    )
    strides = {t.block_stride for t in kv_cache_config.kv_cache_tensors}
    assert len(strides) == 1
    (block_stride,) = strides
    # Blocks sit a whole number of 32-row MLA pages and 64-state indexer pages
    # apart so both caches can be re-paged for their kernels.
    assert block_stride % (32 * 576 * 2) == 0
    assert block_stride % (64 * 132) == 0


def new_swa_mla_spec(head_size=576, sliding_window=128, model_version=None):
    return SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=head_size,
        dtype=torch.float32,
        sliding_window=sliding_window,
        model_version=model_version,
    )


def new_indexer_mla_spec(block_size=16):
    # Sparse-attention indexer k_cache: an MLAAttentionSpec with a much smaller
    # page size than the main MLA attention (uint8, small head), so their pages
    # cannot be unified.
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
    )


def test_mixed_page_size_groups_use_spec_compatibility():
    specs = {}
    for i in range(3):
        specs[f"mla.{i}"] = new_mla_spec()
        specs[f"indexer.{i}"] = new_indexer_mla_spec()
    specs.update({f"swa.{i}": new_swa_mla_spec(head_size=1024) for i in range(5)})

    config = _grouping_config()
    config.cache_config = CacheConfig()
    config.cache_config.kv_cache_layout = "BLNHC"
    groups = get_kv_cache_groups(config, specs)

    assert len(groups) == 3
    assert {name for group in groups for name in group.layer_names} == set(specs)
    assert sorted(len(group.layer_names) for group in groups) == [2, 3, 6]


def _grouping_config():
    cache_config = CacheConfig()
    cache_config.kv_cache_layout = "LBNHC"
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        cache_config=cache_config,
    )


def test_hidden_state_group_preserves_hybrid_prefix_cache_granularity():
    block_size = 544
    full_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        # FullAttentionSpec stores both K and V, so this produces a
        # 544 * 1 * (512 + 512) * 2 = 1,114,112-byte page.
        head_size=512,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        block_size=block_size,
        shapes=((557056,),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
    )
    hidden_spec = HiddenStateCacheSpec(
        block_size=block_size,
        num_kv_heads=3,
        head_size=1024,
        dtype=torch.bfloat16,
    )
    assert full_spec.page_size_bytes == mamba_spec.page_size_bytes

    groups = get_kv_cache_groups(
        _grouping_config(),
        {
            "model.full_attn": full_spec,
            "model.mamba": mamba_spec,
            "cache_only_layers.0": hidden_spec,
        },
    )

    hidden_group = next(
        group
        for group in groups
        if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
    )
    assert hidden_group.kv_cache_spec.block_size == 136

    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=True,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=object(),
    )
    assert kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    ) == (544, 136)


def test_multi_run_layer_compact_strides_place_hoisted_heads():
    """A layer-compact run region is its own dense allocation: under LHBNC the head
    groups sit between the layers and the blocks, so a run's block stride is one head
    group's slice, not the whole page (regression test for setStorage out-of-bounds
    on ROCm hybrid models)."""
    full = new_kv_cache_spec()
    swa = new_sliding_window_spec(sliding_window=full.block_size * 2)
    page = full.page_size_bytes
    assert swa.page_size_bytes == page

    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    vllm_config.cache_config.kv_cache_layout = "LHBNC"
    config = kv_cache_planning.get_kv_cache_config_from_groups(
        vllm_config,
        [
            KVCacheGroupSpec(["full.0", "full.1"], full),
            KVCacheGroupSpec(["swa.0", "swa.1"], swa),
        ],
        available_memory=4 * page * 8,
    )

    num_blocks = config.num_blocks
    head_group = full.block_size * full.state_content_size_bytes
    for tensor in config.kv_cache_tensors:
        # One head group per block, head groups between the layers and blocks.
        assert tensor.block_stride == head_group
        assert tensor.layer_stride == full.num_heads * num_blocks * head_group
        # The last block of the last layer's last head group must stay within the
        # allocation (the old page-based stride overran it).
        end = (
            tensor.offset
            + tensor.layer_stride * (len(tensor.layers) - 1)
            + (full.num_heads - 1) * num_blocks * head_group
            + tensor.block_stride * (num_blocks - 1)
            + head_group
        )
        assert end <= tensor.size


def test_mla_draft_prefers_standard_layout_when_pages_can_be_unified():
    specs = {
        "target.0.attn": new_mla_spec(),
        "draft.0": new_sliding_window_spec(num_kv_heads=1, head_size=288),
    }
    assert len({spec.page_size_bytes for spec in specs.values()}) == 1

    groups = get_kv_cache_groups(_grouping_config(), specs)

    assert len(groups) == 2
    assert all(
        not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in groups
    )


def test_mla_with_incompatible_swa_uses_one_full_allocation_group(caplog_vllm):
    # Sparse MLA pages cannot be padded safely. Keeping the draft's attention
    # compute sliding-window while promoting only its allocation semantics lets
    # every layer share the target's block table and remain contiguous.
    draft = new_sliding_window_spec(block_size=16)
    specs = {
        "target.0.attn": new_mla_spec(block_size=64),
        "target.0.indexer": new_indexer_mla_spec(block_size=64),
        "draft.0": draft,
    }

    groups = get_kv_cache_groups(_grouping_config(), specs)
    assert len(groups) == 1
    assert set(groups[0].layer_names) == set(specs)
    group_spec = groups[0].kv_cache_spec
    assert isinstance(group_spec, UniformTypeKVCacheSpecs)
    assert group_spec.block_size == 64
    promoted_draft = group_spec.kv_cache_specs["draft.0"]
    assert isinstance(promoted_draft, FullAttentionSpec)
    assert not isinstance(promoted_draft, SlidingWindowSpec)
    assert promoted_draft.block_size == 64
    assert promoted_draft.sliding_window == draft.sliding_window
    assert specs["draft.0"] is draft
    assert "attention compute is unchanged" in caplog_vllm.text


def test_hidden_states_with_tp_scales_page_size():
    """When TP shrinks KV pages below the hidden-state per-token cost,
    get_kv_cache_groups must scale up target block sizes so that the
    common page accommodates the unsharded hidden states."""
    # Simulate TP=4 sharding a model with 8 KV heads → 2 per rank.
    # KV page = block_size(16) * num_kv_heads(2) * head_size(64) * dtype(2)
    #         = 16 * 2 * 64 * 2 = 4096 bytes.
    kv_spec = new_kv_cache_spec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
    )
    # Hidden-state per-token cost = num_hidden_states(6) * hidden_size(512)
    #   * dtype(2) = 6144 bytes, which exceeds the 4096-byte KV page.
    hs_spec = HiddenStateCacheSpec(
        block_size=16,
        num_kv_heads=6,
        head_size=512,
        dtype=torch.bfloat16,
    )
    specs = {
        "target.0.attn": kv_spec,
        "target.1.attn": kv_spec,
        "cache_only_layers.48": hs_spec,
    }

    groups = get_kv_cache_groups(_grouping_config(), specs)

    # The hidden-state layer should be present and no assertion should fire.
    all_layers = {name for g in groups for name in g.layer_names}
    assert "cache_only_layers.48" in all_layers

    # The target group block sizes must have been scaled up.
    for g in groups:
        if "cache_only_layers.48" not in g.layer_names:
            assert g.kv_cache_spec.block_size > kv_spec.block_size


def test_auto_fit_max_model_len():
    """Test that max_model_len=-1 auto-fits to available GPU memory."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=1024)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # With enough memory, max_model_len stays at the derived max
    large_available_memory = mem_per_block_per_layer * 2 * 1024  # plenty of memory
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [large_available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024

    # Reset for next test
    model_config = ModelConfig(max_model_len=1024)
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    # With limited memory, max_model_len should be reduced
    # Need memory for at least max_model_len tokens
    # 32 blocks worth of memory for 2 layers = can fit 32*16=512 tokens
    limited_memory = mem_per_block_per_layer * 2 * 32
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [limited_memory]
    )
    # Should be reduced to fit in memory
    assert vllm_config.model_config.max_model_len < 1024
    assert vllm_config.model_config.max_model_len > 0


def test_auto_fit_max_model_len_with_hybrid():
    """Test that auto-fit works with hybrid KV cache specs."""
    # Create config with original_max_model_len=-1 to trigger auto-fit
    model_config = ModelConfig(max_model_len=8192)
    # Simulate the user passing -1 by setting original_max_model_len
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2  # 16KB per block per layer
    gamma = 2
    kv_cache_specs = {
        "layer_1": new_mamba_spec(num_speculative_blocks=gamma),
        "layer_2": new_kv_cache_spec(),
    }

    # One extra block on top of what a 1024-token request needs: the pool
    # reserves one block as the null block.
    available_memory = mem_per_block_per_layer * (1024 // 16 + 1 + gamma + 1)
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [available_memory]
    )
    assert vllm_config.model_config.max_model_len == 1024


def test_auto_fit_max_model_len_not_triggered():
    """Test that auto-fit is not triggered when original_max_model_len is not -1."""
    model_config = ModelConfig(max_model_len=16)
    # original_max_model_len should be None by default, not -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }

    # This should work normally without auto-fit
    _kv_cache_configs = get_kv_cache_configs(
        vllm_config, [kv_cache_specs], [mem_per_block_per_layer * 2 * 32]
    )
    assert vllm_config.model_config.max_model_len == 16


def test_auto_fit_max_model_len_respects_num_gpu_blocks_override():
    """Auto-fit must size max_model_len against the override-clamped pool, not
    the raw `available_memory`. Without this, auto-fit could pick a
    max_model_len that no longer fits once `num_gpu_blocks_override` is applied.
    """
    model_config = ModelConfig(max_model_len=16384)
    model_config.original_max_model_len = -1  # request auto-fit
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    # Cap the cache to 32 blocks regardless of available memory.
    vllm_config.cache_config.num_gpu_blocks_override = 32

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),  # block_size=16
        "layer_2": new_kv_cache_spec(),
    }
    # Plenty of raw memory (1024 blocks per layer would fit max_model_len=16384).
    large_available_memory = mem_per_block_per_layer * 2 * 1024

    get_kv_cache_configs(vllm_config, [kv_cache_specs], [large_available_memory])

    # 32 blocks * block_size 16 = 512 token slots, so max_model_len must
    # auto-fit at or below that.
    assert 0 < vllm_config.model_config.max_model_len <= 32 * 16


def test_check_enough_kv_cache_memory_respects_num_gpu_blocks_override():
    """Admission check must use the override-clamped pool size, not raw
    `available_memory`. Without this, startup could accept a max_model_len
    that does not actually fit in `num_gpu_blocks_override` blocks.
    """
    model_config = ModelConfig(max_model_len=16384)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    # 32 blocks is far too small for max_model_len=16384 (would need 1024).
    vllm_config.cache_config.num_gpu_blocks_override = 32

    mem_per_block_per_layer = 16 * 2 * 64 * 4 * 2
    kv_cache_specs = {
        "layer_1": new_kv_cache_spec(),
        "layer_2": new_kv_cache_spec(),
    }
    # Plenty of raw memory: a bytes-only check against this would pass.
    large_available_memory = mem_per_block_per_layer * 2 * 1024

    with pytest.raises(ValueError, match="max seq len"):
        get_kv_cache_configs(vllm_config, [kv_cache_specs], [large_available_memory])


def test_unify_kv_cache_page_size_uses_padding_for_non_divisible_sizes():
    """DFlash drafters can have a smaller head size than the target model.

    For example, MiMo uses 192-dim target KV heads while its DFlash draft uses
    128-dim KV heads. The resulting page sizes are 3:2 rather than an integer
    block-size multiple, so the smaller page must be padded instead.
    """
    # Attention layers read padded pages through a strided view, so padding is
    # allowed.
    target_spec = new_kv_cache_spec(
        block_size=16,
        num_kv_heads=1,
        head_size=192,
        dtype=torch.bfloat16,
    )
    draft_spec = new_sliding_window_spec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=1024,
    )

    unified_specs = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "target_attn": target_spec,
            "draft_attn": draft_spec,
        }
    )

    assert unified_specs["target_attn"] == target_spec
    unified_draft_spec = unified_specs["draft_attn"]
    assert unified_draft_spec.block_size == draft_spec.block_size
    assert unified_draft_spec.real_page_size_bytes == draft_spec.real_page_size_bytes
    assert unified_draft_spec.page_size_padded == target_spec.page_size_bytes
    assert unified_draft_spec.page_size_bytes == target_spec.page_size_bytes


def test_unify_hybrid_kv_cache_specs():
    # 1. has_full_attention and has_sliding_window
    before_spec_1 = new_kv_cache_spec(block_size=64)
    before_spec_2 = new_sliding_window_spec(
        block_size=16, page_size_padded=32 * 1024, sliding_window=1024
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_spec["draft_layer_1"] = replace(before_spec_1, dcp_sharded=False)
    kv_cache_spec["draft_layer_2"] = replace(before_spec_2, dcp_sharded=False)
    kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec(block_size=64)
    expected_spec_2 = new_kv_cache_spec(
        block_size=64, page_size_padded=64 * 1024, sliding_window=1024
    )
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2
    assert kv_cache_spec["layer_2"].page_size_bytes == 64 * 1024
    assert kv_cache_spec["draft_layer_2"] == replace(expected_spec_2, dcp_sharded=False)

    # 2. has_full_attention and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
    }
    kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )

    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2

    # 3. has_full_attention, has_sliding_window and has_chunked_local_attention
    before_spec_1 = new_kv_cache_spec()
    before_spec_2 = new_sliding_window_spec(
        page_size_padded=32 * 1024, sliding_window=1024
    )
    before_spec_3 = new_chunked_local_attention_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    kv_cache_spec = {
        "layer_1": before_spec_1,
        "layer_2": before_spec_2,
        "layer_3": before_spec_3,
    }
    kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)
    expected_spec_1 = new_kv_cache_spec()
    expected_spec_2 = new_kv_cache_spec(page_size_padded=32 * 1024, sliding_window=1024)
    expected_spec_3 = new_kv_cache_spec(
        page_size_padded=32 * 1024, attention_chunk_size=512
    )
    assert kv_cache_spec["layer_1"] == expected_spec_1
    assert kv_cache_spec["layer_2"] == expected_spec_2
    assert kv_cache_spec["layer_3"] == expected_spec_3

    # 4. No FullAttentionSpec, should not convert
    kv_cache_spec = {
        "layer_1": new_sliding_window_spec(sliding_window=1024),
        "layer_2": new_chunked_local_attention_spec(attention_chunk_size=512),
    }

    with pytest.raises(ValueError):
        kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)

    # Replicated Mamba state still requires the hybrid cache manager.
    kv_cache_spec = {"attention": new_kv_cache_spec(), "mamba": new_mamba_spec()}
    with pytest.raises(ValueError):
        kv_cache_planning.unify_hybrid_kv_cache_specs(kv_cache_spec)


def test_unify_kv_cache_spec_page_size_mamba():
    """Regression test for https://github.com/vllm-project/vllm/issues/43626.

    MambaSpec's page_size_bytes is determined by its state shapes and does not
    change with block_size, so unify_kv_cache_spec_page_size must pad the Mamba
    page instead of scaling its block_size. This situation arises when a layer
    with a page larger than the (already platform-aligned) Mamba page joins the
    specs, e.g. a dense draft model with more KV heads than the hybrid main
    model.
    """
    # 1. Hybrid main model (Mamba + full attention, pages already aligned at
    # 16KB) plus a dense draft model layer with a 2x larger page (32KB).
    # Reproduces the bare AssertionError from #43626: 32768 % 16384 == 0, so
    # the old code scaled the Mamba block_size, which left page_size_bytes
    # unchanged at 16384.
    mamba_spec = new_mamba_spec()  # page_size_bytes = 16384
    main_attn_spec = new_kv_cache_spec()  # page_size_bytes = 16384
    draft_attn_spec = new_kv_cache_spec(num_kv_heads=4)  # page_size_bytes = 32768
    assert mamba_spec.page_size_bytes == main_attn_spec.page_size_bytes == 16384
    assert draft_attn_spec.page_size_bytes == 32768

    unified = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "mamba_layer": mamba_spec,
            "main_attn_layer": main_attn_spec,
            "draft_attn_layer": draft_attn_spec,
        }
    )
    # Mamba page is padded; block_size (caching granularity) is unchanged.
    assert unified["mamba_layer"].page_size_bytes == 32768
    assert unified["mamba_layer"].page_size_padded == 32768
    assert unified["mamba_layer"].block_size == mamba_spec.block_size
    # Attention layer with smaller page still unifies by scaling block_size.
    assert unified["main_attn_layer"].page_size_bytes == 32768
    assert unified["main_attn_layer"].block_size == 2 * main_attn_spec.block_size
    assert unified["main_attn_layer"].page_size_padded is None
    # Layer already at max page size is unchanged.
    assert unified["draft_attn_layer"] == draft_attn_spec

    # 2. Mamba page already padded by the platform (state smaller than the
    # padded page); the padding is re-applied at the new maximum.
    padded_mamba_spec = new_mamba_spec(
        shapes=((2, 256), (3, 32, 32)), page_size_padded=16384
    )
    assert padded_mamba_spec.page_size_bytes == 16384
    unified = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "mamba_layer": padded_mamba_spec,
            "draft_attn_layer": draft_attn_spec,
        }
    )
    assert unified["mamba_layer"].page_size_bytes == 32768
    assert unified["mamba_layer"].page_size_padded == 32768

    # 3. Mamba page that does not evenly divide the maximum page size is
    # padded as well (the divisibility constraint only applies to block_size
    # scaling).
    odd_mamba_spec = new_mamba_spec(shapes=((6144,),))
    assert odd_mamba_spec.page_size_bytes == 24576
    assert 32768 % odd_mamba_spec.page_size_bytes != 0
    unified = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "mamba_layer": odd_mamba_spec,
            "draft_attn_layer": draft_attn_spec,
        }
    )
    assert unified["mamba_layer"].page_size_bytes == 32768

    # 4. Attention layers with non-divisible page sizes are padded too: every
    # backend reads a padded page through the view's block stride, so there is
    # no longer a case that must raise.
    unified = kv_cache_planning.unify_kv_cache_spec_page_size(
        {
            "attn_layer": new_kv_cache_spec(block_size=24),  # 24576
            "draft_attn_layer": draft_attn_spec,  # 32768
        }
    )
    assert unified["attn_layer"].page_size_padded == 32768
    assert unified["attn_layer"].page_size_bytes == 32768
    assert unified["attn_layer"].block_size == 24

    # 5. Uniform page sizes are returned unchanged.
    specs = {
        "mamba_layer": new_mamba_spec(),
        "attn_layer": new_kv_cache_spec(),
    }
    assert kv_cache_planning.unify_kv_cache_spec_page_size(specs) == specs


@pytest.mark.parametrize("use_override", [True, False])
def test_kv_cache_reserves_null_block_for_max_model_len(use_override):
    """A KV cache sized to exactly the blocks a max_model_len request needs must
    be rejected. BlockPool keeps one block as the null block, so only
    num_blocks - 1 are usable; accepting num_blocks == needed would leave the
    request unschedulable and hang the engine. Covers both the
    num_gpu_blocks_override path and the memory-derived block count.
    """
    block_size = 16
    max_model_len = 512  # needs 512 / 16 = 32 blocks
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=max_model_len))
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    spec = new_kv_cache_spec(block_size=block_size)

    # 32 blocks -> only 31 usable after the null block: one short -> reject.
    if use_override:
        # Ample raw memory; the override alone constrains the block count.
        vllm_config.cache_config.num_gpu_blocks_override = 32
        available_memory = [spec.page_size_bytes * 1024]
    else:
        available_memory = [spec.page_size_bytes * 32]

    with pytest.raises(ValueError, match="max seq len"):
        get_kv_cache_configs(vllm_config, [{"layer1": spec}], available_memory)


def test_auto_fit_max_model_len_reserves_null_block():
    """Auto-fit (max_model_len=-1) must size max_model_len against usable
    blocks, not the total pool. With memory for exactly 64 blocks, one is the
    null block, so auto-fit must settle on 63 * block_size; picking the full
    pool would leave a max-length request unschedulable and hang the engine.
    """
    block_size = 16
    model_config = ModelConfig(max_model_len=1024)
    model_config.original_max_model_len = -1
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    spec = new_kv_cache_spec(block_size=block_size)

    # Exactly the 1024 / 16 = 64 blocks a full-length request would need.
    available_memory = [spec.page_size_bytes * 64]

    get_kv_cache_configs(vllm_config, [{"layer1": spec}], available_memory)

    assert vllm_config.model_config.max_model_len == 63 * block_size


def test_check_enough_kv_cache_memory_reserves_null_block():
    """The public admission check must reject a KV cache sized to exactly the
    blocks a max_model_len request needs. BlockPool keeps one block as the
    null block, so only num_blocks - 1 are usable; accepting
    num_blocks == needed would leave the request unschedulable and hang the
    engine.
    """
    block_size = 16
    max_model_len = 512  # needs 512 / 16 = 32 blocks
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=max_model_len))
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    spec = new_kv_cache_spec(block_size=block_size)

    # 32 blocks -> only 31 usable after the null block: one short -> reject.
    with pytest.raises(ValueError, match="max seq len"):
        check_enough_kv_cache_memory(
            vllm_config, {"layer1": spec}, spec.page_size_bytes * 32
        )

    # 33 blocks -> 32 usable after the null block -> accept.
    check_enough_kv_cache_memory(
        vllm_config, {"layer1": spec}, spec.page_size_bytes * 33
    )


def _spec_decode_grouping_config(method="dspark", model_type=None):
    """Grouping config with an EAGLE-family speculative method enabled."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_block_outermost=True
            )
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        speculative_config=SimpleNamespace(
            method=method,
            use_eagle=lambda: True,
            use_eagle_block_drop=lambda: True,
        ),
    )


def _hybrid_specs_with_draft(draft: bool, draft_shares_target_spec: bool = False):
    """A K3-shaped hybrid: MLA full attention + Mamba, optionally plus a
    DSpark-style draft MLA layer marked non_causal_multi_token_decode.

    The target's fp8 KV dtype is one difference from the draft on Kimi-K3
    (target `--kv-cache-dtype fp8_e4m3`, draft `auto`). Pass
    draft_shares_target_spec to match that dtype while retaining the draft
    marker as a group boundary.
    """
    target_dtype = None if draft_shares_target_spec else "fp8_e4m3"
    specs = {
        "target.attn.0": new_mla_spec(block_size=64, cache_dtype_str=target_dtype),
        "target.attn.1": new_mla_spec(block_size=64, cache_dtype_str=target_dtype),
        "target.mamba.0": new_mamba_spec(block_size=64, mamba_cache_mode="align"),
        "target.mamba.1": new_mamba_spec(block_size=64, mamba_cache_mode="align"),
    }
    if draft:
        draft_spec = new_mla_spec(block_size=64)
        specs["draft.attn.0"] = replace(draft_spec, non_causal_multi_token_decode=True)
    return specs


def test_draft_group_annotated_on_hybrid_general_path():
    # A drafter's MLA layer carries non_causal_multi_token_decode, so its group
    # is identifiable without keying off a model version.
    groups = get_kv_cache_groups(
        _spec_decode_grouping_config(), _hybrid_specs_with_draft(draft=True)
    )

    flagged = [g for g in groups if g.is_eagle_group]
    assert len(flagged) == 1
    assert "draft.attn.0" in flagged[0].layer_names


def test_mamba_groups_never_flagged_even_when_draft_shares_a_group():
    # Packed uniform-type groups can contain distinct target and draft layer
    # specs; the combined group still holds volatile draft KV and must be
    # flagged. What must never happen is a Mamba group being flagged: that
    # widens its lookup window to two consecutive chunks, which align-mode
    # checkpointing never produces, zeroing every lookup.
    groups = get_kv_cache_groups(
        _spec_decode_grouping_config(),
        _hybrid_specs_with_draft(draft=True, draft_shares_target_spec=True),
    )

    for group in groups:
        if "draft.attn.0" in group.layer_names:
            assert group.is_eagle_group
        if isinstance(group.kv_cache_spec, MambaSpec):
            assert not group.is_eagle_group


def test_draft_group_not_annotated_without_spec_decode():
    # The marker alone must not flag anything; the eagle semantics only apply
    # when a speculative method is actually enabled.
    config = _spec_decode_grouping_config()
    config.speculative_config = None
    groups = get_kv_cache_groups(config, _hybrid_specs_with_draft(draft=True))

    assert not any(g.is_eagle_group for g in groups)


def test_unidentifiable_draft_with_mamba_warns(caplog_vllm):
    # No group carries the draft marker, so consumers fall back to
    # conservative behavior that silently breaks reuse for Mamba groups.
    # That must at least be visible.
    groups = get_kv_cache_groups(
        _spec_decode_grouping_config(), _hybrid_specs_with_draft(draft=False)
    )

    assert not any(g.is_eagle_group for g in groups)
    assert "no KV cache group could be identified as the draft model's" in (
        caplog_vllm.text
    )


def test_unidentifiable_draft_without_mamba_does_not_warn(caplog_vllm):
    # Pure-attention models degrade gracefully under the consumers'
    # conservative fallback (a one-block hit drop at most), so the warning
    # stays silent to avoid noise on every unannotated EAGLE deployment.
    specs = {
        "target.attn.0": new_mla_spec(block_size=64),
        "target.attn.1": new_mla_spec(block_size=64),
    }
    groups = get_kv_cache_groups(_spec_decode_grouping_config(), specs)

    assert not any(g.is_eagle_group for g in groups)
    assert "could be identified as the draft model's" not in caplog_vllm.text


def test_no_warning_when_draft_group_is_identified(caplog_vllm):
    get_kv_cache_groups(
        _spec_decode_grouping_config(), _hybrid_specs_with_draft(draft=True)
    )

    assert "could be identified as the draft model's" not in caplog_vllm.text


def _deepseek_v4_specs(model_version="deepseek_v4"):
    """DeepseekV4-shaped specs: full MLA layers plus sliding-window MLA layers
    at differing page sizes, with the MTP draft layer registered last."""
    return {
        "model.layers.0.self_attn.attn": new_mla_spec(),
        "model.layers.1.self_attn.attn": new_mla_spec(),
        "model.layers.2.self_attn.attn": new_swa_mla_spec(
            head_size=1024, model_version=model_version
        ),
        # The MTP block registers last, and its sliding-window size differs, so
        # it lands in a group of its own.
        "model.layers.3.self_attn.attn": new_swa_mla_spec(
            head_size=1024, sliding_window=256, model_version=model_version
        ),
    }


@pytest.mark.parametrize(
    ("method", "model_type"),
    [("mtp", "deepseek_v4"), ("dspark", "deepseek_v4"), ("dspark", "deepseek_v41")],
)
def test_deepseek_v4_draft_group_annotated_on_packed_path(method, model_type):
    # DeepseekV4's MTP block reuses the target's decoder layer, so its spec
    # carries no draft marker and only the positional rule can find it. This
    # pins the pre-existing behaviour that the unified annotator must preserve.
    groups = get_kv_cache_groups(
        _spec_decode_grouping_config(method=method, model_type=model_type),
        _deepseek_v4_specs(model_version=None),
    )

    flagged = [g for g in groups if g.is_eagle_group]
    assert len(flagged) == 1
    assert "model.layers.3.self_attn.attn" in flagged[0].layer_names


def test_trailing_layer_fallback_applies_to_any_mtp_model():
    # The positional rule is sound for every MTP drafter, not just DeepseekV4:
    # MTP blocks reuse the target's decoder layer (no spec marker) and always
    # register after every target layer. The model_type must not gate it.
    groups = get_kv_cache_groups(
        _spec_decode_grouping_config(method="mtp", model_type="other"),
        _deepseek_v4_specs(),
    )

    flagged = [g for g in groups if g.is_eagle_group]
    assert len(flagged) == 1
    assert "model.layers.3.self_attn.attn" in flagged[0].layer_names


def _qwen3_5_hybrid_specs(with_mtp_layer: bool):
    """Qwen3.5-shaped hybrid: repeating [GDN x3, full-attn x1] blocks, with
    the MTP drafter's full-attn layer (spec-identical to the target's)
    registered last."""
    specs = {}
    idx = 0
    for _ in range(2):
        for _ in range(3):
            specs[f"model.layers.{idx}.linear_attn"] = new_mamba_spec(
                mamba_cache_mode="align"
            )
            idx += 1
        specs[f"model.layers.{idx}.self_attn.attn"] = new_kv_cache_spec()
        idx += 1
    if with_mtp_layer:
        specs["mtp.layers.0.self_attn.attn"] = new_kv_cache_spec()
    return specs


def test_qwen3_5_mtp_draft_group_annotated_on_hybrid_path(caplog_vllm):
    # A hybrid mamba + full-attention model with an MTP drafter that is
    # spec-indistinguishable from the target reaches the general multi-group
    # path. The trailing-layer rule must locate the draft group there so the
    # Mamba groups are not swept up by the flag-all consumer fallback.
    groups = get_kv_cache_groups(
        _spec_decode_grouping_config(method="mtp", model_type="qwen3_5"),
        _qwen3_5_hybrid_specs(with_mtp_layer=True),
    )

    flagged = [g for g in groups if g.is_eagle_group]
    assert len(flagged) == 1
    assert "mtp.layers.0.self_attn.attn" in flagged[0].layer_names
    for group in groups:
        if any(
            isinstance(spec, MambaSpec)
            for spec in iter_layer_specs(group.kv_cache_spec)
        ):
            assert not group.is_eagle_group
    assert "could be identified as the draft model's" not in caplog_vllm.text


@pytest.mark.parametrize("method", ["eagle", "eagle3", "dspark"])
def test_non_mtp_eagle_hybrid_still_warns(caplog_vllm, method):
    # Other drafters are not covered by the trailing-layer rule, so an
    # unidentifiable hybrid draft must still warn.
    groups = get_kv_cache_groups(
        _spec_decode_grouping_config(method=method, model_type="qwen3_5"),
        _qwen3_5_hybrid_specs(with_mtp_layer=True),
    )

    assert not any(g.is_eagle_group for g in groups)
    assert "could be identified as the draft model's" in caplog_vllm.text


def test_trailing_layer_fallback_requires_exact_partition():
    # If the groups do not partition the layers exactly (e.g. a caller that
    # dropped or duplicated layers), the positional rule is meaningless and
    # must not fire.
    from vllm.v1.core.kv_cache_planning import _annotate_eagle_groups

    specs = _qwen3_5_hybrid_specs(with_mtp_layer=True)
    config = _spec_decode_grouping_config(method="mtp", model_type="qwen3_5")
    groups = get_kv_cache_groups(config, specs)
    for g in groups:
        g.is_eagle_group = False
    # Remove one layer from its group: no longer an exact partition.
    trimmed = [
        KVCacheGroupSpec(
            [n for n in g.layer_names if n != "model.layers.0.linear_attn"],
            g.kv_cache_spec,
        )
        for g in groups
    ]
    _annotate_eagle_groups(config, specs, trimmed, use_trailing_layer_fallback=True)

    assert not any(g.is_eagle_group for g in trimmed)


_GQA_SPEC = FullAttentionSpec(
    block_size=16, num_kv_heads=8, head_size=64, dtype=torch.float32
)


_GQA_SWA_SPEC = SlidingWindowSpec(
    block_size=16, num_kv_heads=8, head_size=64, dtype=torch.float32, sliding_window=128
)


@pytest.mark.parametrize(
    "specs,expected",
    [
        pytest.param({"l.0": new_mla_spec(), "l.1": new_mla_spec()}, 4, id="mla"),
        pytest.param(
            {
                "l.0": new_mla_spec(),
                "l.1": new_mla_spec(),
                "s.0": new_swa_mla_spec(),
                "s.1": new_swa_mla_spec(),
            },
            4,
            id="mla-and-swa-mla",
        ),
        pytest.param({"l.0": _GQA_SPEC, "l.1": _GQA_SPEC}, 1, id="gqa"),
        pytest.param(
            {"l.0": new_mla_spec(), "l.1": _GQA_SPEC},
            1,
            id="mla-and-gqa-uniform-group",
        ),
        pytest.param(
            {
                "l.0": new_mla_spec(),
                "l.1": new_mla_spec(),
                "s.0": _GQA_SWA_SPEC,
                "s.1": _GQA_SWA_SPEC,
            },
            1,
            id="mla-and-gqa-swa",
        ),
    ],
)
def test_kv_tp_replicas(monkeypatch, specs, expected):
    """Resolved per layer before scheduler flattening hides mixed groups."""
    from vllm.config import ParallelConfig
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_count", lambda: 4)
    parallel_config = ParallelConfig(tensor_parallel_size=4)
    vllm_config = VllmConfig(
        model_config=ModelConfig(max_model_len=16), parallel_config=parallel_config
    )
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    mem = sum(s.page_size_bytes for s in specs.values()) * 10
    configs = get_kv_cache_configs(vllm_config, [specs], [mem])
    assert configs[0].kv_tp_replicas == expected
    scheduler = generate_scheduler_kv_cache_config(configs)
    assert scheduler.kv_tp_replicas == expected


@pytest.mark.parametrize(
    "specs,tp_size,dcp_size,expected",
    [
        pytest.param([new_mla_spec()], 8, 1, 8, id="mla"),
        pytest.param([new_mla_spec()], 8, 2, 1, id="mla-dcp"),
        pytest.param(
            [replace(new_mla_spec(), max_tp_shards=None)], 8, 1, 1, id="mla-unset"
        ),
        pytest.param([replace(_GQA_SPEC, max_tp_shards=2)], 8, 1, 4, id="gqa-partial"),
        pytest.param([replace(_GQA_SPEC, max_tp_shards=8)], 4, 1, 1, id="gqa-sharded"),
        pytest.param(
            [new_mla_spec(), replace(_GQA_SWA_SPEC, max_tp_shards=4)],
            8,
            1,
            2,
            id="mla-and-gqa-partial",
        ),
        pytest.param(
            [
                HiddenStateCacheSpec(
                    block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float32
                )
            ],
            8,
            1,
            1,
            id="hidden-state",
        ),
    ],
)
def test_kv_cache_groups_tp_replicas(specs, tp_size, dcp_size, expected):
    """Replicas are the gcd of each layer's tp_size // max_tp_shards."""
    groups = [KVCacheGroupSpec([f"l.{i}"], spec) for i, spec in enumerate(specs)]
    assert kv_cache_groups_tp_replicas(groups, tp_size, dcp_size) == expected
