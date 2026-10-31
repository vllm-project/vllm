# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for translating vLLM cache metadata to native offloading config."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, PreTrainedConfig

from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.config.quantization import QuantizationConfigArgs
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    SchedulerOffloadConfig,
)
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_planning import kv_cache_groups_tp_replicas
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_model_config_hash,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def _make_vllm_config(
    *,
    extra_config: dict[str, Any] | None = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
) -> VllmConfig:
    config = MagicMock()
    config.cache_config.block_size = 16
    config.cache_config.enable_prefix_caching = True
    config.cache_config.prefix_match_unit = None
    config.cache_config.cache_dtype = torch.float16
    config.cache_config.prefix_cache_retention_interval = None
    config.model_config.model = "test-model"
    config.model_config.max_model_len = 4096
    config.model_config.use_mla = False
    # _full_attention_spec's heads at tp=1: the parallelism-agnostic gate
    # requires the head shard to cover the model's KV heads exactly
    config.model_config.get_total_num_kv_heads.return_value = 4
    world_size = (
        tensor_parallel_size * pipeline_parallel_size * prefill_context_parallel_size
    )
    with patch.object(current_platform, "device_count", return_value=world_size):
        config.parallel_config = ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
        )
    config.kv_events_config = None
    config.ec_transfer_config = None
    config.use_v2_model_runner = True
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=dict(extra_config or {}),
    )
    return cast(VllmConfig, config)


def _make_kv_cache_config() -> KVCacheConfig:
    num_blocks = 16
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_tensor = KVCacheTensor(
        size=spec.page_size_bytes * num_blocks,
        layers=["layer"],
        layer_stride=spec.page_size_bytes * num_blocks,
        block_stride=spec.page_size_bytes,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )


def _make_identity_config() -> VllmConfig:
    config = _make_vllm_config()
    config.model_config.hf_text_config = PreTrainedConfig(
        max_position_embeddings=16,
        rope_parameters={
            "full_attention": {
                "rope_type": "linear",
                "factor": 1.0,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
            "sliding_attention": None,
        },
    )
    config.model_config.dtype = torch.bfloat16
    config.model_config.max_model_len = 16
    config.model_config.hf_config = config.model_config.hf_text_config
    config.model_config.quantization = None
    config.model_config.quantization_config = None
    return config


@pytest.fixture
def model_identity_config(tmp_path) -> ModelConfig:
    text_config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )
    config = LlavaConfig(
        architectures=["LlavaForConditionalGeneration"],
        text_config=text_config,
        vision_config=CLIPVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
        ),
    )
    config.save_pretrained(tmp_path)
    return ModelConfig(model=str(tmp_path), dtype="bfloat16", max_model_len=16)


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantization", "fp8"),
        ("quantization_config", {"linear": "fp8_per_tensor_dynamic"}),
        (
            "quantization_config",
            QuantizationConfigArgs(linear="fp8_per_tensor_dynamic"),
        ),
    ],
)
def test_runtime_quantization_overrides_isolate_model_identity(
    model_identity_config, field, value
):
    original = get_kv_cache_model_config_hash(model_identity_config)
    setattr(model_identity_config, field, value)
    assert get_kv_cache_model_config_hash(model_identity_config) != original


def test_multimodal_projector_settings_isolate_model_identity(model_identity_config):
    original = get_kv_cache_model_config_hash(model_identity_config)
    text_config = model_identity_config.hf_text_config.to_dict()
    model_identity_config.hf_config.projector_hidden_act = "silu"
    assert model_identity_config.hf_text_config.to_dict() == text_config
    assert get_kv_cache_model_config_hash(model_identity_config) != original


def _identity_mapper(config: VllmConfig, kv_config: KVCacheConfig) -> FileMapper:
    return FileMapper.from_offloading_spec(
        "/tmp/cache",
        MockOffloadingSpec(build_offloading_config(config, kv_config)),
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("rope_parameters", "full_attention", "factor"), 2.0),
        (("rope_parameters", "full_attention", "rope_theta"), 20000.0),
        (("rope_parameters", "full_attention", "partial_rotary_factor"), 0.5),
        (
            ("rope_parameters", "sliding_attention"),
            {"rope_type": "default", "rope_theta": 10000.0},
        ),
        (("max_position_embeddings",), 32),
    ],
)
def test_model_config_changes_isolate_persisted_blocks(path, value):
    config = _make_identity_config()
    first = _make_kv_cache_config()
    first.model_config_hash = get_kv_cache_model_config_hash(config.model_config)
    old_path = _identity_mapper(config, first).base_path

    hf_config = config.model_config.hf_text_config
    if len(path) == 1:
        setattr(hf_config, path[0], value)
    else:
        target = getattr(hf_config, path[0])
        for name in path[1:-1]:
            target = target[name]
        target[path[-1]] = value
    changed = _make_kv_cache_config()
    changed.model_config_hash = get_kv_cache_model_config_hash(config.model_config)
    assert _identity_mapper(config, changed).base_path != old_path


def test_model_dtype_isolates_blocks_with_the_same_kv_dtype():
    config = _make_identity_config()
    first = _make_kv_cache_config()
    first.model_config_hash = get_kv_cache_model_config_hash(config.model_config)
    config.model_config.dtype = torch.float16
    changed = _make_kv_cache_config()
    changed.model_config_hash = get_kv_cache_model_config_hash(config.model_config)
    first_mapper = _identity_mapper(config, first)
    changed_mapper = _identity_mapper(config, changed)
    assert first_mapper.fields["dtype"] == changed_mapper.fields["dtype"]
    assert first_mapper.base_path != changed_mapper.base_path


def test_metadata_and_mapping_order_preserve_model_identity():
    config = _make_identity_config()
    serialized = config.model_config.hf_text_config.to_dict()
    metadata = {
        "_name_or_path": "old-path",
        "_commit_hash": "old-revision",
        "transformers_version": "old-version",
    }
    serialized.update(metadata)
    serialized["decoder"] = {**metadata, "hidden_size": 8}
    # PreTrainedConfig.to_dict itself drops or rewrites some metadata. Return
    # the serialized values directly to exercise our own metadata filtering.
    config.model_config.hf_config = SimpleNamespace(to_dict=lambda: serialized)
    before = copy.deepcopy(serialized)
    expected = get_kv_cache_model_config_hash(config.model_config)
    assert serialized == before

    for name in metadata:
        serialized[name] = "new-value"
        serialized["decoder"][name] = "new-nested-value"
    serialized["rope_parameters"] = dict(
        reversed(serialized["rope_parameters"].items())
    )
    full = serialized["rope_parameters"]["full_attention"]
    serialized["rope_parameters"]["full_attention"] = dict(reversed(full.items()))
    assert get_kv_cache_model_config_hash(config.model_config) == expected


def test_engine_propagates_snapshot_after_model_loading_and_auto_fit(
    model_identity_config,
):
    from vllm.v1.engine.core import EngineCore

    config = VllmConfig(
        model_config=model_identity_config,
        cache_config=CacheConfig(block_size=16),
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both"
        ),
    )
    snapshot = get_kv_cache_model_config_hash(config.model_config)
    workers = [_make_kv_cache_config(), _make_kv_cache_config()]
    executor = MagicMock()
    executor.get_kv_cache_specs.return_value = [
        {"layer": worker.kv_cache_groups[0].kv_cache_spec} for worker in workers
    ]
    executor.determine_available_memory.return_value = [1 << 20] * len(workers)
    engine = SimpleNamespace(
        model_executor=executor,
        _kv_cache_model_config_hash=snapshot,
        available_gpu_memory_for_kv_cache=-1,
        collective_rpc=executor.collective_rpc,
    )
    config.compilation_config.compilation_time = 0
    config.compilation_config.encoder_compilation_time = 0
    config.cache_config.kv_cache_layout = "LBNHC"

    # Model loading may rewrite RoPE types; auto-fit may shorten the context.
    config.model_config.hf_text_config.rope_parameters["rope_type"] = "deepseek_yarn"

    def auto_fit(vllm_config, specs, available_memory, global_kv_cache_groups):
        vllm_config.model_config.max_model_len = 8
        return workers

    with (
        patch("vllm.v1.engine.core.register_all_kvcache_specs"),
        patch(
            "vllm.v1.engine.core.resolve_kv_cache_layout",
            return_value=SimpleNamespace(name="LBNHC"),
        ),
        patch("vllm.v1.engine.core.get_kv_cache_configs", side_effect=auto_fit),
        patch("vllm.v1.engine.core.update_kv_cache_capacity"),
    ):
        scheduler_config = EngineCore._initialize_kv_caches(engine, config)

    remote_configs = executor.initialize_from_config.call_args.args[0]
    assert len(remote_configs) == len(workers)
    assert scheduler_config is not workers[0]
    assert config.model_config.max_model_len == 8
    assert get_kv_cache_model_config_hash(config.model_config) != snapshot
    for retained in [*remote_configs, scheduler_config]:
        assert retained.model_config_hash == snapshot
        assert (
            _identity_mapper(config, retained).fields["model_config_hash"] == snapshot
        )


@pytest.mark.parametrize(
    "connector,should_snapshot",
    [
        pytest.param("OffloadingConnector", True, id="native"),
        pytest.param(["NixlConnector", "OffloadingConnector"], True, id="mixed-multi"),
        pytest.param(
            ["NixlConnector", ["LMCacheConnectorV1", "OffloadingConnector"]],
            True,
            id="nested-native",
        ),
        pytest.param(None, False, id="no-connector"),
        pytest.param("NixlConnector", False, id="nixl-with-unrelated-extra-config"),
        pytest.param("LMCacheConnectorV1", False, id="lmcache"),
        pytest.param(
            ["NixlConnector", ["LMCacheConnectorV1"]], False, id="nested-unrelated"
        ),
    ],
)
def test_engine_snapshots_identity_before_model_loading(connector, should_snapshot):
    from vllm.v1.engine.core import EngineCore

    def connector_options(name_or_children):
        if isinstance(name_or_children, list):
            return {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "connectors": [connector_options(c) for c in name_or_children]
                },
            }
        return {"kv_connector": name_or_children, "kv_role": "kv_both"}

    config = _make_identity_config()
    config.kv_transfer_config = (
        KVTransferConfig(**connector_options(connector))
        if connector is not None
        else None
    )
    if connector == "NixlConnector":
        config.kv_transfer_config.kv_connector_extra_config = {
            "connectors": [{"kv_connector": "OffloadingConnector"}]
        }
    expected = (
        get_kv_cache_model_config_hash(config.model_config) if should_snapshot else None
    )
    engine = EngineCore.__new__(EngineCore)

    def load_model(vllm_config):
        vllm_config.model_config.hf_text_config.rope_parameters["full_attention"][
            "rope_type"
        ] = "deepseek_yarn"
        return MagicMock()

    with (
        patch("vllm.plugins.load_general_plugins"),
        (
            nullcontext()
            if should_snapshot
            else patch.object(
                config.model_config.hf_config,
                "to_dict",
                side_effect=AssertionError(
                    "Unrelated connectors must not snapshot HF config"
                ),
            )
        ),
        patch.object(
            EngineCore, "_initialize_kv_caches", side_effect=RuntimeError("stop init")
        ),
        pytest.raises(RuntimeError, match="stop init"),
    ):
        EngineCore.__init__(engine, config, load_model, log_stats=False)

    assert engine._kv_cache_model_config_hash == expected
    if should_snapshot:
        assert get_kv_cache_model_config_hash(config.model_config) != expected


def test_longrope_runtime_choice_changes_keys_and_persistent_identity():
    from vllm.model_executor.layers.rotary_embedding.phi3_long_rope_scaled_rope import (  # noqa: E501
        Phi3LongRoPEScaledRotaryEmbedding,
    )

    config = _make_identity_config()
    config.model_config.hf_text_config.rope_parameters = {
        "rope_type": "longrope",
        "short_factor": [1.0] * 4,
        "long_factor": [2.0] * 4,
        "original_max_position_embeddings": 8,
    }
    paths, keys = [], []
    for max_model_len in (8, 16):
        config.model_config.max_model_len = max_model_len
        kv_config = _make_kv_cache_config()
        kv_config.model_config_hash = get_kv_cache_model_config_hash(
            config.model_config
        )
        paths.append(_identity_mapper(config, kv_config).base_path)
        with patch(
            "vllm.model_executor.layers.rotary_embedding."
            "phi3_long_rope_scaled_rope.get_current_vllm_config",
            return_value=config,
        ):
            rope = Phi3LongRoPEScaledRotaryEmbedding(
                8, 8, 16, 8, 10000.0, True, torch.float32, [1.0] * 4, [2.0] * 4
            )
        _, key = rope(torch.arange(8), torch.ones(8, 8), torch.ones(8, 8))
        keys.append(key)
    assert not torch.equal(keys[0], keys[1])
    assert paths[0] != paths[1]


def _make_sizing_kv_cache_config(packed: bool) -> KVCacheConfig:
    """One 16 byte-per-block allocation, described two ways.

    Packed: both layers are one dense run. Unpacked: the same bytes as two runs, the
    second starting after the first layer's region. Either way the connector accounts
    for 16 KV bytes per block.
    """
    num_blocks = 4
    page = 8
    size = 2 * page * num_blocks
    if packed:
        kv_cache_tensors = [
            KVCacheTensor(
                size=size,
                layers=["layer0", "layer1"],
                layer_stride=page * num_blocks,
                block_stride=page,
            )
        ]
    else:
        kv_cache_tensors = [
            KVCacheTensor(
                size=size,
                layers=[layer],
                layer_stride=page * num_blocks,
                block_stride=page,
                offset=i * page * num_blocks,
            )
            for i, layer in enumerate(("layer0", "layer1"))
        ]

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0", "layer1"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def _full_attention_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float32,
    )


def _mla_spec(
    block_size: int = 16,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
    tokens_per_state: int = 1,
) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=tokens_per_state,
    )


def _swa_spec(sliding_window: int = 128) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )


_HIDDEN_STATE_KWARGS: dict[str, Any] = {
    "block_size": 16,
    "num_kv_heads": 1,
    "head_size": 512,
    "dtype": torch.float32,
}

_MAMBA_SPEC = MambaSpec(
    block_size=16,
    shapes=((16, 1),),
    dtypes=(torch.float32,),
)
# Page sizes of the specs the replicated-layout cases below are built from.
_MLA_PAGE = _mla_spec().page_size_bytes
_HALF_MLA_PAGE = _mla_spec(head_size=256).page_size_bytes
_FULL_PAGE = _full_attention_spec().page_size_bytes
_MAMBA_PAGE = _MAMBA_SPEC.page_size_bytes


def _make_mla_kv_cache_config(
    layer_names: list[str] | None = None,
    head_size: int = 512,
    dtype: torch.dtype = torch.float32,
    num_blocks: int = 4,
) -> KVCacheConfig:
    if layer_names is None:
        layer_names = ["layer0", "layer1"]
    spec = _mla_spec(head_size=head_size, dtype=dtype)
    layer_stride = spec.page_size_bytes * num_blocks
    kv_cache_tensors = [
        KVCacheTensor(
            size=layer_stride * len(layer_names),
            layers=layer_names,
            layer_stride=layer_stride,
            block_stride=spec.page_size_bytes,
        )
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
    )


def _make_hybrid_kv_cache_config() -> KVCacheConfig:
    num_blocks = 4
    full_spec = _full_attention_spec(block_size=12)
    mla_spec = _mla_spec()
    # Mixed page sizes across overlaying groups: a block is a window of the largest
    # group's packing, so the layer dim sits inside the block dim.
    window = max(full_spec.page_size_bytes, mla_spec.page_size_bytes)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=window * num_blocks,
                layers=["full_layer"],
                layer_stride=full_spec.page_size_bytes,
                block_stride=window,
            ),
            KVCacheTensor(
                size=window * num_blocks,
                layers=["mla_layer"],
                layer_stride=mla_spec.page_size_bytes,
                block_stride=window,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["full_layer"], full_spec),
            KVCacheGroupSpec(["mla_layer"], mla_spec),
        ],
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _uniform_spec(spec_kind: str) -> UniformTypeKVCacheSpecs:
    # DSA models merge their indexer and MLA layers into a UniformType group,
    # a container rather than an AttentionSpec, but still sharded across DCP.
    specs: dict[str, Any] = (
        {"mla_layer": _mla_spec(), "indexer_layer": _mla_spec(head_size=128)}
        if spec_kind == "attention"
        else {"mamba_layer0": _mamba_spec(), "mamba_layer1": _mamba_spec()}
    )
    return UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs)


def _make_mamba_hybrid_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full_layer"], _full_attention_spec()),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=16,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )


def _parallelism_agnostic(
    kv_cache_groups: list[KVCacheGroupSpec],
    *,
    canonical: bool = False,
    v2: bool = True,
) -> bool:
    config = _make_vllm_config(
        extra_config={"canonical_layout": True} if canonical else None
    )
    config.use_v2_model_runner = v2
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
    )
    return build_offloading_config(
        config, kv_cache_config
    ).parallel.is_parallelism_agnostic


def _replicated_layout(
    kv_cache_config: KVCacheConfig,
    *,
    tensor_parallel_size: int = 4,
    pipeline_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    use_mla: bool = True,
    use_v2_model_runner: bool = True,
    distributed_executor_backend: Any = "mp",
    nnodes: int = 1,
    world_size: int | None = None,
) -> bool:
    config = _make_vllm_config(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
    )
    config.model_config.use_mla = use_mla
    config.use_v2_model_runner = use_v2_model_runner
    config.parallel_config.distributed_executor_backend = distributed_executor_backend
    config.parallel_config.nnodes = nnodes
    if world_size is not None:
        config.parallel_config.world_size = world_size
    kv_cache_config.kv_tp_replicas = kv_cache_groups_tp_replicas(
        kv_cache_config.kv_cache_groups,
        config.parallel_config.tensor_parallel_size,
        config.parallel_config.decode_context_parallel_size,
    )

    # Scheduler and Worker offload config
    worker_offload_config = build_offloading_config(config, kv_cache_config)
    scheduler_offload_config = build_offloading_config(
        config, generate_scheduler_kv_cache_config([kv_cache_config])
    )
    # Verify that the Worker and Scheduler agree on replication.
    # and worker-bytes-per-block
    assert (
        worker_offload_config.replicated_layout
        == scheduler_offload_config.replicated_layout
    )
    assert (
        worker_offload_config.worker_kv_bytes_per_block
        == scheduler_offload_config.worker_kv_bytes_per_block
    )

    return worker_offload_config.replicated_layout


_SWA_MLA_PAGE = SlidingWindowMLASpec(
    block_size=16,
    num_kv_heads=1,
    max_tp_shards=1,
    head_size=512,
    dtype=torch.float32,
    sliding_window=128,
).page_size_bytes


def _make_swa_mla_kv_cache_config(
    num_blocks: int = 4,
) -> KVCacheConfig:
    """Single SlidingWindowMLASpec group — DSV4 sliding-window layers."""
    spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.float32,
        sliding_window=128,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=_SWA_MLA_PAGE * num_blocks,
                layers=["layer"],
                layer_stride=_SWA_MLA_PAGE * num_blocks,
                block_stride=_SWA_MLA_PAGE,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )


def _make_dsv4_flash_kv_cache_config(num_blocks: int = 4) -> KVCacheConfig:
    """Minimal fake of DeepSeek-V4-Flash's five UniformTypeKVCacheSpecs groups.

    The real model has the following groups (layer counts from runtime logs);
    2 per group is enough to exercise the structural properties that govern
    replicated_layout:
        group[0]: 22 SlidingWindowMLASpec layers, sliding_window=128
        group[1]: 21 SlidingWindowMLASpec layers, sliding_window=128
        group[2]: 62 MLAAttentionSpec layers  (MLA + indexer.k_cache)
        group[3]: 42 SlidingWindowMLASpec layers, sliding_window=8
        group[4]: 20 SlidingWindowMLASpec layers, sliding_window=128
    All inner specs have num_kv_heads=1 → every group is TP-replicated.
    """
    block_size = 16
    dtype = torch.float32
    head_size = 512

    def _swa_group(tag: str, sw: int) -> KVCacheGroupSpec:
        names = [f"{tag}_0", f"{tag}_1"]
        spec = SlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=head_size,
            dtype=dtype,
            sliding_window=sw,
        )
        return KVCacheGroupSpec(
            names,
            UniformTypeKVCacheSpecs(
                block_size=block_size,
                kv_cache_specs={name: spec for name in names},
            ),
        )

    def _mla_group(tag: str) -> KVCacheGroupSpec:
        names = [f"{tag}_0", f"{tag}_1"]
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=head_size,
            dtype=dtype,
        )
        return KVCacheGroupSpec(
            names,
            UniformTypeKVCacheSpecs(
                block_size=block_size,
                kv_cache_specs={name: spec for name in names},
            ),
        )

    groups = [
        _swa_group("swa_128a", 128),  # group[0]: SWA-MLA, sw=128
        _swa_group("swa_128b", 128),  # group[1]: SWA-MLA, sw=128
        _mla_group("mla"),  # group[2]: MLA + indexer layers
        _swa_group("swa_8", 8),  # group[3]: SWA-MLA, sw=8
        _swa_group("swa_128c", 128),  # group[4]: SWA-MLA, sw=128
    ]
    # All specs share the same page size (num_kv_heads=1, head_size=512, fp32).
    page = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=head_size, dtype=dtype
    ).page_size_bytes
    all_layers = [name for g in groups for name in g.layer_names]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=page * len(all_layers) * num_blocks,
                layers=all_layers,
                layer_stride=page * num_blocks,
                block_stride=page,
            )
        ],
        kv_cache_groups=groups,
    )


def _make_dsv3_2_kv_cache_config(num_blocks: int = 4) -> KVCacheConfig:
    """Minimal fake of DeepSeek-V3.2-Exp's single UniformTypeKVCacheSpecs group.

    The real model has 1 group (layer counts from runtime logs):
        group[0]: 122 MLAAttentionSpec layers (self_attn.attn + indexer.k_cache)
    All inner specs have num_kv_heads=1 → the group is TP-replicated.

    GLM-5.3 has the same topology (one UniformType group of MLAAttentionSpec
    layers, num_kv_heads=1) with 99 layers instead of 122, so this config
    covers both models.
    """
    block_size = 16
    names = [f"mla_{i}" for i in range(4)]  # 4 layers; real model has 122
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.float32,
    )
    group = KVCacheGroupSpec(
        names,
        UniformTypeKVCacheSpecs(
            block_size=block_size,
            kv_cache_specs={name: spec for name in names},
        ),
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * len(names) * num_blocks,
                layers=names,
                layer_stride=spec.page_size_bytes * num_blocks,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[group],
    )


@pytest.mark.parametrize("packed", [False, True])
def test_worker_kv_bytes_preserves_tensor_layout(packed: bool):
    config = _make_vllm_config(
        extra_config={"block_size": 32},
        tensor_parallel_size=3,
        pipeline_parallel_size=2,
    )

    offloading_config = build_offloading_config(
        config, _make_sizing_kv_cache_config(packed)
    )

    assert offloading_config.worker_kv_bytes_per_block == 16
    assert offloading_config.parallel.world_size == 6
    assert offloading_config.cache.blocks_per_chunk == 2


def test_offloading_skips_scratch_group():
    kv_cache_config = _make_mamba_hybrid_kv_cache_config()
    kv_cache_config.kv_cache_groups.insert(
        1,
        KVCacheGroupSpec(
            ["scratch"],
            CircularBufferSpec(
                block_size=4, num_kv_heads=1, head_size=128, dtype=torch.float32
            ),
        ),
    )
    page_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
    kv_cache_config.kv_cache_tensors = [
        KVCacheTensor(
            size=page_size * kv_cache_config.num_blocks,
            layers=["full_layer", "scratch", "mamba_layer"],
            layer_stride=0,
            block_stride=page_size,
        )
    ]
    config = _make_vllm_config()

    offloading_config = build_offloading_config(config, kv_cache_config)
    scheduler_config = SchedulerOffloadConfig.from_spec(
        MockOffloadingSpec(offloading_config), config, kv_cache_config
    )

    assert [group.group_id for group in offloading_config.groups] == [0, 2]
    assert [group.group_idx for group in scheduler_config.kv_group_configs] == [0, 2]
    assert offloading_config.worker_kv_bytes_per_block == page_size


def test_hisparse_offloads_only_indexer_group():
    source = KVCacheGroupSpec(
        ["source"],
        _full_attention_spec(),
        host_resident=True,
    )
    indexer = KVCacheGroupSpec(
        ["indexer"],
        _full_attention_spec(),
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[source, indexer],
        hisparse_host_num_blocks=4,
    )
    config = _make_vllm_config()

    offloading_config = build_offloading_config(config, kv_cache_config)
    scheduler_config = SchedulerOffloadConfig.from_spec(
        MockOffloadingSpec(offloading_config), config, kv_cache_config
    )

    assert [
        (group.group_id, group.layer_names) for group in offloading_config.groups
    ] == [(1, ("indexer",))]
    assert [group.group_idx for group in scheduler_config.kv_group_configs] == [1]


def test_hisparse_partial_group_size_survives_scheduler_flattening():
    """Worker and scheduler representations must produce identical mmap rows."""
    layer_specs = {name: _full_attention_spec() for name in ("indexer.0", "indexer.1")}
    wrapped = UniformTypeKVCacheSpecs.from_specs(layer_specs)
    assert wrapped is not None
    source = KVCacheGroupSpec(
        ["source"],
        _full_attention_spec(),
        host_resident=True,
        role=KVCacheGroupRole.HISPARSE_SOURCE,
    )
    worker_indexer = KVCacheGroupSpec(
        list(layer_specs), wrapped, role=KVCacheGroupRole.HISPARSE_INDEXER
    )
    scheduler_indexer = KVCacheGroupSpec(
        list(layer_specs),
        next(iter(layer_specs.values())),
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )

    def make_config(indexer: KVCacheGroupSpec) -> KVCacheConfig:
        return KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[],
            kv_cache_groups=[source, indexer],
            hisparse_host_num_blocks=4,
        )

    config = _make_vllm_config()
    worker = build_offloading_config(config, make_config(worker_indexer))
    scheduler = build_offloading_config(config, make_config(scheduler_indexer))

    assert worker.worker_kv_bytes_per_block == scheduler.worker_kv_bytes_per_block
    assert worker.worker_kv_bytes_per_block == wrapped.page_size_bytes


def test_zero_blocks_skips_tensor_layout_validation():
    kv_cache_config = _make_sizing_kv_cache_config(packed=False)
    kv_cache_config.num_blocks = 0

    offloading_config = build_offloading_config(_make_vllm_config(), kv_cache_config)

    assert offloading_config.worker_kv_bytes_per_block == 0


def test_prefill_context_parallelism_does_not_scale_group_blocks():
    config = _make_vllm_config(
        extra_config={"block_size": 64},
        prefill_context_parallel_size=2,
    )

    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (16,)
    assert offloading_config.cache.tokens_per_hash == 16
    assert offloading_config.cache.blocks_per_chunk == 4


def test_dcp_scales_attention_but_not_mamba_group_blocks():
    config = _make_vllm_config(tensor_parallel_size=2, decode_context_parallel_size=2)
    config.speculative_config = None

    offloading_config = build_offloading_config(
        config, _make_mamba_hybrid_kv_cache_config()
    )

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        32,
        16,
    )
    scheduler_config = SchedulerOffloadConfig.from_spec(
        MockOffloadingSpec(offloading_config),
        config,
        _make_mamba_hybrid_kv_cache_config(),
    )
    mamba_group = scheduler_config.kv_group_configs[1]
    assert mamba_group.sliding_window_size_in_chunks == 1
    assert scheduler_config.alignment_tokens is not None
    manager_cls = KVCacheSpecRegistry.get_manager_class(mamba_group.kv_cache_spec)
    assert manager_cls is not None
    block_mask = manager_cls.reachable_block_mask(
        start_block=0,
        end_block=4,
        alignment_tokens=scheduler_config.alignment_tokens,
        kv_cache_spec=mamba_group.kv_cache_spec,
        use_eagle=mamba_group.is_eagle_group,
    )
    assert block_mask is None or [i for i in range(4) if block_mask[i]] == [1, 3]


@pytest.mark.parametrize("dcp_size,expected", [(1, 16), (2, 32)])
def test_dcp_scales_uniform_type_attention_group_blocks(dcp_size, expected):
    config = _make_vllm_config(
        tensor_parallel_size=2, decode_context_parallel_size=dcp_size
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=64,
                layers=["mla_layer", "indexer_layer"],
                layer_stride=32,
                block_stride=8,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["mla_layer", "indexer_layer"], _uniform_spec("attention"))
        ],
    )

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert offloading_config.groups[0].tokens_per_block == expected
    assert offloading_config.cache.tokens_per_hash == expected


@pytest.mark.parametrize(
    "spec_kind,expected",
    [
        ("attention", 32),
        # Mamba state is replicated across DCP ranks, so a uniform group of
        # Mamba layers keeps its span.
        ("mamba", 16),
    ],
)
def test_dcp_scales_uniform_type_group_alongside_mamba(spec_kind, expected):
    config = _make_vllm_config(tensor_parallel_size=2, decode_context_parallel_size=2)
    config.speculative_config = None
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer0", "layer1"], _uniform_spec(spec_kind)),
            KVCacheGroupSpec(["mamba_layer"], _mamba_spec()),
        ],
    )

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        expected,
        16,
    )


@pytest.mark.parametrize(
    "kv_cache_spec,window_chunks",
    [
        (_full_attention_spec(), None),
        (_mla_spec(), None),
        (HiddenStateCacheSpec(**_HIDDEN_STATE_KWARGS), None),
        (_MAMBA_SPEC, 1),
        (_swa_spec(), 8),
        (
            UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs={
                    "mla": _mla_spec(),
                    "hidden": HiddenStateCacheSpec(**_HIDDEN_STATE_KWARGS),
                },
            ),
            None,
        ),
        (
            UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs={"swa0": _swa_spec(), "swa1": _swa_spec()},
            ),
            8,
        ),
    ],
)
def test_group_carries_the_chunks_the_tier_keeps_per_request(
    kv_cache_spec: KVCacheSpec, window_chunks: int | None
):
    """The group reports how far back its attention reaches, counted in chunks.

    A capacity estimate needs that bound: a windowed or recurrent group keeps a
    fixed number of chunks however long the request grows, while an unbounded
    group keeps one for every chunk of the request. A worker-side group hands
    over an aggregate of its layers, which must resolve to the same bound as the
    single layer it wraps.
    """
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    offloading_config = build_offloading_config(_make_vllm_config(), kv_cache_config)

    assert offloading_config.groups[0].sliding_window_size_in_chunks == window_chunks


def test_model_config_carries_the_longest_request():
    """The capacity estimate of a tier holds at max_model_len, so it must travel.

    CPUOffloadingInfo.from_config in vllm/v1/kv_offload/cpu/common.py reads it.
    A model config without it leaves the estimate unknown.
    """
    vllm_config = _make_vllm_config()
    vllm_config.model_config.max_model_len = 32768
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], _full_attention_spec())],
    )

    offloading_config = build_offloading_config(vllm_config, kv_cache_config)

    assert offloading_config.model.max_model_len == 32768


def test_preserves_data_parallel_config():
    config = _make_vllm_config()
    config.parallel_config.data_parallel_index = 2
    config.parallel_config.data_parallel_size = 4
    config.parallel_config.data_parallel_rank_local = 1

    offloading_config = build_offloading_config(config, _make_kv_cache_config())

    assert offloading_config.parallel.data_parallel_index == 2
    assert offloading_config.parallel.data_parallel_size == 4
    assert offloading_config.parallel.data_parallel_rank_local == 1


def test_resolves_heterogeneous_hybrid_block_sizes():
    config = _make_vllm_config()
    config.cache_config.block_size = 4

    offloading_config = build_offloading_config(config, _make_hybrid_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        12,
        16,
    )
    assert offloading_config.cache.tokens_per_hash == 4
    assert offloading_config.cache.blocks_per_chunk == 1


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_replicated_layout_enabled_for_pure_mla_tp_mp_single_node(
    world_size: int,
    use_v2_model_runner: bool,
):
    assert _replicated_layout(
        _make_mla_kv_cache_config(),
        tensor_parallel_size=world_size,
        use_v2_model_runner=use_v2_model_runner,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_replicated_layout_enabled_for_swa_mla(world_size: int):
    """SlidingWindowMLASpec groups are also replicated and qualify."""
    assert _replicated_layout(
        _make_swa_mla_kv_cache_config(),
        tensor_parallel_size=world_size,
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_replicated_layout_dsv4_flash(world_size: int):
    """DeepSeek-V4-Flash: all 5 UniformType groups are MLA/SWA-MLA with
    num_kv_heads=1, so replicated_layout must be True under TP/mp single-node."""
    assert _replicated_layout(
        _make_dsv4_flash_kv_cache_config(),
        tensor_parallel_size=world_size,
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_replicated_layout_dsv3_2(world_size: int):
    """DeepSeek-V3.2 / GLM-5.3: single UniformType group of MLAAttentionSpec
    layers (122 and 99 respectively), num_kv_heads=1 → replicated_layout must
    be True under TP/mp single-node."""
    assert _replicated_layout(
        _make_dsv3_2_kv_cache_config(),
        tensor_parallel_size=world_size,
    )


def test_replicated_layout_enabled_for_uniform_wrapper_mla():
    """UniformTypeKVCacheSpecs wrapping MLA layers of different sizes qualifies:
    each layer is replicated and the page accounting sums correctly."""
    mla_layers = ["layer0", "layer1"]
    spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"layer0": _mla_spec(), "layer1": _mla_spec(head_size=256)},
    )
    total_page = spec.page_size_bytes
    num_blocks = 4
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=total_page * num_blocks,
                layers=mla_layers,
                layer_stride=total_page * num_blocks,
                block_stride=total_page,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(mla_layers, spec)],
    )
    assert _replicated_layout(kv_cache_config)


@pytest.mark.parametrize(
    ("kv_cache_config", "case"),
    [
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_MLA_PAGE * 4,
                        layers=["layer"],
                        layer_stride=_MLA_PAGE * 4,
                        block_stride=_MLA_PAGE,
                    )
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(
                        ["layer"],
                        HiddenStateCacheSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=512,
                            dtype=torch.float32,
                        ),
                    )
                ],
            ),
            "hidden-state",
        ),
        (
            # Overlaid groups with different page sizes: a block is a window of
            # the largest group's packing.
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_FULL_PAGE * 4,
                        layers=["mla"],
                        layer_stride=_MLA_PAGE,
                        block_stride=_FULL_PAGE,
                    ),
                    KVCacheTensor(
                        size=_FULL_PAGE * 4,
                        layers=["full"],
                        layer_stride=_FULL_PAGE,
                        block_stride=_FULL_PAGE,
                    ),
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(["mla"], _mla_spec()),
                    KVCacheGroupSpec(["full"], _full_attention_spec()),
                ],
            ),
            "mla-full-hybrid",
        ),
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_MLA_PAGE * 4,
                        layers=["mla"],
                        layer_stride=_MLA_PAGE,
                        block_stride=_MLA_PAGE,
                    ),
                    KVCacheTensor(
                        size=_MLA_PAGE * 4,
                        layers=["mamba"],
                        layer_stride=_MAMBA_PAGE,
                        block_stride=_MLA_PAGE,
                    ),
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(["mla"], _mla_spec()),
                    KVCacheGroupSpec(["mamba"], _MAMBA_SPEC),
                ],
            ),
            "mla-mamba-hybrid",
        ),
    ],
    ids=[
        "hidden-state",
        "mla-full-hybrid",
        "mla-mamba-hybrid",
    ],
)
def test_replicated_layout_excludes_unproven_cache_shapes(
    kv_cache_config: KVCacheConfig,
    case: str,
):
    assert not _replicated_layout(kv_cache_config), case


@pytest.mark.parametrize(
    ("kv_cache_config", "case"),
    [
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_MLA_PAGE * 4,
                        layers=["layer"],
                        layer_stride=_MLA_PAGE * 4,
                        block_stride=_MLA_PAGE,
                    )
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(
                        ["layer"],
                        SlidingWindowMLASpec(
                            block_size=16,
                            num_kv_heads=1,
                            max_tp_shards=1,
                            head_size=512,
                            dtype=torch.float32,
                            sliding_window=128,
                        ),
                    )
                ],
            ),
            "sliding-window-mla",
        ),
        (
            # One group, two page sizes: layer1's run starts past layer0's region.
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=(_MLA_PAGE + _HALF_MLA_PAGE) * 4,
                        layers=["layer0"],
                        layer_stride=_MLA_PAGE * 4,
                        block_stride=_MLA_PAGE,
                    ),
                    KVCacheTensor(
                        size=(_MLA_PAGE + _HALF_MLA_PAGE) * 4,
                        layers=["layer1"],
                        layer_stride=_HALF_MLA_PAGE * 4,
                        block_stride=_HALF_MLA_PAGE,
                        offset=_MLA_PAGE * 4,
                    ),
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(
                        ["layer0", "layer1"],
                        UniformTypeKVCacheSpecs(
                            block_size=16,
                            kv_cache_specs={
                                "layer0": _mla_spec(),
                                "layer1": _mla_spec(head_size=256),
                            },
                        ),
                    )
                ],
            ),
            "uniform-wrapper",
        ),
        (
            KVCacheConfig(
                num_blocks=4,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=_MLA_PAGE * 4,
                        layers=[layer],
                        layer_stride=_MLA_PAGE * 4,
                        block_stride=_MLA_PAGE,
                    )
                    for layer in ("layer0", "layer1")
                ],
                kv_cache_groups=[
                    KVCacheGroupSpec(["layer0"], _mla_spec()),
                    KVCacheGroupSpec(["layer1"], _mla_spec()),
                ],
            ),
            "multi-group-mla",
        ),
    ],
    ids=[
        "sliding-window-mla",
        "uniform-wrapper",
        "multi-group-mla",
    ],
)
def test_replicated_layout_supported_cache_shapes(
    kv_cache_config: KVCacheConfig,
    case: str,
):
    assert _replicated_layout(kv_cache_config), case


@pytest.mark.parametrize(
    ("main_spec", "add_spec", "expected"),
    [
        (
            _mla_spec(head_size=512),
            _mla_spec(head_size=128, dtype=torch.uint8),  # indexer.k_cache spec
            True,
        ),
        (
            _mla_spec(head_size=512),
            _full_attention_spec(),
            False,
        ),
    ],
    ids=["uniform-mla-indexer", "uniform-mla-fa"],
)
def test_replicated_layout_for_dsa_style_groups(
    main_spec: KVCacheSpec, add_spec: KVCacheSpec, expected: bool
):
    """KV Cache groups with packed-specs inside a UniformTypeKVCacheSpec."""
    num_blocks = 4
    main_layers = [f"main_{i}" for i in range(61)]
    add_layers = [f"add_{i}" for i in range(61)]

    main_bytes = main_spec.page_size_bytes * len(main_layers)
    add_bytes = add_spec.page_size_bytes * len(add_layers)
    size = (main_bytes + add_bytes) * num_blocks

    kv_cache_specs = {
        **{layer: main_spec for layer in main_layers},
        **{layer: add_spec for layer in add_layers},
    }
    layer_names = main_layers + add_layers
    group_spec: KVCacheSpec = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs=kv_cache_specs
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=size,
                layers=main_layers,
                layer_stride=main_spec.page_size_bytes * num_blocks,
                block_stride=main_spec.page_size_bytes,
            ),
            KVCacheTensor(
                size=size,
                layers=add_layers,
                layer_stride=add_spec.page_size_bytes * num_blocks,
                block_stride=add_spec.page_size_bytes,
                offset=main_bytes * num_blocks,
            ),
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names, group_spec)],
    )

    assert _replicated_layout(kv_cache_config) == expected


@pytest.mark.parametrize(
    ("kwargs", "case"),
    [
        ({"tensor_parallel_size": 1}, "tp1"),
        ({"use_mla": False}, "use-mla-false"),
        ({"pipeline_parallel_size": 2, "world_size": 4}, "pp2"),
        ({"prefill_context_parallel_size": 2, "world_size": 4}, "pcp2"),
        ({"decode_context_parallel_size": 2}, "dcp2"),
        ({"world_size": 8}, "world-ne-tp"),
        ({"distributed_executor_backend": "ray"}, "ray"),
        ({"distributed_executor_backend": "uni"}, "uni"),
        ({"distributed_executor_backend": type("DummyExecutor", (), {})}, "class"),
        ({"nnodes": 2}, "multi-node"),
    ],
    ids=[
        "tp1",
        "use-mla-false",
        "pp2",
        "pcp2",
        "dcp2",
        "world-ne-tp",
        "ray",
        "uni",
        "class",
        "multi-node",
    ],
)
def test_replicated_layout_parallel_gate(kwargs: dict[str, Any], case: str):
    assert not _replicated_layout(_make_mla_kv_cache_config(), **kwargs), case


def test_parallelism_agnostic_for_single_full_attention_group():
    # The direct (non-canonical) parallelism-agnostic layout is MRV1-only.
    assert _parallelism_agnostic(
        [KVCacheGroupSpec(["l0"], _full_attention_spec())], v2=False
    )


_SWA_SPEC = _swa_spec()
_SWA_MLA_SPEC = SlidingWindowMLASpec(
    block_size=16,
    num_kv_heads=1,
    max_tp_shards=1,
    head_size=576,
    dtype=torch.float32,
    sliding_window=128,
)


def _groups(*specs: KVCacheSpec) -> list[KVCacheGroupSpec]:
    return [KVCacheGroupSpec([f"l{i}"], spec) for i, spec in enumerate(specs)]


def _uniform_group(*specs: KVCacheSpec) -> KVCacheGroupSpec:
    """GLM-5.2/DSv3.2-style wrapper: same-type layers whose specs differ."""
    names = [f"l{i}" for i in range(len(specs))]
    return KVCacheGroupSpec(
        names,
        UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=dict(zip(names, specs))),
    )


@pytest.mark.parametrize(
    "kv_cache_groups",
    [
        _groups(_mla_spec(head_size=576)),
        _groups(_SWA_SPEC),
        _groups(_full_attention_spec(), _full_attention_spec()),
    ],
)
def test_parallelism_agnostic_excluded(kv_cache_groups: list[KVCacheGroupSpec]):
    # The direct layout is never parallelism-agnostic on MRV2, so check the
    # MRV1 gate.
    assert not _parallelism_agnostic(kv_cache_groups, v2=False)


@pytest.mark.parametrize(
    ("kv_cache_groups", "certified"),
    [
        pytest.param(_groups(_mla_spec(head_size=576)), True, id="mla-latent"),
        pytest.param(
            _groups(_full_attention_spec(), _SWA_SPEC), True, id="attention-hybrid"
        ),
        pytest.param(_groups(_SWA_MLA_SPEC), False, id="swa-mla"),
        pytest.param(
            list(_make_mamba_hybrid_kv_cache_config().kv_cache_groups),
            False,
            id="mamba-hybrid",
        ),
        pytest.param(
            [_uniform_group(_mla_spec(head_size=576), _mla_spec(head_size=128))],
            True,
            id="uniform-mla-wrapper",
        ),
        pytest.param(
            [_uniform_group(_mla_spec(head_size=576), _mla_spec(tokens_per_state=2))],
            False,
            id="uniform-uncertifiable-inner",
        ),
    ],
)
def test_canonical_layout_gate(kv_cache_groups, certified):
    """The canonical layout certifies portability group by group; none of
    these shapes are portable in the direct layout."""
    assert not _parallelism_agnostic(kv_cache_groups, v2=False)
    assert _parallelism_agnostic(kv_cache_groups, canonical=True) is certified


@pytest.mark.parametrize(
    ("spec"),
    [_SWA_MLA_SPEC, _mla_spec(tokens_per_state=2)],
    ids=["swa", "compressed-mla"],
)
def test_uncertifiable_canonical_layout_resets_replication(spec: KVCacheSpec):
    def _get_offloading_config(canonical: bool):
        config = _make_vllm_config(
            tensor_parallel_size=2,
            extra_config={"canonical_layout": True} if canonical else None,
        )
        config.model_config.use_mla = True

        PAGE = spec.page_size_bytes
        kv_config = KVCacheConfig(
            num_blocks=4,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=PAGE * 4,
                    layers=["layer"],
                    layer_stride=PAGE * 4,
                    block_stride=PAGE,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer"], spec),
            ],
        )

        kv_config.kv_tp_replicas = kv_cache_groups_tp_replicas(
            kv_config.kv_cache_groups,
            config.parallel_config.tensor_parallel_size,
            config.parallel_config.decode_context_parallel_size,
        )

        return build_offloading_config(config, kv_config)

    offload_config = _get_offloading_config(canonical=False)
    assert offload_config.replicated_layout
    assert not offload_config.parallel.is_parallelism_agnostic

    canon_offload_config = _get_offloading_config(canonical=True)
    assert not canon_offload_config.replicated_layout
    assert not canon_offload_config.parallel.is_parallelism_agnostic


def test_canonical_layout_certifies_v2_model_runner():
    """Canonical bytes are certified per layer against live tensor strides at
    registration, so the static gate must not depend on the model-runner
    version — the v2 runner is the case the canonical layout exists for."""
    groups = _groups(_full_attention_spec())
    assert _parallelism_agnostic(groups, canonical=True)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("scheduler", [False, True])
def test_canonical_mla_dsa_rows_are_tp_independent(tp_size, scheduler):
    specs = {
        f"l{i}": _mla_spec(block_size=64, head_size=size, dtype=torch.uint8)
        for i, size in enumerate([656] * 4 + [132] * 3)
    }
    uniform = UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs)
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * uniform.page_size_bytes,
                layers=list(specs),
                layer_stride=0,
                block_stride=uniform.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(list(specs), uniform)],
        model_config_hash="test-model-config",
    )
    if scheduler:
        kv_cache_config = generate_scheduler_kv_cache_config([kv_cache_config])
    config = _make_vllm_config(
        tensor_parallel_size=tp_size,
        extra_config={"canonical_layout": True, "cpu_bytes_to_use": 196608 * 8},
    )
    config.cache_config.kv_cache_layout = "LBHNC"
    config.parallel_config.distributed_executor_backend = "mp"
    offloading_config = build_offloading_config(config, kv_cache_config)
    spec = TieringOffloadingSpec(offloading_config)
    assert spec.kv_bytes_per_chunk == 196608
    assert spec.num_chunks == 8
    mapper = FileMapper.from_offloading_spec("/cache", spec, parallel_agnostic=True)
    assert mapper.get_run_config()["tp_size"] == 1
    assert mapper.get_run_config()["replicated_layout"] is True


def test_parallelism_agnostic_disabled_on_v2_model_runner():
    config = _make_vllm_config()
    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["l0"], _full_attention_spec())],
    )

    offloading_config = build_offloading_config(config, kv_cache_config)

    assert not offloading_config.parallel.is_parallelism_agnostic


def test_accepts_blocks_per_chunk_for_heterogeneous_groups():
    config = _make_vllm_config(extra_config={"blocks_per_chunk": 2})

    offloading_config = build_offloading_config(config, _make_hybrid_kv_cache_config())

    assert tuple(group.tokens_per_block for group in offloading_config.groups) == (
        12,
        16,
    )
    assert offloading_config.cache.blocks_per_chunk == 2


def test_block_size_and_blocks_per_chunk_are_mutually_exclusive():
    config = _make_vllm_config(extra_config={"block_size": 64, "blocks_per_chunk": 2})

    with pytest.raises(ValueError, match="Specify only one"):
        build_offloading_config(config, _make_kv_cache_config())


def test_blocks_per_chunk_must_be_positive():
    config = _make_vllm_config(extra_config={"blocks_per_chunk": 0})

    with pytest.raises(ValueError, match="greater than 0"):
        build_offloading_config(config, _make_kv_cache_config())
