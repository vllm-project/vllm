# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.models.qwen4_exp.common.qsa_cache import (
    QSACompressedKeyCache,
    QSAStateBackend,
    _build_qsa_metadata_torch,
    compressed_qsa_slot_mapping,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.core.kv_cache_planning import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    map_kv_caches_to_kernel_blocks,
    prepare_kernel_block_sizes,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 784


def test_compressed_qsa_null_block_has_no_slot() -> None:
    """The reserved null block must not receive compressed-cache writes."""
    slots = compressed_qsa_slot_mapping(
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([3, 7], dtype=torch.int64),
        storage_block_size=1,
        compress_ratio=4,
    )
    assert slots.tolist() == [-1, 1]


def test_torch_qsa_metadata_masks_null_block() -> None:
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=8,
        max_query_len=8,
        max_seq_len=8,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.arange(8, dtype=torch.int64),
        _token_to_req_indices_cache=torch.zeros(8, dtype=torch.int32),
    )
    slots = _build_qsa_metadata_torch(
        metadata,
        torch.empty(8, dtype=torch.int32),
        torch.empty(8, dtype=torch.int64),
        torch.empty(8, dtype=torch.int32),
        torch.empty(8, dtype=torch.int64),
        storage_block_size=16,
        compress_ratio=4,
    )[3]
    assert slots.tolist() == [-1] * 8


class _IndependentSelectorWithoutGenericSlots(MLAAttentionSpec):
    @property
    def has_independent_slot_mapping(self) -> bool:
        return True

    @property
    def uses_slot_mapping(self) -> bool:
        return False


def _full(block_size: int, head_size: int = 128) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _independent_selector(
    block_size: int, *, tokens_per_state: int = 8, dcp_sharded: bool = False
) -> _IndependentSelectorWithoutGenericSlots:
    return _IndependentSelectorWithoutGenericSlots(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=tokens_per_state,
        dcp_sharded=dcp_sharded,
    )


def _main_kv() -> FullAttentionSpec:
    return _full(BLOCK_SIZE)


def _selector(dcp_world_size: int) -> MLAAttentionSpec:
    block_size = BLOCK_SIZE * dcp_world_size
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=8,
        dcp_sharded=False,
    )


def _qsa_selector(dcp_world_size: int, compress_ratio: int = 8) -> MLAAttentionSpec:
    cache = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_world_size)
    )
    selector = QSACompressedKeyCache.get_kv_cache_spec(cache, config)
    assert isinstance(selector, MLAAttentionSpec)
    return selector


def _config(
    dcp_world_size: int,
    *,
    max_model_len: int = BLOCK_SIZE * 16,
    layout: str = "BLHNC",
) -> VllmConfig:
    config = VllmConfig(
        model_config=ModelConfig(max_model_len=BLOCK_SIZE * 16),
        device_config=DeviceConfig(device="cpu"),
    )
    config.model_config.max_model_len = max_model_len
    config.parallel_config.decode_context_parallel_size = dcp_world_size
    config.cache_config.kv_cache_layout = layout
    return config


@pytest.mark.parametrize("dcp_world_size", [2, 4])
def test_unmarked_replicated_cache_stays_separate(dcp_world_size: int) -> None:
    main = _main_kv()
    selector = _selector(dcp_world_size)
    specs = {"main": main, "selector": selector}
    assert not UniformTypeKVCacheSpecs.is_uniform_type(specs, dcp_world_size)
    group = UniformTypeKVCacheSpecs.from_specs(specs, dcp_world_size)
    assert group is None


@pytest.mark.parametrize(
    "dcp_world_size,selector_first", [(1, False), (2, False), (4, True)]
)
def test_qsa_selector_shares_sharded_main_block_table(
    dcp_world_size: int, selector_first: bool
) -> None:
    config = _config(dcp_world_size)
    selector = _qsa_selector(dcp_world_size)
    specs = (
        {"selector": selector, "main": _main_kv()}
        if selector_first
        else {"main": _main_kv(), "selector": selector}
    )
    groups = get_kv_cache_groups(config, specs)

    assert len(groups) == 1
    group = groups[0]
    assert group.layer_names == list(specs)
    assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    assert group.kv_cache_spec.block_size == BLOCK_SIZE
    assert group.kv_cache_spec.dcp_sharded
    assert (
        group.kv_cache_spec.max_num_blocks_per_req(
            config, config.model_config.max_model_len
        )
        == 16 // dcp_world_size
    )
    scheduler_config = generate_scheduler_kv_cache_config(
        [KVCacheConfig(128, [], groups)]
    )
    scheduler_spec = scheduler_config.kv_cache_groups[0].kv_cache_spec
    assert scheduler_spec.dcp_sharded
    assert scheduler_spec.block_size == BLOCK_SIZE
    assert type(scheduler_spec) is FullAttentionSpec
    assert selector.num_states == BLOCK_SIZE * dcp_world_size // 8


def test_different_global_spans_stay_separate_in_grouping() -> None:
    config = _config(2)
    selector = replace(_qsa_selector(1), dcp_sharded=False)
    groups = get_kv_cache_groups(config, {"main": _main_kv(), "selector": selector})
    assert len(groups) == 2


def test_independent_selector_cannot_disable_main_slot_mapping() -> None:
    config = _config(2)
    selector = _independent_selector(BLOCK_SIZE * 2)
    specs = {"selector": selector, "main": _main_kv()}
    groups = get_kv_cache_groups(config, specs)
    assert len(groups) == 1
    assert groups[0].kv_cache_spec.uses_slot_mapping


class _QSAMainBackend:
    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(spec=None):
        return [MultipleOf(16)]


def _packed_qsa_views(dcp_world_size: int):
    config = _config(dcp_world_size, layout="BLNHC")
    main = _full(BLOCK_SIZE, head_size=256)
    selector = _qsa_selector(dcp_world_size, compress_ratio=4)
    specs = {"main": main, "selector": selector}
    groups = get_kv_cache_groups(config, specs)
    cache = get_kv_cache_config_from_groups(
        config,
        groups,
        available_memory=2 * sum(spec.page_size_bytes for spec in specs.values()),
    )
    attn_groups = [
        [
            AttentionGroup(
                QSAStateBackend if name == "selector" else _QSAMainBackend,
                [name],
                specs[name],
                group_id,
            )
            for name in group.layer_names
        ]
        for group_id, group in enumerate(groups)
    ]
    kernel_sizes = prepare_kernel_block_sizes(cache, attn_groups)
    views = allocate_kv_cache(
        cache, torch.device("cpu"), KVCacheLayout.BLNHC, kernel_sizes
    )
    selector_group = next(g for g in attn_groups[0] if g.layer_names == ["selector"])
    return SimpleNamespace(
        config=config,
        groups=groups,
        cache=cache,
        attn_groups=attn_groups,
        kernel_sizes=kernel_sizes,
        views=views,
        selector=selector,
        selector_group=selector_group,
    )


def test_packed_replicated_selector_maps_two_manager_blocks_without_aliasing() -> None:
    """The selector's virtual pages must stay inside its packed physical pages."""
    run = _packed_qsa_views(2)
    assert len(run.groups) == 1
    assert run.groups[0].kv_cache_spec.block_size == BLOCK_SIZE
    assert run.selector.block_size == 2 * BLOCK_SIZE
    assert run.kernel_sizes == [BLOCK_SIZE]
    assert len(run.cache.kv_cache_tensors) == 2
    assert all(t.block_stride > t.layer_stride for t in run.cache.kv_cache_tensors)

    run.selector_group.create_metadata_builders(
        run.config, torch.device("cpu"), kernel_block_size=BLOCK_SIZE
    )
    raw_selector = run.views["selector"]
    assert raw_selector.shape == (2, 1, 392, 128)
    mapped = map_kv_caches_to_kernel_blocks(run.views, run.attn_groups[0])["selector"]
    stride = run.selector_group.kernel_block_stride
    assert stride == run.cache.kv_cache_tensors[0].block_stride // (
        run.selector.page_size_bytes // 2
    )
    assert mapped.shape == (stride + 2, 1, 196, 128)

    block_table = run.selector_group.map_to_kernel_block_table(
        torch.tensor([[0, 1]], dtype=torch.int32)
    )
    assert block_table.tolist() == [[0, 1, stride, stride + 1]]
    # Token 1567 ends block 0's second virtual page; 1603 lands in
    # state 8 of manager block 1 after the intervening packed layers.
    slots = compressed_qsa_slot_mapping(
        block_table,
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([1567, 1603], dtype=torch.int64),
        storage_block_size=196,
        compress_ratio=4,
    )
    assert slots.tolist() == [391, stride * 196 + 8]

    mapped[1, 0, 0, 0] = 2
    mapped[1, 0, 195, 0] = 4
    mapped[stride, 0, 8, 0] = 3
    assert raw_selector[0, 0, 196, 0] == 2
    assert raw_selector[0, 0, 391, 0] == 4
    assert raw_selector[1, 0, 8, 0] == 3
    assert torch.count_nonzero(run.views["main"]) == 0

    restored = run.selector_group.map_kv_cache_to_manager_blocks(mapped, 2)
    assert restored.data_ptr() == raw_selector.data_ptr()
    assert restored.shape == raw_selector.shape
    assert restored.stride() == raw_selector.stride()
    assert torch.equal(restored, raw_selector)


def test_dcp1_packed_selector_keeps_its_original_view() -> None:
    """Equal manager and kernel blocks must not remap the DCP=1 cache."""
    run = _packed_qsa_views(1)
    assert run.kernel_sizes == [BLOCK_SIZE]
    assert run.selector.block_size == BLOCK_SIZE
    assert run.views["selector"].shape == (2, 1, 196, 128)
    run.selector_group.create_metadata_builders(
        run.config, torch.device("cpu"), kernel_block_size=BLOCK_SIZE
    )
    assert run.selector_group.kernel_block_size is None
    mapped = map_kv_caches_to_kernel_blocks(run.views, run.attn_groups[0])
    assert mapped["selector"] is run.views["selector"]


@pytest.mark.parametrize("kind", ["uncompressed", "overlapping_manager_blocks"])
def test_packed_cache_rejects_unsafe_virtual_split(kind: str) -> None:
    if kind == "uncompressed":
        spec = _full(2 * BLOCK_SIZE)
        stride = 2 * spec.page_size_bytes
    else:
        spec = _independent_selector(2 * BLOCK_SIZE, tokens_per_state=4)
        stride = spec.page_size_bytes // 2
    cache = _single_packed_cache(spec, stride)
    with pytest.raises(ValueError, match="cannot be split"):
        allocate_kv_cache(cache, torch.device("cpu"), KVCacheLayout.BLNHC, [BLOCK_SIZE])


def _single_packed_cache(spec: FullAttentionSpec, stride: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=max(2 * stride, stride + spec.page_size_bytes),
                layers=["main"],
                layer_stride=spec.page_size_bytes,
                block_stride=stride,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["main"], spec)],
    )


def test_packed_compressed_cache_rejects_nonintegral_state_split() -> None:
    """Each manager state must belong to a kernel page."""
    spec = _independent_selector(10, tokens_per_state=3)
    cache = torch.zeros(
        (2, 1, spec.num_states, spec.state_content_size_bytes // 2),
        dtype=torch.bfloat16,
    )
    group = AttentionGroup(_QSAMainBackend, ["main"], spec, 0)
    group.kernel_block_size = 5
    with pytest.raises(ValueError, match="cannot be split"):
        group.map_kv_cache_to_kernel_blocks(cache)


class _BlockWidthBuilder:
    requires_block_table_width = True

    def __init__(self, spec, layer_names, config, device, block_table_width):
        self.block_table_width = block_table_width

    @staticmethod
    def get_cudagraph_support(config, spec):
        return AttentionCGSupport.ALWAYS


class _BlockWidthBackend:
    kernel_block_size = BLOCK_SIZE

    @staticmethod
    def get_name():
        return "BLOCK_WIDTH_TEST"

    @staticmethod
    def full_cls_name():
        return ("block_width_test", "Backend")

    @classmethod
    def get_supported_kernel_block_sizes(cls, spec=None):
        return [cls.kernel_block_size]

    @staticmethod
    def get_builder_cls():
        return _BlockWidthBuilder


def _mixed_group_with_runner_width(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    dcp_world_size: int,
    max_model_len: int,
    kernel_block_size: int,
) -> tuple[AttentionGroup, int]:
    config = _config(dcp_world_size, max_model_len=max_model_len)
    main = _full(block_size)
    selector = _independent_selector(block_size * dcp_world_size)
    groups = get_kv_cache_groups(config, {"main": main, "selector": selector})
    assert len(groups) == 1
    group_spec = groups[0].kv_cache_spec
    runner_width = get_block_table_width(
        group_spec.max_num_blocks_per_req(config, max_model_len),
        group_spec.block_size,
        token_alignment=group_spec.block_table_token_alignment,
    )

    monkeypatch.setattr(_BlockWidthBackend, "kernel_block_size", kernel_block_size)
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda: _BlockWidthBackend, num_heads=1)
        for name in ("main", "selector")
    }
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda config: {})
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda config, layer_type, layer_names: {
            name: layers[name] for name in layer_names
        },
    )
    attn_groups, _, kernel_sizes = attn_utils.init_attn_backend(
        KVCacheConfig(num_blocks=2, kv_cache_tensors=[], kv_cache_groups=groups),
        config,
        torch.device("cpu"),
        block_table_widths=[runner_width],
    )
    assert kernel_sizes == [kernel_block_size]
    selector_group = next(
        group for group in attn_groups[0] if group.layer_names == ["selector"]
    )
    return selector_group, runner_width


@pytest.mark.parametrize(
    "block_size,dcp_world_size,max_model_len,kernel_block_size",
    [(480, 4, 262144, 480), (96, 2, 1152, 48), (128, 2, 1152, 128)],
    ids=["reported_crash", "cross_row_alias", "b200_safe_shape"],
)
def test_mixed_group_mapping_preserves_request_rows(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    dcp_world_size: int,
    max_model_len: int,
    kernel_block_size: int,
) -> None:
    group, runner_width = _mixed_group_with_runner_width(
        monkeypatch, block_size, dcp_world_size, max_model_len, kernel_block_size
    )
    block_table = torch.tensor(
        [[1] * runner_width, [10] * runner_width], dtype=torch.int32
    )
    mapped = group.map_to_kernel_block_table(block_table)
    assert group.kernel_block_size is not None
    blocks_per_kv_block = group.kv_cache_spec.block_size // group.kernel_block_size
    expected = torch.tensor(
        [
            [
                blocks_per_kv_block * block + offset
                for block in row
                for offset in range(blocks_per_kv_block)
            ]
            for row in block_table.tolist()
        ],
        dtype=torch.int32,
    )
    assert torch.equal(mapped, expected)
    if block_size in (128, 480):
        layer_width = get_block_table_width(
            group.kv_cache_spec.max_num_blocks_per_req(
                _config(dcp_world_size, max_model_len=max_model_len), max_model_len
            ),
            group.kv_cache_spec.block_size,
        )
        if block_size == 128:
            assert runner_width == layer_width
        else:
            assert layer_width < runner_width
    assert group.kernel_block_table is not None
    short_width = runner_width * (blocks_per_kv_block - 1)
    group.kernel_block_table = group.kernel_block_table[..., :short_width]
    with pytest.raises(ValueError, match="kernel block table row has"):
        group.map_to_kernel_block_table(
            torch.ones((2, runner_width), dtype=torch.int32)
        )


def test_mixed_group_copy_preserves_runner_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group, runner_width = _mixed_group_with_runner_width(
        monkeypatch, 480, 4, 262144, 480
    )
    copied = replace(group, metadata_builders=[])
    assert copied.block_table_width == runner_width
    config = _config(4, max_model_len=262144)
    copied.create_metadata_builders(config, torch.device("cpu"), kernel_block_size=480)
    assert copied.kernel_block_table is not None
    assert copied.kernel_block_table.shape[-1] == runner_width * 4
