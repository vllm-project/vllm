# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

import pytest
import torch

from vllm.config import DeviceConfig, KVTransferConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheAxis as Axis
from vllm.v1.kv_cache_layout import KVCacheLayout, KVCacheTensorLayout
from vllm.v1.worker.utils import (
    add_kv_sharing_layers_to_kv_cache_groups,
    allocate_kv_cache,
)

pytestmark = pytest.mark.cpu_test


class TensorRegistrationConnector(ExampleConnector):
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.registered_caches = kv_caches


class AxisRegistrationConnector(TensorRegistrationConnector):
    def register_kv_caches_with_layouts(
        self,
        kv_caches: dict[str, torch.Tensor],
        layouts: Mapping[str, KVCacheTensorLayout],
    ) -> None:
        super().register_kv_caches_with_layouts(kv_caches, layouts)
        self.layouts = layouts


@pytest.fixture
def make_connector():
    def make(kv_cache_config: KVCacheConfig) -> MultiConnector:
        config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=SchedulerConfig(
                max_model_len=16,
                max_num_batched_tokens=16,
                max_num_seqs=1,
                is_encoder_decoder=False,
                disable_hybrid_kv_cache_manager=True,
            ),
            kv_transfer_config=KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "connectors": [
                        {
                            "kv_connector": cls.__name__,
                            "kv_connector_module_path": __name__,
                            "kv_role": "kv_both",
                        }
                        for cls in (
                            TensorRegistrationConnector,
                            AxisRegistrationConnector,
                        )
                    ]
                },
            ),
        )
        return MultiConnector(config, KVConnectorRole.WORKER, kv_cache_config)

    return make


@pytest.mark.parametrize(
    ("axes", "expected"),
    [
        ((Axis.BLOCK, Axis.KV), [0, 1, 4, 5, 2, 3, 6, 7]),
        ((Axis.KV, Axis.BLOCK), list(range(8))),
    ],
    ids=["blocks-first", "kv-first"],
)
def test_two_blocks_keep_kv_semantics_through_mixed_connectors(
    make_connector, axes, expected
):
    """Equal shapes must still give the consumer the producer's K/V values."""
    connector = make_connector(KVCacheConfig(2, [], []))
    layout = KVCacheTensorLayout(
        (*axes, Axis.HEAD, Axis.SINGLETON, Axis.STATE, Axis.HEAD_DIM)
    )
    order = (Axis.KV, Axis.BLOCK, Axis.STATE, Axis.HEAD, Axis.HEAD_DIM, Axis.SINGLETON)
    for offset in (0, 100):
        caches = {"layer": (torch.arange(8) + offset).reshape(2, 2, 1, 1, 2, 1)}
        connector.register_kv_caches_with_layouts(caches, {"layer": layout})
        legacy, consumer = connector.sub_connectors
        assert isinstance(legacy, TensorRegistrationConnector)
        assert isinstance(consumer, AxisRegistrationConnector)
        assert legacy.registered_caches is consumer.registered_caches is caches
        canonical = consumer.registered_caches["layer"].permute(
            *(consumer.layouts["layer"].axis_index(axis) for axis in order)
        )
        assert canonical.flatten().tolist() == [value + offset for value in expected]


@pytest.mark.parametrize(
    ("physical_layout", "padded"),
    [(layout, False) for layout in KVCacheLayout]
    + [(KVCacheLayout.LBHNC, True), (KVCacheLayout.BLHNC, True)],
)
def test_allocator_views_and_aliases_keep_their_values(
    make_connector, physical_layout, padded
):
    """Registration describes indexing order while preserving real strides."""
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=144 if padded else None,
    )
    nb, nl = 2, 2
    strides = compute_layout_strides(spec, nb, nl, physical_layout)
    config = KVCacheConfig(
        num_blocks=nb,
        kv_cache_tensors=[
            KVCacheTensor(
                size=nb * nl * spec.page_size_bytes,
                layers=["first", "second"],
                layer_stride=strides[0],
                block_stride=strides[1],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["first", "second"],
                UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"first": spec, "second": spec}
                ),
            )
        ],
    )
    caches = allocate_kv_cache(config, torch.device("cpu"), physical_layout)
    caches["first"].copy_(torch.arange(64).reshape(2, 2, 4, 4))
    caches["second"].fill_(-1)
    add_kv_sharing_layers_to_kv_cache_groups({"alias": "first"}, config.kv_cache_groups)
    caches["alias"] = caches["first"]
    connector = make_connector(config)
    connector.register_kv_caches_with_layouts(caches, config.get_tensor_layouts(caches))
    consumer = connector.sub_connectors[1]
    assert isinstance(consumer, AxisRegistrationConnector)
    assert consumer.registered_caches is caches
    assert set(consumer.layouts) == set(caches)
    assert consumer.layouts["alias"] == consumer.layouts["first"]
    layout = consumer.layouts["first"]
    index = tuple(
        {Axis.BLOCK: 1, Axis.HEAD: 0, Axis.STATE: 2, Axis.CONTENT: 3}[axis]
        for axis in layout.axes
    )
    assert consumer.registered_caches["alias"][index].item() == 43
    assert torch.all(caches["second"] == -1)
    assert caches["first"].stride(0) * caches["first"].element_size() == strides[1]


@pytest.mark.parametrize(
    ("shape", "message"),
    [((3, 2, 1), "rank"), ((3, 3, 1, 4), "KV axis"), ((3, 2, 2, 4), "SINGLETON")],
)
def test_invalid_descriptor_is_rejected_before_legacy_registration(
    make_connector, shape, message
):
    connector = make_connector(KVCacheConfig(3, [], []))
    layout = KVCacheTensorLayout((Axis.BLOCK, Axis.KV, Axis.SINGLETON, Axis.HEAD_DIM))
    with pytest.raises(ValueError, match=message):
        connector.register_kv_caches_with_layouts(
            {"layer": torch.empty(shape)}, {"layer": layout}
        )
    assert all(
        not hasattr(child, "registered_caches") for child in connector.sub_connectors
    )


def test_unknown_layer_descriptor_is_rejected(make_connector):
    connector = make_connector(KVCacheConfig(2, [], []))
    with pytest.raises(ValueError, match="unregistered layer"):
        connector.register_kv_caches_with_layouts(
            {}, {"missing": KVCacheTensorLayout((Axis.BLOCK,))}
        )


def test_rejects_duplicate_semantic_axes():
    with pytest.raises(ValueError, match="unique"):
        KVCacheTensorLayout((Axis.BLOCK, Axis.BLOCK))


def test_missing_descriptors_stay_unspecified(make_connector):
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["raw"], HiSparseResidentSpec(block_size=4, page_size=128))
        ],
    )
    caches = {"raw": torch.empty(256, dtype=torch.uint8), "custom": torch.empty(2)}
    connector = make_connector(config)
    layouts = config.get_tensor_layouts(caches)
    assert layouts == {}
    connector.register_kv_caches_with_layouts(caches, layouts)
    consumer = connector.sub_connectors[1]
    assert isinstance(consumer, AxisRegistrationConnector)
    assert consumer.registered_caches is caches
    assert consumer.layouts == {}
