# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

aiter_available = importlib.util.find_spec("aiter") is not None
mori_available = importlib.util.find_spec("mori") is not None

if not (current_platform.is_rocm() and mori_available):
    pytest.skip(
        "MoRIIOs are only available on ROCm with mori package installed",
        allow_module_level=True,
    )

moriio_layout = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout"
)


def _full_spec(block_size: int = 4) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=3,
        dtype=torch.bfloat16,
    )


def _mla_spec(block_size: int = 4) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )


def _worker(
    kv_caches: dict[str, torch.Tensor],
    layer_to_spec: dict[str, object],
    num_blocks: int = 8,
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_caches=kv_caches,
        layer_to_spec=layer_to_spec,
        num_blocks=num_blocks,
        block_size=4,
    )


def _remote_meta(num_blocks: int = 16) -> SimpleNamespace:
    return SimpleNamespace(num_blocks=num_blocks)


def test_separated_kv_layout_uses_kv_axis_zero_and_block_axis_one():
    cache = torch.empty((2, 8, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()})

    geometry = moriio_layout.get_layer_transfer_geometry(
        "layer", cache, worker.layer_to_spec, remote_num_blocks=16
    )
    assert geometry.block_stride == 24
    assert geometry.local_kv_stride == 192
    assert geometry.remote_kv_stride == 384
    assert geometry.split_kv_regions

    assert moriio_layout.compute_block_transfer_offsets(
        "layer", cache, worker.layer_to_spec, [1, 3], [4, 5], _remote_meta().num_blocks
    ) == ([48, 144, 432, 528], [192, 240, 960, 1008], [48, 48, 48, 48])


def test_interleaved_kv_layout_uses_block_axis_zero_and_kv_axis_one():
    cache = torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()})

    geometry = moriio_layout.get_layer_transfer_geometry(
        "layer", cache, worker.layer_to_spec, remote_num_blocks=16
    )
    assert geometry.block_stride == 48
    assert geometry.local_kv_stride == 24
    assert geometry.remote_kv_stride == 24
    assert not geometry.split_kv_regions

    assert moriio_layout.compute_block_transfer_offsets(
        "layer", cache, worker.layer_to_spec, [1, 3], [4, 5], _remote_meta().num_blocks
    ) == ([96, 288], [384, 480], [96, 96])


def test_mla_key_only_layout_transfers_one_slab_per_block():
    cache = torch.empty((8, 4, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _mla_spec()})

    geometry = moriio_layout.get_layer_transfer_geometry(
        "layer", cache, worker.layer_to_spec, remote_num_blocks=16
    )
    assert geometry.block_stride == 12
    assert geometry.local_kv_stride is None
    assert geometry.remote_kv_stride is None
    assert geometry.transfers_per_block == 1

    assert moriio_layout.compute_block_transfer_offsets(
        "layer", cache, worker.layer_to_spec, [1, 3], [4, 5], _remote_meta().num_blocks
    ) == ([24, 72], [96, 120], [24, 24])


def test_mixed_layers_compute_distinct_offsets_per_layer():
    kv_caches = {
        "separated": torch.empty((2, 8, 4, 2, 3), dtype=torch.bfloat16),
        "interleaved": torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16),
        "indexer": torch.empty((8, 4, 3), dtype=torch.bfloat16),
    }
    worker = _worker(
        kv_caches,
        {
            "separated": _full_spec(),
            "interleaved": _full_spec(),
            "indexer": _mla_spec(),
        },
    )

    separated = moriio_layout.compute_block_transfer_offsets(
        "separated",
        kv_caches["separated"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
        _remote_meta().num_blocks,
    )
    interleaved = moriio_layout.compute_block_transfer_offsets(
        "interleaved",
        kv_caches["interleaved"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
        _remote_meta().num_blocks,
    )
    indexer = moriio_layout.compute_block_transfer_offsets(
        "indexer",
        kv_caches["indexer"],
        worker.layer_to_spec,
        [1, 3],
        [4, 5],
        _remote_meta().num_blocks,
    )

    assert separated != interleaved
    assert separated != indexer
    assert interleaved != indexer


def test_block_id_length_mismatch_raises_value_error():
    cache = torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()})

    with pytest.raises(ValueError, match="must have the same length"):
        moriio_layout.compute_block_transfer_offsets(
            "layer", cache, worker.layer_to_spec, [1, 3], [4], _remote_meta().num_blocks
        )


def test_registration_regions_do_not_split_interleaved_or_mla_cache():
    separated = torch.empty((2, 8, 4, 2, 3), dtype=torch.bfloat16)
    interleaved = torch.empty((8, 2, 4, 2, 3), dtype=torch.bfloat16)
    indexer = torch.empty((8, 4, 3), dtype=torch.bfloat16)
    worker = _worker(
        {
            "separated": separated,
            "interleaved": interleaved,
            "indexer": indexer,
        },
        {
            "separated": _full_spec(),
            "interleaved": _full_spec(),
            "indexer": _mla_spec(),
        },
    )

    separated_regions = moriio_layout.iter_layer_registration_regions(
        "separated", separated, worker.layer_to_spec
    )
    interleaved_regions = moriio_layout.iter_layer_registration_regions(
        "interleaved", interleaved, worker.layer_to_spec
    )
    indexer_regions = moriio_layout.iter_layer_registration_regions(
        "indexer", indexer, worker.layer_to_spec
    )

    assert [region[0].data_ptr() for region in separated_regions] == [
        separated[0].data_ptr(),
        separated[1].data_ptr(),
    ]
    assert separated_regions[0][1] == 8 * 48
    assert separated_regions[1][1] == 8 * 48

    assert len(interleaved_regions) == 1
    assert interleaved_regions[0][0].data_ptr() == interleaved.data_ptr()
    assert interleaved_regions[0][1] == 8 * 2 * 48

    assert len(indexer_regions) == 1
    assert indexer_regions[0][0].data_ptr() == indexer.data_ptr()
    assert indexer_regions[0][1] == 8 * 24


def test_registration_regions_use_layer_num_blocks():
    cache = torch.empty((4, 2, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()}, num_blocks=8)

    regions = moriio_layout.iter_layer_registration_regions(
        "layer", cache, worker.layer_to_spec
    )

    assert len(regions) == 1
    assert regions[0][1] == 4 * 2 * 48


def test_unsupported_shape_raises_value_error():
    cache = torch.empty((8, 4, 2, 3), dtype=torch.bfloat16)
    worker = _worker({"layer": cache}, {"layer": _full_spec()})

    with pytest.raises(ValueError, match="Unsupported MoRIIO K/V cache shape"):
        moriio_layout.get_layer_transfer_geometry("layer", cache, worker.layer_to_spec)
