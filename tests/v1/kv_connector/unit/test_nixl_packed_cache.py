# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for packed NIXL KV-cache registration and descriptors."""

from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

_PACKED_BASE = 0x10000
_PACKED_STRIDE = 256
_PACKED_PAGE = 128


class _TestWorker(NixlPushConnectorWorker):
    """NIXL worker shell that never starts or shuts down background state."""

    def shutdown(self) -> None:
        pass


def _new_worker() -> _TestWorker:
    worker = object.__new__(_TestWorker)
    # Keep the push override of register_kv_caches from starting a writer.
    worker._push_writer_thread = MagicMock()
    return worker


def _packed_metadata() -> NixlAgentMetadata:
    return NixlAgentMetadata(
        engine_id="remote-engine",
        agent_metadata=b"agent",
        kv_caches_base_addr=[_PACKED_BASE],
        device_id=7,
        num_blocks=2,
        block_lens=[_PACKED_STRIDE],
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        region_members=[["L0", "L1"]],
        packed_block_stride=_PACKED_STRIDE,
        packed_member_layouts={
            "L0": (0, _PACKED_PAGE),
            "L1": (_PACKED_PAGE, _PACKED_PAGE),
        },
    )


def _member_worker(
    region_members: list[list[str]],
    group_by_member: dict[str, int],
    *,
    hma: bool = True,
    packed_stride: int = 0,
    packed_layout: dict[str, tuple[int, int]] | None = None,
    specs: dict[str, Any] | None = None,
    pp_size: int = 2,
) -> _TestWorker:
    worker = _new_worker()
    worker.pp_size = pp_size
    worker._has_mamba = False
    worker._is_hma_required = hma
    worker._layer_name_to_kv_group_index = group_by_member
    worker._packed_block_stride = packed_stride
    worker._packed_layer_info = packed_layout or {}
    worker.transfer_topo = MagicMock()
    worker._set_region_members(region_members)
    worker._layer_specs = specs or {
        name: MagicMock(spec=MLAAttentionSpec)
        for members in region_members
        for name in members
    }
    return worker


@pytest.mark.parametrize("supports_member_identity", [True, False])
def test_config_declared_packed_contiguous_member_uses_packed_registration(
    supports_member_identity: bool,
):
    # A one-member packed allocation can have stride == page size. The config,
    # rather than runtime stride, must still select packed registration.
    worker = _new_worker()
    worker._supports_member_identity = supports_member_identity
    worker._has_mamba = False
    worker._register_packed_kv_cache = MagicMock()
    member = torch.empty((2, 4))
    worker.kv_cache_config = MagicMock(
        kv_cache_tensors=[
            KVCacheTensor(
                size=member.untyped_storage().nbytes(),
                shared_by=["L0"],
                block_stride=member.stride(0) * member.element_size(),
            )
        ]
    )

    worker.register_kv_caches({"L0": member})

    storage, caches = worker._register_packed_kv_cache.call_args.args
    assert storage.data_ptr() == member.untyped_storage().data_ptr()
    assert caches["L0"] is member


def test_config_declared_unpacked_padded_member_is_not_packed():
    # Runtime stride alone must not turn a padded unpacked cache into a packed
    # allocation; packing is an explicit KVCacheConfig contract.
    worker = _new_worker()
    worker.kv_cache_config = MagicMock(
        kv_cache_tensors=[KVCacheTensor(size=64, shared_by=["L0"])]
    )
    backing = torch.empty((2, 2, 4))
    padded_member = backing[:, 0]
    assert (
        padded_member.stride(0) * padded_member.element_size()
        > (padded_member.numel() // padded_member.shape[0])
        * padded_member.element_size()
    )

    assert worker._get_packed_kv_cache_storage({"L0": padded_member}) is None


def test_packed_member_descriptors():
    worker = _member_worker(
        [["L0", "L1"]],
        {"L0": 0, "L1": 0},
        hma=False,
        packed_stride=_PACKED_STRIDE,
        packed_layout={
            "L0": (0, _PACKED_PAGE),
            "L1": (_PACKED_PAGE, _PACKED_PAGE),
        },
    )
    worker.num_blocks = 2
    worker.device_id = 7

    expanded, plan = worker._expand_remote_members(_packed_metadata())

    assert plan.local_regions == (0, 0)
    assert plan.group_ids == (0, 0)
    assert plan.local_layouts == (
        (0, _PACKED_PAGE),
        (_PACKED_PAGE, _PACKED_PAGE),
    )
    assert plan.remote_block_stride == _PACKED_STRIDE
    assert expanded.kv_caches_base_addr == [
        _PACKED_BASE,
        _PACKED_BASE + _PACKED_PAGE,
    ]
    assert expanded.block_lens == [_PACKED_PAGE, _PACKED_PAGE]

    local_base = 0x20000
    assert worker._build_fa_local([local_base], 1, plan).tolist() == [
        [local_base, _PACKED_PAGE, 7],
        [local_base + _PACKED_STRIDE, _PACKED_PAGE, 7],
        [local_base + _PACKED_PAGE, _PACKED_PAGE, 7],
        [local_base + _PACKED_PAGE + _PACKED_STRIDE, _PACKED_PAGE, 7],
    ]
    assert worker._build_fa_remote(MagicMock(), expanded, 1, plan).tolist() == [
        [_PACKED_BASE, _PACKED_PAGE, 7],
        [_PACKED_BASE + _PACKED_STRIDE, _PACKED_PAGE, 7],
        [_PACKED_BASE + _PACKED_PAGE, _PACKED_PAGE, 7],
        [_PACKED_BASE + _PACKED_PAGE + _PACKED_STRIDE, _PACKED_PAGE, 7],
    ]


def test_packed_pp_stage_keeps_local_and_remote_strides():
    worker = _member_worker(
        [["L1"]],
        {"L1": 0},
        packed_stride=_PACKED_PAGE,
        packed_layout={"L1": (0, _PACKED_PAGE)},
    )
    worker.num_blocks = 2
    worker.device_id = 7

    expanded, plan = worker._expand_remote_members(_packed_metadata())

    assert expanded.kv_caches_base_addr == [_PACKED_BASE + _PACKED_PAGE]
    assert plan.local_layouts == ((0, _PACKED_PAGE),)
    assert plan.local_block_stride == _PACKED_PAGE
    assert plan.remote_block_stride == _PACKED_STRIDE

    local_base = 0x20000
    assert worker._build_fa_local([local_base], 1, plan).tolist() == [
        [local_base, _PACKED_PAGE, 7],
        [local_base + _PACKED_PAGE, _PACKED_PAGE, 7],
    ]
    assert worker._build_fa_remote(MagicMock(), expanded, 1, plan).tolist() == [
        [_PACKED_BASE + _PACKED_PAGE, _PACKED_PAGE, 7],
        [_PACKED_BASE + _PACKED_PAGE + _PACKED_STRIDE, _PACKED_PAGE, 7],
    ]


def _fake_packed_cache(
    *,
    base_ptr: int,
    offset_bytes: int,
    page_bytes: int,
    block_stride_bytes: int,
    num_blocks: int,
    elt: int = 4,
) -> MagicMock:
    cache = MagicMock()
    cache.shape = (num_blocks, page_bytes // elt)
    cache.stride.return_value = block_stride_bytes // elt
    cache.element_size.return_value = elt
    cache.numel.return_value = num_blocks * (page_bytes // elt)
    cache.data_ptr.return_value = base_ptr + offset_bytes
    cache.get_device.return_value = 0
    return cache


def _packed_registration_worker(
    specs: dict[str, Any],
) -> _TestWorker:
    worker = _new_worker()
    worker._has_mamba = False
    worker.tp_rank = 0
    worker.world_size = 1
    worker.engine_id = "test-engine"
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_agent_metadata.return_value = b"agent"
    worker.register_local_xfer_handler = MagicMock(return_value=(0, []))
    worker.num_blocks = 2
    worker.block_size = 16
    worker.use_mla = False
    worker.backend_name = "FLASH_ATTN"
    worker.kv_cache_layout = "HND"
    worker.nixl_memory_type = "VRAM"
    worker.nixl_backends = ["UCX"]
    worker._mamba_ssm_size = (0, 0)
    worker._physical_blocks_per_logical_kv_block = 1
    worker._registered_descs = []
    worker.src_xfer_handles_by_block_size = {}
    worker.kv_caches_base_addr = defaultdict(dict)
    worker.dst_num_blocks = {}
    worker.model_config = MagicMock()
    worker.attn_backends = [MagicMock()]
    worker.vllm_config = MagicMock()
    worker._layer_specs = specs
    return worker


def test_packed_registration_accepts_non_mla_whole_region():
    # Whole-region packed registration is spec-agnostic; only PP member-major
    # slicing is restricted to MLA.
    worker = _packed_registration_worker(
        {
            "L0": MagicMock(spec=FullAttentionSpec),
            "L1": MagicMock(spec=FullAttentionSpec),
        }
    )
    storage = MagicMock()
    storage.nbytes.return_value = 64
    storage.data_ptr.return_value = _PACKED_BASE
    storage.device.index = 0
    caches = {
        "L0": _fake_packed_cache(
            base_ptr=_PACKED_BASE,
            offset_bytes=0,
            page_bytes=16,
            block_stride_bytes=32,
            num_blocks=2,
        ),
        "L1": _fake_packed_cache(
            base_ptr=_PACKED_BASE,
            offset_bytes=16,
            page_bytes=16,
            block_stride_bytes=32,
            num_blocks=2,
        ),
    }

    module = "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker"
    with (
        patch(f"{module}.TransferTopology") as topo_cls,
        patch(f"{module}.compute_nixl_compatibility_hash", return_value="hash"),
    ):
        topo_cls.return_value = MagicMock(cross_layers_blocks=True)
        worker._register_packed_kv_cache(storage, caches)

    assert worker.num_regions == 1
    assert worker.block_len_per_layer == [32]
    assert worker._packed_layer_info == {"L0": (0, 16), "L1": (16, 16)}
    assert worker.region_members == [["L0", "L1"]]


def test_packed_member_routing_rejects_non_mla():
    worker = _member_worker(
        [["L0", "L1"]],
        {"L0": 0, "L1": 0},
        hma=False,
        packed_stride=_PACKED_STRIDE,
        packed_layout={
            "L0": (0, _PACKED_PAGE),
            "L1": (_PACKED_PAGE, _PACKED_PAGE),
        },
        specs={
            "L0": MagicMock(spec=FullAttentionSpec),
            "L1": MagicMock(spec=FullAttentionSpec),
        },
    )

    with pytest.raises(NotImplementedError, match="only MLA"):
        worker._expand_remote_members(_packed_metadata())


def test_pp1_packed_push_preserves_whole_region_route_for_non_mla():
    worker = _member_worker(
        [["L0", "L1"]],
        {"L0": 0, "L1": 0},
        hma=False,
        packed_stride=_PACKED_STRIDE,
        packed_layout={
            "L0": (0, _PACKED_PAGE),
            "L1": (_PACKED_PAGE, _PACKED_PAGE),
        },
        specs={
            "L0": MagicMock(spec=FullAttentionSpec),
            "L1": MagicMock(spec=FullAttentionSpec),
        },
        pp_size=1,
    )

    assert not worker._use_member_identity(_packed_metadata())
