# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end NIXL descriptor geometry invariants for hybrid MLA+SSM models
under heterogeneous P/D block geometry (TP-sharded KDA-style state, so
the mamba-aligned logical block size differs between P and D while the
kernel-granularity pages stay equal).

The invariant under test: every LOCAL byte range a request's READ transfers
into must lie within that request's own blocks. A violation means an
incoming transfer can overwrite a co-resident request's KV or mamba state
mid-decode (silent corruption of an unrelated request).
"""

from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from .utils import create_vllm_config


class _RecordingNixl:
    """Minimal NIXL wrapper stand-in that records descriptor lists and
    prepared transfers so tests can resolve desc ids to byte ranges."""

    def __init__(self, *args, **kwargs):
        self.dlists: dict[int, np.ndarray] = {}
        self.xfers: list[tuple] = []
        self.registered: list[tuple[list[tuple], object]] = []
        self._next_handle = 1

    def get_reg_descs(self, caches_data, mem_type):
        return caches_data

    def register_memory(self, descs, backends=None):
        self.registered.append((descs, backends))

    def deregister_memory(self, descs):
        pass

    def get_agent_metadata(self):
        return b"agent-meta"

    def get_xfer_descs(self, blocks_data, mem_type):
        return blocks_data

    def prep_xfer_dlist(self, agent, descs):
        handle = self._next_handle
        self._next_handle += 1
        self.dlists[handle] = np.asarray(descs, dtype=np.uint64).reshape(-1, 3)
        return handle

    def add_remote_agent(self, metadata):
        return "remote-agent"

    def make_prepped_xfer(
        self, op, local_handle, local_ids, remote_handle, remote_ids, notif_msg=None
    ):
        handle = self._next_handle
        self._next_handle += 1
        self.xfers.append(
            (
                op,
                local_handle,
                np.asarray(local_ids),
                remote_handle,
                np.asarray(remote_ids),
            )
        )
        return handle

    def transfer(self, handle):
        pass

    def check_xfer_state(self, handle):
        return "DONE"

    def get_xfer_telemetry(self, handle):
        from types import SimpleNamespace

        return SimpleNamespace(
            xferDuration=1.0, postDuration=1.0, totalBytes=1, descCount=1
        )

    def release_xfer_handle(self, handle):
        pass

    def release_dlist_handle(self, handle):
        pass

    def send_notif(self, agent, notif_msg=None):
        pass

    def get_new_notifs(self):
        return {}

    def remove_remote_agent(self, agent):
        pass


@pytest.mark.cpu_test
def test_local_descriptors_follow_each_region_pool_capacity():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    worker.transfer_topo = MagicMock()
    worker.device_id = 0
    worker.block_len_per_layer = [16, 16]
    worker.region_strides = [16, 16]
    worker.region_num_blocks = [2, 3]

    descriptors = worker._build_fa_local([100, 1000], block_size_ratio=1)

    assert descriptors[:, 0].tolist() == [100, 116, 1000, 1016, 1032]


@pytest.mark.cpu_test
def test_overlaid_transfer_groups_share_region_geometry():
    """Groups overlaid on one allocation share its transfer region."""
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker as bw
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MLAAttentionSpec,
    )

    num_blocks = 4
    spec = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.uint8,
    )
    page_size = spec.page_size_bytes
    block_stride = 2 * page_size
    backing = torch.zeros(num_blocks, block_stride, dtype=torch.uint8)
    caches = {
        "layer.0": backing[:, :page_size],
        "layer.1": backing[:, :page_size],
    }
    groups = [KVCacheGroupSpec([layer_name], spec) for layer_name in caches]

    worker = object.__new__(NixlConnectorWorker)
    worker.tp_rank = 0
    worker.world_size = 1
    worker.block_size = 4
    worker.engine_id = "local-engine"
    worker.use_mla = True
    worker.model_config = MagicMock()
    worker.model_config.get_total_num_kv_heads.return_value = 1
    worker.attn_backends = []
    worker._has_mamba = False
    worker.vllm_config = MagicMock()
    worker.backend_name = "FLASHMLA"
    worker.num_blocks = num_blocks
    worker.nixl_memory_type = "VRAM"
    worker.nixl_backends = None
    worker.nixl_wrapper = _RecordingNixl()
    worker._registered_descs = []
    worker.dst_num_blocks = {}
    worker.dst_region_num_blocks = {}
    worker.dst_region_group_ids = {}
    worker.dst_region_block_sizes = {}
    worker.dst_region_split_ratios = {}
    worker.src_xfer_handles_by_block_size = {}
    worker.kv_caches_base_addr = defaultdict(dict)
    worker._mamba_ssm_size = (0, 0)
    worker.kv_cache_layout = "NHD"
    worker.host_buffer_kv_cache_layout = "NHD"
    worker._physical_blocks_per_logical_kv_block = 1
    worker._logical_num_blocks = num_blocks
    worker.region_mem_types = []
    worker.region_strides = []
    worker.region_group_ids = []
    worker.region_block_sizes = []
    worker.region_names = []
    worker.region_num_blocks = []
    worker._mixed_mem_types = False
    worker._desc_is_dram_by_block_size = {}
    worker._desc_pos_by_block_size = {}
    worker._dram_src_handles_by_block_size = {}
    worker._region_is_mla = []
    worker.block_len_per_layer = []
    worker.block_stride_per_layer = []
    worker.device_id = 0
    worker.use_host_buffer = False
    worker.host_xfer_buffers = {}
    worker.device_kv_caches = {}
    worker.pp_size = 1
    worker.dcp_size = 1
    worker.pcp_size = 1
    worker.kv_buffer_device = "cuda"
    worker._layer_specs = {name: spec for name in caches}
    worker._nixl_adapter = None
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing.nbytes,
                layers=[name],
                layer_stride=page_size,
                block_stride=block_stride,
            )
            for name in caches
        ],
        kv_cache_groups=groups,
        num_blocks_by_pool=[num_blocks],
    )

    transfer_topology = MagicMock()

    with (
        patch.object(bw, "TransferTopology", return_value=transfer_topology),
        patch.object(bw, "compute_nixl_compatibility_hash", return_value="hash"),
    ):
        worker.register_kv_caches(caches)

    assert worker.region_group_ids == [-1]
    assert worker.region_strides == [block_stride]
    assert worker.nixl_wrapper.registered[0][0] == [
        (backing.data_ptr(), backing.nbytes, 0, "")
    ]
    expected_addrs = [
        backing.data_ptr() + block * block_stride for block in range(num_blocks)
    ]
    assert worker.src_blocks_data[:, 0].tolist() == expected_addrs

    metadata = msgspec.msgpack.decode(
        worker.xfer_handshake_metadata.agent_metadata_bytes,
        type=NixlAgentMetadata,
    )
    assert metadata.region_group_ids == [-1]
    assert metadata.region_num_blocks == [num_blocks]
    assert worker._block_ids_by_region(([0], [2]), worker.region_group_ids) == [[0, 2]]


def _make_mla_hybrid_worker(local_block_size, kernel_block_size, num_logical_blocks):
    """Build a real pull worker with a hybrid MLA + 2xKDA HMA layout."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        base_worker as bw,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MambaSpec,
        MLAAttentionSpec,
    )

    mla_spec = MLAAttentionSpec(
        block_size=local_block_size,
        num_kv_heads=1,
        head_size=6,
        dtype=torch.float16,
    )
    unified_page = mla_spec.page_size_bytes
    kda_spec = MambaSpec(
        block_size=local_block_size,
        shapes=((8, 3), (1, 4, 4)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=unified_page,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    # The three groups overlay each other, so layer i of every group aliases the same
    # region: mla.i, kda_a.i and kda_b.i all live at i * layer_stride.
    layer_stride = num_logical_blocks * unified_page
    kv_cache_config = KVCacheConfig(
        num_blocks=num_logical_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * layer_stride,
                layers=[f"{prefix}.0", f"{prefix}.1"],
                layer_stride=layer_stride,
                block_stride=unified_page,
            )
            for prefix in ("mla", "kda_a", "kda_b")
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["mla.0", "mla.1"], mla_spec),
            KVCacheGroupSpec(["kda_a.0", "kda_a.1"], kda_spec),
            KVCacheGroupSpec(["kda_b.0", "kda_b.1"], kda_spec),
        ],
    )

    vllm_config = create_vllm_config(block_size=local_block_size)
    vllm_config.cache_config.enable_prefix_caching = False
    # kv_buffer_device defaults to the *real* platform's device type, which on
    # a CPU-only test host would make this a host-buffer worker: host xfer
    # buffers are per-layer, so the HMA shared-tensor regions this test builds
    # would not be deduplicated. Pin it to the faked device type.
    vllm_config.kv_transfer_config.kv_buffer_device = "cuda"

    fake_backend = MagicMock()
    fake_backend.get_supported_kernel_block_sizes.return_value = [kernel_block_size]
    fake_backend.get_name.return_value = "FLASHMLA"
    fake_backend.full_cls_name.return_value = "fake.FLASHMLA"
    fake_platform = MagicMock()
    fake_platform.device_type = "cuda"
    fake_platform.get_nixl_memory_type.return_value = "VRAM"

    from vllm.config import set_current_vllm_config

    with (
        patch.object(bw, "NixlWrapper", _RecordingNixl),
        patch.object(bw, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(bw, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(bw, "get_current_attn_backends", return_value=[fake_backend]),
        patch.object(bw, "current_platform", fake_platform),
        patch(
            "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
            return_value="DS",
        ),
        set_current_vllm_config(vllm_config),
    ):
        worker = NixlConnectorWorker(vllm_config, "local-engine", kv_cache_config)
        worker.use_mla = True

        # Attention caches are kernel-block granular on dim 0, as the
        # receive post-process assumes.
        ppl = local_block_size // kernel_block_size
        tensors = [
            torch.zeros(
                num_logical_blocks * ppl, unified_page // ppl, dtype=torch.uint8
            )
            for _ in range(2)
        ]
        worker.register_kv_caches(
            {
                "kda_a.0": tensors[0],
                "mla.0": tensors[0],
                "kda_b.0": tensors[0],
                "kda_a.1": tensors[1],
                "mla.1": tensors[1],
                "kda_b.1": tensors[1],
            }
        )
    # Keep tensors alive alongside the worker; flat views for byte checks.
    worker._test_tensors = [t.view(-1) for t in tensors]
    worker._test_tensors_2d = tensors
    worker._test_unified_page = unified_page
    return worker


def _make_remote_meta(
    worker,
    remote_block_size,
    remote_kernel_block_size,
    remote_num_logical,
    remote_ssm_sizes,
):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )

    remote_ppl = remote_block_size // remote_kernel_block_size
    # Kernel-granularity pages are TP-independent for MLA hybrids and must
    # match the local ones for the handshake to pass, scaled down by the
    # block-size ratio when the remote's kernel block is smaller.
    block_size_ratio = worker.block_size // remote_kernel_block_size
    kernel_page = worker.block_len_per_layer[0] // block_size_ratio
    return NixlAgentMetadata(
        engine_id="remote-engine",
        agent_metadata=b"remote-agent-meta",
        device_id=0,
        kv_caches_base_addr=[0x10_000_000, 0x20_000_000],
        num_blocks=remote_num_logical * remote_ppl,
        block_lens=[kernel_page, kernel_page],
        # Non-interleaved remote: consecutive blocks abut, so stride == page length.
        block_strides=[kernel_page, kernel_page],
        kv_cache_layout=worker.kv_cache_layout,
        block_size=remote_kernel_block_size,
        ssm_sizes=remote_ssm_sizes,
        attn_backend_name=worker.backend_name,
        physical_blocks_per_logical_kv_block=remote_ppl,
    )


def _register_remote_agents(worker, metadata, tp_size):
    """Mirror the async handshake callback that publishes prepared agents."""
    worker._remote_agents[metadata.engine_id] = {
        (0, rank): worker.add_remote_agent(
            metadata, remote_tp_rank=rank, remote_tp_size=tp_size
        )
        for rank in range(tp_size)
    }


def _owned_byte_ranges(worker, group_logical_ids):
    """Byte ranges owned by a request: for each HMA region tensor, every
    logical block id of every group maps to one unified page."""
    unified_page = worker._test_unified_page
    bases = [t.data_ptr() for t in worker._test_tensors]
    owned = []
    for base in bases:
        for ids in group_logical_ids:
            for b in ids:
                owned.append((base + b * unified_page, base + (b + 1) * unified_page))
    return owned


def _assert_local_writes_within(worker, owned_ranges):
    nixl = worker.nixl_wrapper
    assert nixl.xfers, "no transfers were posted"
    violations = []
    total_descs = 0
    for op, local_handle, local_ids, _, remote_ids in nixl.xfers:
        assert len(local_ids) == len(remote_ids)
        desc_arr = nixl.dlists[local_handle]
        for i in local_ids:
            addr, length, _dev = desc_arr[int(i)]
            addr, length = int(addr), int(length)
            total_descs += 1
            if not any(lo <= addr and addr + length <= hi for lo, hi in owned_ranges):
                violations.append((int(i), hex(addr), length))
    assert not violations, (
        f"{len(violations)}/{total_descs} local descriptors write outside "
        f"the request's own blocks: {violations[:10]}"
    )
    return total_descs


@pytest.mark.cpu_test
def test_hetero_ppl_multi_read_writes_stay_within_request_blocks():
    """MLA-hybrid hetero geometry: local (D, TP1) logical blocks of 12 tokens
    (kernel 4, ppl=3) vs remote (P, TP2) logical blocks of 8 tokens (ppl=2),
    equal kernel pages, tp_ratio=-2 multi-read with replicated MLA and
    TP-sharded KDA state. Every local descriptor of the request's reads must
    stay within its own blocks."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    worker = _make_mla_hybrid_worker(
        local_block_size=12, kernel_block_size=4, num_logical_blocks=8
    )
    assert worker._physical_blocks_per_logical_kv_block == 3

    meta_r = _make_remote_meta(
        worker,
        remote_block_size=8,
        remote_kernel_block_size=4,
        remote_num_logical=12,
        remote_ssm_sizes=(24, 32),
    )
    _register_remote_agents(worker, meta_r, 2)

    # Request B: 17 matched tokens. Local: 2 logical blocks (24 tok
    # capacity); remote: 16 prefilled tokens -> 2 remote logical blocks.
    # Sparse, non-contiguous ids so neighbor blocks exist on all sides.
    local_ids = ([2, 5], [1], [7])
    remote_ids = [[1, 4], [5], [2]]

    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="req-b",
        local_block_ids=local_ids,
        kv_transfer_params={
            "remote_block_ids": remote_ids,
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-req-b",
            "remote_host": "remote-host",
            "remote_port": 1234,
            "tp_size": 2,
        },
    )
    meta = metadata.reqs_to_recv["req-b"]
    meta.local_physical_block_ids = worker._logical_to_kernel_block_ids(
        meta.local_block_ids, worker._physical_blocks_per_logical_kv_block
    )
    worker._recving_metadata["req-b"] = meta

    worker._read_blocks_for_req("req-b", meta)

    owned = _owned_byte_ranges(worker, local_ids)
    total = _assert_local_writes_within(worker, owned)
    # Multi-read: rank 0 carries the replicated MLA + its SSM shard,
    # rank 1 carries only its SSM shard.
    assert len(worker.nixl_wrapper.xfers) == 2
    assert total > 0


def _resolve(
    desc_arr,
    idx,
    bases,
    region_size,
    unified_page,
    desc_page,
    logical_ids_attn,
    block_tokens,
):
    """Resolve a desc id to (region, kind, token_start) where kind is 'attn'
    (desc-page sized, sub-block-aligned, in the request's attention blocks)
    or 'mamba'. token_start is the request-relative token offset, so local
    and remote are comparable even when their kernel blocks differ in size."""
    addr, length, _ = (int(x) for x in desc_arr[int(idx)])
    for region, base in enumerate(bases):
        off = addr - base
        if 0 <= off < region_size:
            b = off // unified_page
            rem = off % unified_page
            if length == desc_page and rem % desc_page == 0 and b in logical_ids_attn:
                pos = logical_ids_attn.index(b)
                tokens_per_desc = block_tokens * desc_page // unified_page
                sub = rem // desc_page
                return (region, "attn", pos * block_tokens + sub * tokens_per_desc)
            return (region, "mamba", None)
    raise AssertionError(f"desc {idx} addr {addr:#x} not in any region")


def _run_hetero_case(
    local_block, kernel, remote_block, num_tokens, tp_size=2, remote_kernel=None
):
    """Full pull-path run for one geometry; returns pairing records.

    ``remote_kernel`` defaults to the local kernel block size; a smaller
    value additionally exercises block_size_ratio > 1.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    remote_kernel = remote_kernel or kernel
    block_size_ratio = kernel // remote_kernel
    remote_ppl = remote_block // remote_kernel
    matched = num_tokens - 1  # mamba N-1 rule
    n_local = -(-num_tokens // local_block)
    n_remote = -(-matched // remote_block)

    worker = _make_mla_hybrid_worker(
        local_block_size=local_block,
        kernel_block_size=kernel,
        num_logical_blocks=max(2 * n_local + 4, 8),
    )
    # Local KDA state pages are (48, 64) bytes; the remote holds 1/tp_size
    # shards of each.
    meta_r = _make_remote_meta(
        worker,
        remote_block_size=remote_block,
        remote_kernel_block_size=remote_kernel,
        remote_num_logical=max(2 * n_remote + 4, 8),
        remote_ssm_sizes=(48 // tp_size, 64 // tp_size),
    )
    _register_remote_agents(worker, meta_r, tp_size)

    # Sparse ids so neighbors exist between the request's blocks.
    local_attn = [2 * i + 1 for i in range(n_local)]
    remote_attn = [2 * i + 2 for i in range(n_remote)]
    local_ids = (local_attn, [0], [2 * n_local + 2])
    remote_ids = [remote_attn, [1], [0]]

    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="req-b",
        local_block_ids=local_ids,
        kv_transfer_params={
            "remote_block_ids": remote_ids,
            "remote_engine_id": "remote-engine",
            "remote_request_id": "prefill-req-b",
            "remote_host": "remote-host",
            "remote_port": 1234,
            "tp_size": tp_size,
        },
    )
    meta = metadata.reqs_to_recv["req-b"]
    meta.local_physical_block_ids = worker._logical_to_kernel_block_ids(
        meta.local_block_ids, worker._physical_blocks_per_logical_kv_block
    )
    worker._recving_metadata["req-b"] = meta

    # Sentinel-fill the local KV so untouched bytes are detectable.
    for t in worker._test_tensors:
        t.fill_(0xAA)

    worker._read_blocks_for_req("req-b", meta)

    # Invariant 1: all local writes within the request's own blocks.
    owned = _owned_byte_ranges(worker, local_ids)
    _assert_local_writes_within(worker, owned)

    # Invariant 2: local<->remote attention pairs are token-aligned.
    nixl = worker.nixl_wrapper
    local_bases = [t.data_ptr() for t in worker._test_tensors]
    remote_bases = [0x10_000_000, 0x20_000_000]
    local_unified = worker._test_unified_page
    remote_unified = (local_unified // local_block) * remote_block
    # With block_size_ratio > 1 the local page is split into ratio sub-descs,
    # each the size of a whole remote kernel page.
    desc_page = worker.block_len_per_layer[0] // block_size_ratio
    meta_r_num_blocks_bytes = (meta_r.num_blocks // remote_ppl) * remote_unified
    covered_tokens = set()
    for op, lh, lids, rh, rids in nixl.xfers:
        larr, rarr = nixl.dlists[lh], nixl.dlists[rh]
        for li, ri in zip(lids, rids):
            lreg, lkind, ltok = _resolve(
                larr,
                li,
                local_bases,
                len(worker._test_tensors[0]),
                local_unified,
                desc_page,
                local_attn,
                local_block,
            )
            rreg, rkind, rtok = _resolve(
                rarr,
                ri,
                remote_bases,
                meta_r_num_blocks_bytes,
                remote_unified,
                desc_page,
                remote_attn,
                remote_block,
            )
            assert lkind == rkind, (
                f"pair kind mismatch: local {lkind} vs remote {rkind} "
                f"(local desc {li}, remote desc {ri})"
            )
            assert lreg == rreg, (
                f"region mismatch: local {lreg} vs remote {rreg} for "
                f"tokens {ltok} vs {rtok}"
            )
            if lkind == "attn":
                assert ltok == rtok, (
                    f"TOKEN MISALIGNMENT: local sub-block holds tokens "
                    f"[{ltok}..) but receives remote tokens [{rtok}..) "
                    f"(geometry local_block={local_block}, "
                    f"remote_block={remote_block}, N={num_tokens})"
                )
                covered_tokens.add(ltok)

    # Invariant 3: full coverage of the matched tokens, at the finest
    # transfer granularity (the remote kernel block).
    needed = {t for t in range(0, matched - matched % remote_kernel, remote_kernel)}
    missing = needed - covered_tokens
    assert not missing, (
        f"tokens never transferred: {sorted(missing)[:8]} "
        f"(geometry local_block={local_block}, remote_block={remote_block}, "
        f"N={num_tokens}, matched={matched})"
    )

    # Invariant 4: no stale bytes after receive completion. The scheduler
    # excludes the blocks covering the matched tokens from alloc-time KV
    # zeroing (the zeroing would race the RDMA write), so every byte of
    # those blocks must be either written by the transfer or zeroed by the
    # receive post-process. Stale bytes surface as mid-response garbage
    # once decode grows into the untransferred tail.
    for op, lh, lids, rh, rids in nixl.xfers:
        larr = nixl.dlists[lh]
        for li in lids:
            addr, length, _ = (int(x) for x in larr[int(li)])
            for t in worker._test_tensors:
                off = addr - t.data_ptr()
                if 0 <= off < t.numel():
                    t[off : off + length] = 0  # simulate the RDMA write
                    break
    done_sending, done_recving = worker.get_finished()
    assert "req-b" in done_recving
    n_excluded = -(-matched // local_block)
    stale = []
    for b in local_attn[:n_excluded]:
        for region, t in enumerate(worker._test_tensors):
            page = t[b * local_unified : (b + 1) * local_unified]
            n_stale = int((page == 0xAA).sum())
            if n_stale:
                stale.append((region, b, n_stale))
    assert not stale, (
        f"stale (unzeroed, untransferred) bytes in matched-range attention "
        f"blocks (region, block, bytes): {stale} "
        f"(geometry local_block={local_block}, remote_block={remote_block}, "
        f"N={num_tokens}, matched={matched})"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "local_block,remote_block",
    [
        (12, 8),  # ppl 3 vs 2
        (36, 8),  # ppl 9 vs 2 (large ppl asymmetry, scaled)
        (24, 4),  # ppl 6 vs 1
        (16, 24),  # remote larger than local (D_TP > P_TP direction)
    ],
)
@pytest.mark.parametrize("num_tokens", list(range(2, 40)))
def test_hetero_ppl_token_alignment_sweep(local_block, remote_block, num_tokens):
    """Sweep prompt lengths across block-boundary residues for several
    hetero-ppl geometries; assert neighbor-safety, token alignment, and
    coverage of every transferred kernel block."""
    _run_hetero_case(
        local_block, kernel=4, remote_block=remote_block, num_tokens=num_tokens
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "num_tokens",
    # Residues around the remote kernel block (4), the local kernel block
    # (8), the remote logical block (8) and the local logical block (24).
    [2, 5, 8, 9, 13, 16, 17, 21, 24, 25, 29, 32, 33, 41, 48, 49],
)
def test_hetero_ppl_with_block_size_ratio(num_tokens):
    """Both hetero regimes at once: kernel blocks differ (local 8 / remote
    4, block_size_ratio=2) *and* physical_blocks_per_logical differs (3 vs
    2). The transfer is clipped at remote sub-block granularity by the
    pairing and front-trimmed by _apply_prefix_caching, so the
    untransferred tail can span both a partial block and whole blocks —
    the case each of the two former zeroing paths handled only half of."""
    _run_hetero_case(
        local_block=24,
        kernel=8,
        remote_block=8,
        remote_kernel=4,
        num_tokens=num_tokens,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "num_tokens",
    # Residues around every geometric boundary: kernel block (64), remote
    # logical block (768), local logical block (5760), plus odd offsets.
    [
        2,
        63,
        64,
        65,
        127,
        128,
        300,
        640,
        767,
        768,
        769,
        831,
        832,
        1000,
        1535,
        1536,
        1537,
        2303,
        2304,
        2305,
        3001,
        5759,
        5760,
        5761,
        5824,
        6528,
        6529,
    ],
)
def test_mla_hybrid_large_ppl_geometry(num_tokens):
    """KimiLinear-scale MLA-hybrid geometry (TP8 prefill -> TP1 decode):
    decode (local) logical block 5760 / kernel 64 (ppl=90), prefill
    (remote) logical block 768 (ppl=12), tp_ratio=-8 multi-read with
    replicated MLA and 8-way TP-sharded KDA state."""
    _run_hetero_case(
        local_block=5760,
        kernel=64,
        remote_block=768,
        num_tokens=num_tokens,
        tp_size=8,
    )


@pytest.mark.cpu_test
def test_mismatched_mla_kernel_page_rejected_for_mla_hybrid():
    """The MLA per-token page is TP-independent, so kernel block lengths
    differing by anything other than the block-size ratio must fail the
    handshake loudly rather than transfer at mismatched geometry."""
    worker = _make_mla_hybrid_worker(
        local_block_size=12, kernel_block_size=4, num_logical_blocks=8
    )
    meta_r = _make_remote_meta(
        worker,
        remote_block_size=8,
        remote_kernel_block_size=4,
        remote_num_logical=12,
        remote_ssm_sizes=(24, 32),
    )
    # Equal kernel block sizes (ratio 1), but a half-sized per-token page.
    meta_r.block_lens = [x // 2 for x in worker.block_len_per_layer]
    with pytest.raises((AssertionError, RuntimeError)):
        worker.add_remote_agent(meta_r, remote_tp_rank=0, remote_tp_size=2)
