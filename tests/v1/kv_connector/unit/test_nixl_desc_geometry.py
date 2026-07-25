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

from unittest.mock import patch

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
        self._next_handle = 1

    def get_reg_descs(self, caches_data, mem_type):
        return caches_data

    def register_memory(self, descs, backends=None):
        pass

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
        return None

    def release_xfer_handle(self, handle):
        pass

    def release_dlist_handle(self, handle):
        pass

    def send_notif(self, agent, notif_msg=None):
        pass

    def remove_remote_agent(self, agent):
        pass


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
    kv_cache_config = KVCacheConfig(
        num_blocks=num_logical_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_logical_blocks * unified_page,
                shared_by=[f"mla.{i}", f"kda_a.{i}", f"kda_b.{i}"],
            )
            for i in range(2)
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["mla.0", "mla.1"], mla_spec),
            KVCacheGroupSpec(["kda_a.0", "kda_a.1"], kda_spec),
            KVCacheGroupSpec(["kda_b.0", "kda_b.1"], kda_spec),
        ],
    )

    vllm_config = create_vllm_config(block_size=local_block_size)
    vllm_config.cache_config.enable_prefix_caching = False

    from unittest.mock import MagicMock

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

        tensors = [
            torch.zeros(num_logical_blocks * unified_page, dtype=torch.uint8)
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
    # Keep tensors alive alongside the worker.
    worker._test_tensors = tensors
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
    # match the local ones for the handshake to pass.
    kernel_page = worker.block_len_per_layer[0]
    return NixlAgentMetadata(
        engine_id="remote-engine",
        agent_metadata=b"remote-agent-meta",
        device_id=0,
        kv_caches_base_addr=[0x10_000_000, 0x20_000_000],
        num_blocks=remote_num_logical * remote_ppl,
        block_lens=[kernel_page, kernel_page],
        kv_cache_layout=worker.kv_cache_layout,
        block_size=remote_kernel_block_size,
        ssm_sizes=remote_ssm_sizes,
        attn_backend_name=worker.backend_name,
        physical_blocks_per_logical_kv_block=remote_ppl,
    )


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
    for rank in (0, 1):
        worker.add_remote_agent(meta_r, remote_tp_rank=rank, remote_tp_size=2)

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
            "remote_host": "localhost",
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
    desc_arr, idx, bases, region_size, unified_page, kernel_page, logical_ids_attn
):
    """Resolve a desc id to (region, kind, token_start) where kind is 'attn'
    (kernel-page sized, block-aligned, in the request's attention blocks) or
    'mamba'. token_start is the request-relative kernel-block index."""
    addr, length, _ = (int(x) for x in desc_arr[int(idx)])
    for region, base in enumerate(bases):
        off = addr - base
        if 0 <= off < region_size:
            b = off // unified_page
            rem = off % unified_page
            if (
                length == kernel_page
                and rem % kernel_page == 0
                and (b in logical_ids_attn)
            ):
                pos = logical_ids_attn.index(b)
                tokens_per_block = unified_page // kernel_page
                sub = rem // kernel_page
                return (region, "attn", (pos * tokens_per_block + sub))
            return (region, "mamba", None)
    raise AssertionError(f"desc {idx} addr {addr:#x} not in any region")


def _run_hetero_case(local_block, kernel, remote_block, num_tokens):
    """Full pull-path run for one geometry; returns pairing records."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    remote_ppl = remote_block // kernel
    matched = num_tokens - 1  # mamba N-1 rule
    n_local = -(-num_tokens // local_block)
    n_remote = -(-matched // remote_block)

    worker = _make_mla_hybrid_worker(
        local_block_size=local_block,
        kernel_block_size=kernel,
        num_logical_blocks=max(2 * n_local + 4, 8),
    )
    meta_r = _make_remote_meta(
        worker,
        remote_block_size=remote_block,
        remote_kernel_block_size=kernel,
        remote_num_logical=max(2 * n_remote + 4, 8),
        remote_ssm_sizes=(24, 32),
    )
    for rank in (0, 1):
        worker.add_remote_agent(meta_r, remote_tp_rank=rank, remote_tp_size=2)

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
            "remote_host": "localhost",
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

    # Invariant 1: all local writes within the request's own blocks.
    owned = _owned_byte_ranges(worker, local_ids)
    _assert_local_writes_within(worker, owned)

    # Invariant 2: local<->remote attention pairs are token-aligned.
    nixl = worker.nixl_wrapper
    local_bases = [t.data_ptr() for t in worker._test_tensors]
    remote_bases = [0x10_000_000, 0x20_000_000]
    local_unified = worker._test_unified_page
    remote_unified = (local_unified // local_block) * remote_block
    kernel_page = worker.block_len_per_layer[0]
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
                kernel_page,
                local_attn,
            )
            rreg, rkind, rtok = _resolve(
                rarr,
                ri,
                remote_bases,
                meta_r_num_blocks_bytes,
                remote_unified,
                kernel_page,
                remote_attn,
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
                    f"TOKEN MISALIGNMENT: local kernel block holds tokens "
                    f"[{ltok * kernel}..) but receives remote tokens "
                    f"[{rtok * kernel}..) "
                    f"(geometry local_block={local_block}, "
                    f"remote_block={remote_block}, N={num_tokens})"
                )
                covered_tokens.add(ltok * kernel)

    # Invariant 3: full coverage of the matched tokens.
    needed = {t for t in range(0, matched - matched % kernel, kernel)}
    missing = needed - covered_tokens
    assert not missing, (
        f"tokens never transferred: {sorted(missing)[:8]} "
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
