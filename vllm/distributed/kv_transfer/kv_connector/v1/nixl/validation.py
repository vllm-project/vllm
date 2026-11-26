# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation of a remote NIXL peer against the local worker's layout."""

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )

logger = init_logger(__name__)


def validate_remote_parallel_config(
    local_pcp_size: int, local_dcp_size: int, agent_metadata: NixlAgentMetadata
) -> None:
    remote_pcp_size = agent_metadata.pcp_size
    remote_dcp_size = agent_metadata.dcp_size
    if remote_pcp_size > 1 and remote_dcp_size not in (1, remote_pcp_size):
        raise NotImplementedError(
            "Remote NixlConnector PCP+DCP does not span the full PCP group. "
            f"Remote PCP/DCP={remote_pcp_size}/{remote_dcp_size}."
        )
    if (local_pcp_size > 1 and local_dcp_size == 1 and remote_dcp_size > 1) or (
        remote_pcp_size > 1 and remote_dcp_size == 1 and local_dcp_size > 1
    ):
        raise NotImplementedError(
            "Replicated PCP cannot be paired with a DCP-sharded NIXL peer. "
            f"Local PCP/DCP={local_pcp_size}/{local_dcp_size}; "
            f"remote PCP/DCP={remote_pcp_size}/{remote_dcp_size}."
        )


def validate_csa_linear_tp_layout(
    local_tp_size: int, remote_tp_size: int, total_kv_heads: int
) -> None:
    """Reject P/D pairs whose main-KV pages have different head layouts.

    A rank at TP < total KV heads holds several heads per page, while a
    rank at TP >= total KV heads holds exactly one (replicated beyond
    that). CSA-linear pages are shared tensors that transfer as whole
    block units, so endpoints on opposite sides of that boundary lay
    them out differently and cannot exchange them.
    """
    if (local_tp_size < total_kv_heads) != (remote_tp_size < total_kv_heads):
        raise ValueError(
            "CSA-linear NIXL requires both endpoints on the same side of "
            f"the KV-head sharding boundary ({total_kv_heads} KV heads): "
            f"got local TP {local_tp_size}, remote TP {remote_tp_size}."
        )


def validate_push_block_sizes(
    worker: "NixlBaseConnectorWorker", agent_metadata: NixlAgentMetadata
) -> None:
    local = (worker.block_size, worker._physical_blocks_per_logical_kv_block)
    remote = (
        agent_metadata.block_size,
        max(agent_metadata.physical_blocks_per_group, default=1),
    )
    if local != remote:
        raise NotImplementedError(
            "NixlPushConnector requires identical P/D block sizes. Kernel "
            "block size and kernel blocks per logical block: local "
            f"{local}, remote engine {agent_metadata.engine_id} {remote}."
        )


def check_remote_pp_supported(
    worker: "NixlBaseConnectorWorker", nixl_agent_meta: NixlAgentMetadata
) -> None:
    """Refuse layouts that per-stage pulls from a pipeline-parallel producer
    do not support. Under HMA a stage may hold no layer of a cache group (a
    draft's group on the stages before the last); it then has no pair of
    that group, so its reads skip the group."""
    if (
        worker._has_packed_cache
        or KVCacheLayout[nixl_agent_meta.kv_cache_layout].is_block_outermost
    ):
        raise NotImplementedError(
            "NixlConnector (pull) does not support PP with packed KV cache layouts "
            f"(local {worker.kv_cache_layout}, "
            f"remote {nixl_agent_meta.kv_cache_layout}). "
            "Use supported non-packed layouts on both endpoints; "
            "NixlPushConnector has separate packed PP restrictions."
        )
    if (
        worker.pp_size > 1
        or worker._has_mamba
        or (worker._is_hma_required and (worker.use_mla or worker._is_csa_linear))
        or worker.dcp_size > 1
        or nixl_agent_meta.dcp_size > 1
        or worker._mixed_mem_types
    ):
        raise NotImplementedError(
            "Pulling KV from a pipeline-parallel producer requires a PP=1 "
            "consumer without Mamba, DCP or mixed-memory KV caches, and HMA "
            "only for attention KV without MLA or CSA-linear caches."
        )


def validate_remote_agent_handshake(
    worker: "NixlBaseConnectorWorker",
    nixl_agent_meta: NixlAgentMetadata,
    remote_tp_size: int,
    remote_dcp_size: int = 1,
):
    """Validate the remote agent handshake metadata ensuring the
    invariants hold true."""
    remote_engine_id = nixl_agent_meta.engine_id

    assert worker.transfer_topo is not None
    remote_info = worker.transfer_topo.get_engine_info(remote_engine_id)
    assert remote_info.remote_tp_size == remote_tp_size
    assert remote_info.remote_dcp_size == remote_dcp_size
    # DCP sizes must divide one another; this is what keeps the
    # read-slicing math in pull_worker a closed form.
    assert (
        worker.dcp_size % remote_dcp_size == 0 or remote_dcp_size % worker.dcp_size == 0
    ), (
        f"DCP sizes must divide one another: local={worker.dcp_size}, "
        f"remote={remote_dcp_size} (engine {remote_engine_id})."
    )
    if worker._has_mamba and worker.dcp_size != remote_dcp_size:
        raise RuntimeError(
            "Hybrid MLA+Mamba NIXL transfers require matching DCP sizes, "
            f"got local={worker.dcp_size}, remote={remote_dcp_size}."
        )
    if worker._dcp_replicated_groups and (
        worker.dcp_size > 1 or (remote_dcp_size > 1 and worker._TRANSFER_MODE != "pull")
    ):
        raise NotImplementedError(
            "NIXL with a DCP-replicated MLA draft cache only supports "
            "pulling from a DCP prefill into a non-DCP decode: "
            f"local DCP={worker.dcp_size}, remote DCP={remote_dcp_size}, "
            f"transfer_mode={worker._TRANSFER_MODE}."
        )

    block_size_ratio = worker.transfer_topo.block_size_ratio(nixl_agent_meta.block_size)
    remote_physical_per_logical = max(
        nixl_agent_meta.physical_blocks_per_group, default=1
    )
    groups_match = worker._groups_match(remote_engine_id)
    if not groups_match and worker._TRANSFER_MODE == "pull":
        # Blocks pair by group pair; these were never validated with a
        # remote that puts some layer in another cache group.
        if worker.block_size != nixl_agent_meta.block_size:
            raise NotImplementedError(
                "Different NIXL cache-group layouts require matching physical "
                "block sizes"
            )
        if (
            worker.dcp_size > 1 or remote_dcp_size > 1
        ) and worker._dcp_replicated_groups:
            raise NotImplementedError(
                "Different NIXL cache-group layouts with DCP do not support "
                "DCP-replicated MLA draft groups"
            )
    if (
        (worker.dcp_size > 1 or remote_dcp_size > 1)
        and not groups_match
        and (
            worker._physical_blocks_per_logical_kv_block != 1
            or remote_physical_per_logical != 1
        )
    ):
        raise NotImplementedError(
            "Different NIXL cache-group layouts with DCP require matching "
            "logical and physical block sizes"
        )
    if (
        worker._has_mamba
        and remote_physical_per_logical != worker._physical_blocks_per_logical_kv_block
        and worker.vllm_config.cache_config.enable_prefix_caching
    ):
        raise RuntimeError(
            "Prefix caching with heterogeneous physical_blocks_per_logical "
            "is not supported for Mamba hybrid models. "
            f"Local: {worker._physical_blocks_per_logical_kv_block}, "
            f"Remote: {remote_physical_per_logical}. "
            "Disable prefix caching with --disable-prefix-caching."
        )

    if block_size_ratio != 1:
        # Heterogeneous block sizes transfer at remote-block granularity;
        # the untransferred tail of the last local attention block is
        # zeroed in the receive post-process, and mamba state pages
        # transfer 1:1 (never sub-split).
        assert not worker.use_host_buffer, (
            "Heterogeneous block sizes are not supported with host buffer"
        )
    kv_cache_layout = worker.transfer_kv_cache_layout
    # The layout orders a block's heads and tokens, which a single head
    # (e.g. MLA's latent) does not distinguish.
    multi_head = any(
        isinstance(spec, AttentionSpec) and spec.num_heads > 1
        for spec in worker._layer_specs.values()
    )
    if multi_head and nixl_agent_meta.kv_cache_layout != kv_cache_layout:
        if (
            worker.kv_transfer_config.enable_permute_local_kv
            and nixl_agent_meta.kv_cache_layout == "LBHNC"
            and kv_cache_layout == "LBNHC"
        ):
            logger.info(
                "Remote is LBHNC and local is LBNHC, enabled additional permute "
                "on local device KV."
            )
            assert not worker._is_hma_required, (
                "HMA does not support block size post processing"
            )
            worker.enable_permute_local_kv = True
        else:
            raise RuntimeError(
                "Heterogeneous TP expects same kv_cache_layout (local "
                f"{kv_cache_layout}, remote {nixl_agent_meta.kv_cache_layout}). "
                "Or enable experimental feature to receive LBHNC into a local "
                "LBNHC cache by setting 'enable_permute_local_kv'=True "
                "in --kv-transfer-config."
            )
    # if remote_agent used attn is not same as local,
    # hint heterogenuous attn post process
    if (
        nixl_agent_meta.attn_backend_name != worker.backend_name
        and worker.backend_name in ["CPU_ATTN"]
    ):
        if worker._is_hma_required:
            raise RuntimeError(
                "heterogeneous attn post process is not supported with HMA"
            )
        logger.info(
            "[Experimental] CPU_ATTN backend is used, "
            "hint heterogeneous attn post process"
        )
        worker.enable_heterogeneous_attn_post_process = True

    # Block lengths are checked against the specs in _tp_byte_ranges.
    # TP workers that handhshake with same remote have same #blocks.
    assert worker.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks
    num_remote_regions = len(nixl_agent_meta.kv_caches_base_addr)
    if nixl_agent_meta.region_num_blocks is not None:
        assert len(nixl_agent_meta.region_num_blocks) == num_remote_regions
    assert len(nixl_agent_meta.block_strides) == num_remote_regions
    assert all(
        stride >= block_len
        for stride, block_len in zip(
            nixl_agent_meta.block_strides,
            nixl_agent_meta.block_lens,
            strict=True,
        )
    )
    if nixl_agent_meta.region_group_ids is not None:
        assert len(nixl_agent_meta.region_group_ids) == num_remote_regions
    if nixl_agent_meta.region_mem_types is not None:
        assert len(nixl_agent_meta.region_mem_types) == num_remote_regions
