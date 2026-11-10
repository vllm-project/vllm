# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup checks for the ROCm Step5 sparse attention layers."""

from vllm.config import KVTransferConfig, VllmConfig


def verify_kv_transfer_compatibility(vllm_config: VllmConfig) -> None:
    config = vllm_config.kv_transfer_config
    if config is None or not config.is_kv_transfer_instance:
        return
    configs = [config]
    if config.kv_connector == "MultiConnector":
        children = config.kv_connector_extra_config["connectors"]
        configs = [KVTransferConfig(**child) for child in children]
    if not all(
        c.kv_connector == "MooncakeStoreConnector"
        or (c.kv_connector == "NixlConnector" and c.kv_buffer_device == "cuda")
        for c in configs
    ):
        raise ValueError(
            "Step5 sparse summaries are only transferred by "
            "MooncakeStoreConnector, or by NixlConnector with device buffers, "
            "directly or under a MultiConnector."
        )


def _groups_per_rank(total_groups: int, tp_size: int) -> int:
    if total_groups % tp_size == 0:
        return total_groups // tp_size
    if tp_size % total_groups == 0:
        return 1
    raise ValueError(
        f"Step5 tensor parallel size {tp_size} is incompatible with "
        f"group count {total_groups}: one must divide the other."
    )


def validate_parallel_geometry(
    *,
    total_num_heads: int,
    total_num_kv_heads: int,
    indexer_num_heads: int,
    index_tp_size: int,
    tp_size: int,
) -> None:
    """The indexer scores regions per provider group and attention reads the
    matching KV group, so both partitions must agree globally and per rank."""
    if index_tp_size != total_num_kv_heads:
        raise ValueError(
            "Step5 sparse provider groups must match attention KV groups: "
            f"{index_tp_size} != {total_num_kv_heads}"
        )
    if indexer_num_heads % index_tp_size:
        raise ValueError(
            f"{indexer_num_heads} sparse indexer heads do not divide into "
            f"{index_tp_size} provider groups."
        )
    if total_num_heads % tp_size:
        raise ValueError(
            f"Step5 attention heads ({total_num_heads}) must be divisible by "
            f"the tensor parallel size ({tp_size})."
        )
    local_kv_groups = _groups_per_rank(total_num_kv_heads, tp_size)
    q_heads_per_kv, rem = divmod(total_num_heads // tp_size, local_kv_groups)
    if rem or q_heads_per_kv not in (4, 8, 16):
        raise ValueError(
            "Step5 sparse attention needs 4, 8 or 16 local query heads per KV "
            f"group (num_heads={total_num_heads}, tp_size={tp_size})."
        )
