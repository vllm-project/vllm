# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure layout planning for member-ordered NIXL transfers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)


@dataclass(frozen=True)
class MemberTransferPlan:
    """Local descriptor layout for a member-ordered transfer."""

    member_names: tuple[str, ...]
    local_regions: tuple[int, ...]
    group_ids: tuple[int, ...]


def validate_region_members(region_members: Sequence[Sequence[str]]) -> None:
    """Ensure every layer name belongs to exactly one region."""
    members = [member for region in region_members for member in region]
    if len(members) != len(set(members)):
        raise RuntimeError("A KV cache layer spans multiple NIXL regions")


def plan_member_transfer(
    remote_metadata: NixlAgentMetadata,
    local_region_members: Sequence[Sequence[str]],
    layer_to_group: Mapping[str, int],
) -> tuple[NixlAgentMetadata, MemberTransferPlan]:
    """Build canonical member metadata and a plan without mutating inputs."""
    num_regions = len(remote_metadata.region_members)
    if (
        len(remote_metadata.kv_caches_base_addr) != num_regions
        or len(remote_metadata.block_lens) != num_regions
    ):
        raise RuntimeError("NIXL region metadata has inconsistent lengths")

    remote_region_of: dict[str, int] = {}
    for region_idx, remote_members in enumerate(remote_metadata.region_members):
        for layer_name in remote_members:
            if layer_name in remote_region_of:
                raise RuntimeError(
                    f"Remote advertised KV cache layer {layer_name!r} in "
                    "multiple NIXL regions"
                )
            remote_region_of[layer_name] = region_idx

    member_names: list[str] = []
    local_regions: list[int] = []
    group_ids: list[int] = []
    remote_base_addresses: list[int] = []
    remote_block_lens: list[int] = []
    for local_region, local_members in enumerate(local_region_members):
        for layer_name in local_members:
            remote_region = remote_region_of.get(layer_name)
            if remote_region is None:
                raise RuntimeError(
                    "Remote NIXL metadata is missing locally owned KV cache "
                    f"layer {layer_name!r}; the transfer would leave it stale"
                )
            group_id = layer_to_group.get(layer_name)
            if group_id is None:
                raise RuntimeError(
                    f"KV cache layer {layer_name!r} is not in a local cache group"
                )

            remote_base = remote_metadata.kv_caches_base_addr[remote_region]
            remote_len = remote_metadata.block_lens[remote_region]
            member_names.append(layer_name)
            local_regions.append(local_region)
            group_ids.append(group_id)
            remote_base_addresses.append(remote_base)
            remote_block_lens.append(remote_len)

    if not member_names:
        raise RuntimeError("Remote NIXL metadata has no members owned by this worker")

    prepared_metadata = replace(
        remote_metadata,
        kv_caches_base_addr=remote_base_addresses,
        block_lens=remote_block_lens,
        region_members=[],
    )
    return prepared_metadata, MemberTransferPlan(
        member_names=tuple(member_names),
        local_regions=tuple(local_regions),
        group_ids=tuple(group_ids),
    )
