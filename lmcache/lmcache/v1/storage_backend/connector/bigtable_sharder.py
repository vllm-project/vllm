# SPDX-License-Identifier: Apache-2.0
"""Bigtable payload sharder for layer-group sharding."""

# Standard
from typing import Dict


class BigtablePayloadSharder:
    """Slices contiguous tensor bytes into layer groups and reassembles them."""

    def __init__(self, num_layers: int, layer_group_size: int, kv_size: int = 2):
        """Initializes the BigtablePayloadSharder.

        Args:
            num_layers: Total number of layers.
            layer_group_size: Number of layers per group.
            kv_size: Size of KV dimensions (typically 2 for K and V).
        """
        self.num_layers = num_layers
        self.layer_group_size = layer_group_size
        self.kv_size = kv_size

    def get_group_qualifier(self, group_idx: int) -> str:
        """Generate the column qualifier name for a given group index.

        Args:
            group_idx: The 0-based index of the layer group.

        Returns:
            A string representing the column qualifier (e.g., 'layers_0_9').
        """
        start = group_idx * self.layer_group_size
        end = min((group_idx + 1) * self.layer_group_size, self.num_layers) - 1
        return f"layers_{start}_{end}"

    def shard(self, payload: bytes) -> Dict[str, bytes]:
        """Shard contiguous payload bytes into column qualifier to bytes map.

        Args:
            payload: Contiguous payload bytes.

        Returns:
            A dictionary mapping column qualifiers to their sharded bytes.
        """
        total_bytes = len(payload)

        # Verify divisibility
        if total_bytes % (self.kv_size * self.num_layers) != 0:
            raise ValueError(
                f"Payload size {total_bytes} is not a multiple of "
                f"kv_size * num_layers ({self.kv_size * self.num_layers})"
            )

        layer_size = total_bytes // (self.kv_size * self.num_layers)
        shards: Dict[str, bytes] = {}

        num_groups = (
            self.num_layers + self.layer_group_size - 1
        ) // self.layer_group_size
        for g in range(num_groups):
            start_layer = g * self.layer_group_size
            end_layer = min((g + 1) * self.layer_group_size, self.num_layers)

            group_payload = b""
            for kv_idx in range(self.kv_size):
                # Offset to the start of this kv component
                kv_offset = kv_idx * self.num_layers * layer_size
                start = kv_offset + start_layer * layer_size
                end = kv_offset + end_layer * layer_size
                group_payload += payload[start:end]

            qualifier = self.get_group_qualifier(g)
            shards[qualifier] = group_payload

        return shards

    def reassemble(self, shards: Dict[str, bytes]) -> bytes:
        """Reassemble shards map back into contiguous payload bytes.

        Args:
            shards: A dictionary mapping column qualifiers to sharded bytes.

        Returns:
            The reassembled contiguous payload bytes.
        """
        num_groups = (
            self.num_layers + self.layer_group_size - 1
        ) // self.layer_group_size

        if not shards:
            return b""

        # Determine layer_size from the first available shard
        layer_size = None
        for g in range(num_groups):
            qualifier = self.get_group_qualifier(g)
            if qualifier in shards:
                start_layer = g * self.layer_group_size
                end_layer = min((g + 1) * self.layer_group_size, self.num_layers)
                group_layers = end_layer - start_layer
                layer_size = len(shards[qualifier]) // (self.kv_size * group_layers)
                break

        if layer_size is None:
            raise ValueError("No matching shards found to reassemble")

        kv_parts: list[list[bytes]] = [[] for _ in range(self.kv_size)]

        for g in range(num_groups):
            qualifier = self.get_group_qualifier(g)
            if qualifier not in shards:
                raise ValueError(f"Missing shard for column {qualifier}")

            shard_data = shards[qualifier]
            start_layer = g * self.layer_group_size
            end_layer = min((g + 1) * self.layer_group_size, self.num_layers)
            group_layers = end_layer - start_layer

            expected_size = self.kv_size * group_layers * layer_size
            if len(shard_data) != expected_size:
                raise ValueError(
                    f"Shard {qualifier} size mismatch: "
                    f"expected {expected_size}, got {len(shard_data)}"
                )

            group_part_size = group_layers * layer_size
            for kv_idx in range(self.kv_size):
                start = kv_idx * group_part_size
                end = (kv_idx + 1) * group_part_size
                kv_parts[kv_idx].append(shard_data[start:end])

        # Concatenate each kv component across all groups, then join them
        reassembled = b""
        for kv_idx in range(self.kv_size):
            reassembled += b"".join(kv_parts[kv_idx])

        return reassembled
