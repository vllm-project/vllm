# SPDX-License-Identifier: Apache-2.0
"""
Bigtable key encoder for translating ObjectKeys to row keys.
"""

# First Party
from lmcache.v1.distributed.api import ObjectKey


class BigtableL2KeyEncoder:
    """Encoder to translate ObjectKeys into Bigtable row keys using a template."""

    def __init__(
        self,
        template: str = "{hash_prefix}@{model}@{rank}@{group}@{hash}@{salt}",
        layer_group_size: int = 0,
    ):
        self.template = template
        self.layer_group_size = layer_group_size

    def encode_row_key(self, key: ObjectKey) -> bytes:
        """Translates a logical ObjectKey into a physical Bigtable row key.

        Formats the row key using the configured template, injecting the model
        name, rank, group, chunk hash, and optional cache salt.

        Args:
            key: The ObjectKey to encode.

        Returns:
            The formatted row key as bytes.
        """
        template = self.template
        # Strip @{salt} if salt is empty to prevent a trailing @
        if not key.cache_salt and template.endswith("@{salt}"):
            template = template[:-7]

        model_name = key.model_name
        if self.layer_group_size > 0:
            model_name = f"{model_name}@lg{self.layer_group_size}"

        hash_hex = key.chunk_hash.hex()
        row_key_str = template.format(
            hash_prefix=hash_hex[:4],
            model=model_name,
            rank=f"{key.kv_rank:08x}",
            group=f"{key.object_group_id:x}",
            hash=hash_hex,
            salt=key.cache_salt,
        )
        if hasattr(key, "layer_id"):
            row_key_str += f"@layer_{key.layer_id}"
        return row_key_str.encode("utf-8")
