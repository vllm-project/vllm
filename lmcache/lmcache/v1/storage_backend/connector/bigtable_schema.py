# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any, Optional

# First Party
from lmcache.utils import LayerCacheEngineKey

if TYPE_CHECKING:
    # First Party
    from lmcache.utils import CacheEngineKey


class BigtableSchema:
    def __init__(
        self,
        row_key_template: str,
        family_name: str,
        column_name: str,
        layer_group_size: int = 0,
    ):
        self.row_key_template = row_key_template
        self.family_name = family_name
        self.column_name = column_name
        self.layer_group_size = layer_group_size

    def get_row_key(self, key: "CacheEngineKey") -> bytes:
        fingerprint = (
            f"{key.model_name}@{key.world_size}@{key.worker_id}@{key._dtype_str}"
        )
        if self.layer_group_size > 0:
            fingerprint += f"@lg{self.layer_group_size}"

        if key.tags is not None and len(key.tags) != 0:
            tags_str = "@".join([f"{k}%{v}" for k, v in key.tags])
            fingerprint += f"@{tags_str}"

        template = self.row_key_template
        if "{hash}" not in template and "hash" in template:
            template = template.replace("hash", "{hash}")
        if "{model}" not in template and "model" in template:
            template = template.replace("model", "{model}")

        row_key_str = template.format(
            hash=key.chunk_hash_hex,
            model=fingerprint,
        )

        if isinstance(key, LayerCacheEngineKey):
            row_key_str += f"@layer_{key.layer_id}"

        return row_key_str.encode("utf-8")

    def extract_cell_value(self, row: Any) -> Optional[bytes]:
        if row is None or not hasattr(row, "cells"):
            return None

        col_bytes = (
            self.column_name.encode("utf-8")
            if isinstance(self.column_name, str)
            else self.column_name
        )

        # Handle dict access (classic) or list access (v2 data client) flexibly
        if isinstance(row.cells, dict):
            cells_dict = row.cells.get(self.family_name, {})
            col_cells = cells_dict.get(col_bytes, [])
            if col_cells:
                return col_cells[0].value
        else:
            for cell in row.cells:
                if cell.family == self.family_name and cell.qualifier == col_bytes:
                    return cell.value
        return None
