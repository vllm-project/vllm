# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class BigtablePluginConfig:
    project_id: str
    instance_id: str
    table_name: str
    app_profile_id: Optional[str] = None
    read_timeout_sec: float = 0.2
    write_timeout_sec: float = 0.5
    exists_cache_ttl_seconds: float = 30.0
    exists_cache_size: int = 10000
    thread_pool_size: int = 16
    row_key_template: str = "{hash}#{model}"
    credentials_path: Optional[str] = None
    max_retries: int = 3
    max_chunk_size_mb: float = 90.0
    family_name: str = "cf"
    column_name: str = "data"
    layer_group_size: int = 10

    @classmethod
    def from_extra_config(
        cls, extra_config: Dict[str, Any], plugin_name: Optional[str] = None
    ) -> "BigtablePluginConfig":
        # Standard
        import os

        def get_val(key: str, default: Any = None) -> Any:
            if plugin_name:
                full_key = f"remote_storage_plugin.{plugin_name}.{key}"
                if full_key in extra_config:
                    return extra_config[full_key]
            return extra_config.get(key, default)

        project_id = get_val("bigtable_project_id") or os.environ.get("BT_PROJECT_ID")
        instance_id = get_val("bigtable_instance_id") or os.environ.get(
            "BT_INSTANCE_ID"
        )
        table_name = get_val("bigtable_table_name") or os.environ.get("BT_TABLE_NAME")

        if not project_id or not instance_id or not table_name:
            raise ValueError(
                f"Bigtable out-of-tree connector requires bigtable_project_id, "
                f"bigtable_instance_id, and bigtable_table_name (or BT_* env vars). "
                f"Got project={project_id}, instance={instance_id}, table={table_name}"
            )

        return cls(
            project_id=project_id,
            instance_id=instance_id,
            table_name=table_name,
            app_profile_id=get_val(
                "bigtable_app_profile",
                get_val("app_profile", os.environ.get("BT_APP_PROFILE")),
            ),
            read_timeout_sec=float(
                get_val(
                    "bigtable_read_timeout_ms",
                    get_val(
                        "read_timeout_ms",
                        os.environ.get("BT_READ_TIMEOUT_MS", 200.0),
                    ),
                )
            )
            / 1000.0,
            write_timeout_sec=float(
                get_val(
                    "bigtable_write_timeout_ms",
                    get_val(
                        "write_timeout_ms",
                        os.environ.get("BT_WRITE_TIMEOUT_MS", 500.0),
                    ),
                )
            )
            / 1000.0,
            exists_cache_ttl_seconds=float(
                get_val(
                    "bigtable_exists_cache_ttl_seconds",
                    get_val(
                        "exists_cache_ttl_seconds",
                        os.environ.get("BT_EXISTS_CACHE_TTL_SECONDS", 30.0),
                    ),
                )
            ),
            exists_cache_size=int(
                get_val(
                    "bigtable_exists_cache_size",
                    get_val(
                        "exists_cache_size",
                        os.environ.get("BT_EXISTS_CACHE_SIZE", 10000),
                    ),
                )
            ),
            thread_pool_size=int(
                get_val(
                    "bigtable_thread_pool_size",
                    get_val(
                        "thread_pool_size", os.environ.get("BT_THREAD_POOL_SIZE", 16)
                    ),
                )
            ),
            row_key_template=get_val(
                "bigtable_row_key_template",
                get_val(
                    "row_key_template",
                    os.environ.get("BT_ROW_KEY_TEMPLATE", "{hash}#{model}"),
                ),
            ),
            credentials_path=get_val(
                "bigtable_credentials_path",
                get_val("credentials_path", os.environ.get("BT_CREDENTIALS_PATH")),
            ),
            max_retries=int(
                get_val(
                    "bigtable_max_retries",
                    get_val("max_retries", os.environ.get("BT_MAX_RETRIES", 3)),
                )
            ),
            max_chunk_size_mb=float(
                get_val(
                    "bigtable_max_chunk_size_mb",
                    os.environ.get("BT_MAX_CHUNK_SIZE_MB", 90.0),
                )
            ),
            family_name=get_val(
                "bigtable_family_name", os.environ.get("BT_FAMILY_NAME", "cf")
            ),
            column_name=get_val(
                "bigtable_column_name", os.environ.get("BT_COLUMN_NAME", "data")
            ),
            layer_group_size=int(
                get_val(
                    "bigtable_layer_group_size",
                    get_val(
                        "layer_group_size",
                        os.environ.get("BT_LAYER_GROUP_SIZE", 10),
                    ),
                )
            ),
        )
