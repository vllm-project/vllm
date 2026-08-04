Remote Storage Plugins
========================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


LMCache supports built-in remote storage connectors for Redis, InfiniStore,
MooncakeStore, S3, Hugging Face Buckets, and more.
The remote storage plugin system provides the ability to add custom storage connectors through dynamic loading. This enables extending remote storage capabilities without modifying core code.

.. note::

   The ``remote_url`` configuration is **deprecated** and will be removed in a future release.
   Please use ``remote_storage_plugins`` instead.

Connector Definition Requirements
---------------------------------
A custom remote storage connector requires two classes:

1. **ConnectorAdapter**: Handles URL scheme matching and connector instantiation

   - Inherit from ``ConnectorAdapter``
   - Set the URL scheme in the constructor (e.g., ``mystore://``)
   - Implement the ``create_connector`` method

2. **RemoteConnector**: Implements the actual storage operations

   - Inherit from ``RemoteConnector``
   - Implement all abstract methods: ``exists``, ``exists_sync``, ``get``, ``put``, ``list``, ``close``

.. note::

   The ``ConnectorAdapter`` constructor receives no arguments from LMCache. The scheme should be set by calling the parent constructor with the scheme string.

   The ``create_connector`` method receives a ``ConnectorContext`` object containing the URL, event loop, local CPU backend, config, metadata, and ``plugin_name``.

Plugin Naming Convention
-------------------------
Plugin names follow the format ``{type}`` or ``{type}.{instance}``:

- ``{type}`` — a single instance of that connector type (e.g. ``fs``, ``mooncakestore``)
- ``{type}.{instance}`` — a named instance, allowing **multiple instances of the same type** (e.g. ``fs.primary``, ``fs.backup``)

The framework extracts the *type* portion (everything before the first ``.``) to locate the matching ``ConnectorAdapter``. The full plugin name is used as the configuration key prefix.

Using Built-in Connectors via Plugins
-------------------------------------
Built-in connectors (``fs``, ``mooncakestore``, ``hfbucket``, etc.) can be used
directly via ``remote_storage_plugins`` without specifying ``module_path`` or
``class_name``. Their configuration is placed under ``extra_config``:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    remote_storage_plugins: ["fs"]
    extra_config:
      remote_storage_plugin.fs.base_path: /tmp/lmcache

Multiple instances of the same connector type:

.. code-block:: yaml

    remote_storage_plugins: ["fs.primary", "fs.backup"]
    extra_config:
      remote_storage_plugin.fs.primary.base_path: /data/cache1
      remote_storage_plugin.fs.backup.base_path: /data/cache2

Mixing different connector types:

.. code-block:: yaml

    remote_storage_plugins: ["fs.local", "mooncakestore"]
    extra_config:
      remote_storage_plugin.fs.local.base_path: /data/cache
      remote_storage_plugin.mooncakestore.master_server_address: "localhost:50051"

Built-in Hugging Face Buckets example:

.. code-block:: yaml

    remote_storage_plugins: ["hfbucket"]
    extra_config:
      remote_storage_plugin.hfbucket.bucket_handle: hf://buckets/my-org/lmcache-kv/prod
      remote_storage_plugin.hfbucket.token_env: HF_TOKEN

How to Integrate Custom Remote Storage with LMCache
---------------------------------------------------
1. Install your connector package in the LMCache environment
2. Add ``remote_storage_plugins`` and its related ``module_path`` and ``class_name`` to the ``extra_config`` section of LMCache configuration as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    remote_storage_plugins: ["mystore"]
    extra_config:
      remote_storage_plugin.mystore.module_path: <module_path>
      remote_storage_plugin.mystore.class_name: <adapter_class_name>

An example configuration for a custom remote storage connector:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    remote_storage_plugins: ["mystore"]
    extra_config:
      remote_storage_plugin.mystore.module_path: my_package.my_connector
      remote_storage_plugin.mystore.class_name: MyStoreConnectorAdapter

Multiple instances of a custom connector:

.. code-block:: yaml

    remote_storage_plugins: ["mystore.region_a", "mystore.region_b"]
    extra_config:
      remote_storage_plugin.mystore.region_a.module_path: my_package.my_connector
      remote_storage_plugin.mystore.region_a.class_name: MyStoreConnectorAdapter
      remote_storage_plugin.mystore.region_b.module_path: my_package.my_connector
      remote_storage_plugin.mystore.region_b.class_name: MyStoreConnectorAdapter

.. note::

   - ``remote_url`` is **deprecated**; use ``remote_storage_plugins`` instead
   - ``remote_storage_plugin.<plugin_name>`` uses the full plugin name (including instance suffix) as the key prefix
   - Multiple remote storage plugins can be loaded simultaneously
   - Built-in connectors do not require ``module_path`` / ``class_name``

ConnectorAdapter Implementation
-------------------------------
The ``ConnectorAdapter`` class is responsible for:

- Defining the URL scheme it handles
- Creating the appropriate ``RemoteConnector`` instance

.. code-block:: python

    from lmcache.v1.storage_backend.connector import (
        ConnectorAdapter,
        ConnectorContext,
    )
    from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

    from lmcache.v1.storage_backend.connector import extract_plugin_type

    PLUGIN_TYPE = "mystore"

    class MyStoreConnectorAdapter(ConnectorAdapter):
        """Adapter for MyStore remote storage."""

        def __init__(self) -> None:
            # Register the URL scheme this adapter handles
            super().__init__("mystore://")

        def can_parse(self, url: str) -> bool:
            """Match legacy URL or plugin://{type}[.{instance}] format."""
            if url.startswith(self.schema):
                return True
            if url.startswith("plugin://"):
                pname = url[len("plugin://"):]
                return extract_plugin_type(pname) == PLUGIN_TYPE
            return False

        def create_connector(self, context: ConnectorContext) -> RemoteConnector:
            """Create and return a MyStoreConnector instance."""
            # Access context properties as needed:
            # - context.url: the full remote URL
            # - context.loop: asyncio event loop
            # - context.config: LMCacheEngineConfig
            # - context.metadata: LMCacheMetadata
            # - context.plugin_name: plugin instance name
            #   (e.g. "mystore", "mystore.region_a")
            return MyStoreConnector(
                context.config,
                context.metadata,
                plugin_name=context.plugin_name,
            )

RemoteConnector Implementation
------------------------------
The ``RemoteConnector`` class defines the interface for remote storage operations. Your implementation must provide the following abstract methods:

.. code-block:: python

    from typing import List, Optional
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

    class MyStoreConnector(RemoteConnector):
        """Custom connector for MyStore remote storage."""

        def __init__(
            self,
            config: LMCacheEngineConfig,
            metadata: Optional[LMCacheMetadata]
        ):
            super().__init__(config, metadata)
            # Initialize your connection here

        async def exists(self, key: CacheEngineKey) -> bool:
            """Check if a key exists in the remote store."""
            raise NotImplementedError

        def exists_sync(self, key: CacheEngineKey) -> bool:
            """Synchronous version of exists."""
            raise NotImplementedError

        async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
            """Retrieve a memory object by key. Return None if not found."""
            raise NotImplementedError

        async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
            """Store a memory object with the given key."""
            raise NotImplementedError

        async def list(self) -> List[str]:
            """List all keys in the remote store."""
            raise NotImplementedError

        async def close(self):
            """Close the connection to the remote store."""
            raise NotImplementedError

Optional Methods
----------------
The ``RemoteConnector`` base class also provides optional methods that can be overridden for enhanced functionality:

- ``support_ping()`` / ``ping()``: Health check support
- ``support_batched_get()`` / ``batched_get()``: Batch retrieval operations
- ``support_batched_put()`` / ``batched_put()``: Batch storage operations
- ``support_batched_contains()`` / ``batched_contains()``: Batch existence checks
- ``remove_sync()``: Synchronous key removal

Implementation Example
----------------------
A complete remote storage connector implementation would include both the adapter and connector classes in a single module or package. Here's a minimal working example structure:

.. code-block:: text

    my_connector_package/
    ├── __init__.py
    ├── adapter.py      # Contains MyStoreConnectorAdapter
    └── connector.py    # Contains MyStoreConnector

The adapter module (``adapter.py``):

.. code-block:: python

    from lmcache.v1.storage_backend.connector import (
        ConnectorAdapter,
        ConnectorContext,
        extract_plugin_type,
    )
    from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
    from .connector import MyStoreConnector

    PLUGIN_TYPE = "mystore"

    class MyStoreConnectorAdapter(ConnectorAdapter):
        def __init__(self) -> None:
            super().__init__("mystore://")

        def can_parse(self, url: str) -> bool:
            if url.startswith(self.schema):
                return True
            if url.startswith("plugin://"):
                pname = url[len("plugin://"):]
                return extract_plugin_type(pname) == PLUGIN_TYPE
            return False

        def create_connector(self, context: ConnectorContext) -> RemoteConnector:
            return MyStoreConnector(
                context.config,
                context.metadata,
                plugin_name=context.plugin_name,
            )

Configuration would then reference the adapter:

.. code-block:: yaml

    remote_storage_plugins: ["mystore"]
    extra_config:
      remote_storage_plugin.mystore.module_path: my_connector_package.adapter
      remote_storage_plugin.mystore.class_name: MyStoreConnectorAdapter
