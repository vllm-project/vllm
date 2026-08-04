Filesystem Backend
==================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/fs`.


The filesystem backend uses ``FSConnector`` to store LMCache remote chunks as
files under one or more POSIX filesystem directories. It is useful when you want
a simple persistent remote backend, or when multiple inference workers can see
the same mounted directory through local disk, NFS, a parallel filesystem, or a
container volume.

This backend is different from :doc:`local_storage`. Local disk offloading is a
per-process local tier configured through ``local_disk``. ``FSConnector`` is a
remote backend configured through ``remote_storage_plugins`` or the legacy
``remote_url`` field, so it participates in the same remote backend path as
Redis, S3, Mooncake, and other remote connectors.

When to use it
--------------

Use ``FSConnector`` when:

* You need a lightweight persistent remote backend for development, examples,
  or benchmark runs.
* Multiple LMCache or vLLM processes share a mounted cache directory.
* Your storage is already exposed as a filesystem and does not need a separate
  object-store or key-value service.
* You want to test remote-backend behavior before moving to a production
  backend such as Redis, Valkey, S3, Mooncake, or InfiniStore.

Avoid using it when:

* The filesystem is not shared by every process that must read the cache.
* You need object-store semantics, cross-region persistence, or service-level
  access control.
* The storage path is on a slow network filesystem and sits on the hot request
  path.

Recommended configuration
-------------------------

The recommended form is the built-in remote storage plugin configuration. The
plugin name ``fs`` selects ``FSConnector`` and the base path is configured in
``extra_config``.

.. code-block:: yaml

   chunk_size: 256
   local_cpu: false
   max_local_cpu_size: 1
   save_unfull_chunk: false
   remote_serde: "naive"
   blocking_timeout_secs: 10

   remote_storage_plugins: ["fs"]
   extra_config:
     remote_storage_plugin.fs.base_path: "/tmp/lmcache-fs"
     save_chunk_meta: false

``FSConnector`` creates the base directory if it does not already exist. Each
cache chunk is written as a ``.data`` file whose name is derived from the
LMCache cache key.

Multiple filesystem instances
-----------------------------

You can configure multiple named ``fs`` instances by appending an instance name
after the connector type. The part before the first dot is still the connector
type; the full plugin name becomes the ``extra_config`` prefix.

.. code-block:: yaml

   remote_storage_plugins: ["fs.primary", "fs.backup"]
   extra_config:
     remote_storage_plugin.fs.primary.base_path: "/mnt/cache-primary/lmcache"
     remote_storage_plugin.fs.backup.base_path: "/mnt/cache-backup/lmcache"
     save_chunk_meta: false

This is useful when a deployment wants separate filesystem-backed remote stores
for different cache policies, traffic classes, or experiments.

Multiple base paths
-------------------

``remote_storage_plugin.<name>.base_path`` may contain a comma-separated list of
directories. The connector chooses a directory by hashing the cache chunk key,
which spreads files across the configured paths.

.. code-block:: yaml

   remote_storage_plugins: ["fs"]
   extra_config:
     remote_storage_plugin.fs.base_path: "/mnt/nvme0/lmcache,/mnt/nvme1/lmcache"
     save_chunk_meta: false

Use multiple paths when each path maps to an independent storage device or mount
point. For best results, keep every path visible to the LMCache processes that
need to retrieve the same chunks.

Legacy ``remote_url`` configuration
-----------------------------------

The legacy ``remote_url`` form is still supported. The host and port are parsed
for compatibility with other remote URL formats; ``FSConnector`` uses the path.

.. code-block:: yaml

   chunk_size: 256
   local_cpu: false
   max_local_cpu_size: 1
   save_unfull_chunk: false
   remote_url: "fs://localhost:0/tmp/lmcache-fs"
   remote_serde: "naive"
   blocking_timeout_secs: 10
   extra_config:
     save_chunk_meta: false

Prefer ``remote_storage_plugins`` for new deployments because it also supports
named instances and keeps connector-specific settings grouped by plugin name.

Optional settings
-----------------

The connector reads the following optional settings from ``extra_config``.

``fs_connector_relative_tmp_dir``
   Relative directory used for temporary files before an atomic rename into the
   final chunk path. The value must be relative, not absolute. When omitted,
   temporary files are created next to the final file with a ``.tmp`` suffix.

``fs_connector_read_ahead_size``
   Number of bytes to read first when loading a chunk. If the read fills that
   window, the connector reads the remaining bytes. This can trigger filesystem
   readahead on filesystems that support it.

``fs_connector_use_odirect``
   Enables ``O_DIRECT`` for aligned reads and writes on platforms that expose
   it. The connector falls back to normal I/O when a chunk size is not aligned
   to the filesystem block size. ``O_DIRECT`` is disabled automatically when
   ``save_chunk_meta`` is enabled because the metadata prefix is not block
   aligned.

``fs_base_path``
   Compatibility fallback for plugin mode. Prefer
   ``remote_storage_plugin.<name>.base_path`` so the setting remains scoped to a
   specific plugin instance.

Example with optional settings:

.. code-block:: yaml

   remote_storage_plugins: ["fs"]
   extra_config:
     remote_storage_plugin.fs.base_path: "/data/lmcache-fs"
     fs_connector_relative_tmp_dir: ".tmp"
     fs_connector_read_ahead_size: 1048576
     fs_connector_use_odirect: true
     save_chunk_meta: false

Operational notes
-----------------

* Ensure the LMCache process has permission to create directories and write
  files under every configured base path.
* Put the path on durable storage if cache reuse must survive process restarts.
  Temporary directories such as ``/tmp`` are convenient for tests but may be
  cleaned by the operating system.
* Use the same mounted path for every process that should share cache chunks.
  If one process writes to a private container path, other processes will miss
  those chunks even if they use the same configuration text.
* Leave ``save_chunk_meta`` enabled when workers may infer different metadata
  for the same chunk. Disable it only when you need the lower overhead path and
  the workers share compatible cache metadata.
* For MP mode L2 storage, see :doc:`../../mp/l2_storage/index`, which documents the
  ``fs`` and ``fs_native`` L2 adapters configured through ``--l2-adapter``.

Minimal vLLM usage
------------------

After writing ``fs.yaml`` with one of the configurations above, start vLLM with
LMCache enabled:

.. code-block:: bash

   LMCACHE_CONFIG_FILE=fs.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
       --disable-log-requests

Then send the same long-prefix request twice. The first request stores chunks in
the filesystem backend. The second request should report LMCache hit tokens and
load matching chunks from the configured filesystem path.
