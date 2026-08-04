Valkey
======

An L2 adapter backed by the official ``valkey-glide`` **sync** Python
client. Stores KV cache chunks in a Valkey (or Redis) server and
supports both **standalone** (``GlideClient``) and **Valkey-Cluster**
(``GlideClusterClient``) topologies, selected via ``cluster_mode``.

I/O is dispatched through an internal ``ValkeyWorkerPool`` of worker
threads; each worker holds its own glide client so a batch of keys
parallelizes across threads and (in cluster mode) across primaries.

Why the ``valkey`` adapter (vs. the RESP adapter)
-------------------------------------------------

The :doc:`RESP <resp>` adapter targets a single Redis/Valkey server
over the native C++ RESP connector. The ``valkey`` adapter is a
first-class Valkey backend: it speaks ``valkey-glide-sync`` directly,
adds Valkey-Cluster support (slot routing, MOVED/ASK redirects, and
topology discovery are handled inside ``GlideClusterClient``), and
preserves copy-reduced SET/GET on multi-megabyte KV chunks (see the
`Valkey L2 Adapter design doc
<https://github.com/LMCache/LMCache/blob/dev/docs/design/v1/distributed/l2_adapters/valkey.md>`_
for the full rationale, including why the sync -- not async -- glide
client is used).

Prerequisites -- installing the glide sync client
-------------------------------------------------

The ``valkey`` adapter requires the ``valkey-glide-sync`` package
(version ``>= 2.3.0``, the first release with both copy-reduced SET
and buffer GET). It is lazy-imported the first time a Valkey worker
starts:

.. code-block:: bash

    pip install 'valkey-glide-sync>=2.3.0'

If the package is missing when a Valkey worker starts, adapter
construction fails with::

    Valkey support requires the glide_sync module.
    Install: pip install 'valkey-glide-sync>=2.3.0'
    (note: the plain 'valkey-glide' package is async-only)

**Required fields:**

- ``startup_nodes``: A ``"host:port[,host:port...]"`` string of seed
  nodes.

  - In **standalone** mode (``cluster_mode: false``, the default) only
    the first entry is used; extras trigger a warning.
  - In **cluster** mode any reachable seed is enough -- glide auto-
    discovers the full topology.

**Optional fields:**

- ``cluster_mode`` (bool, default ``false``): When ``true`` use
  ``GlideClusterClient``; otherwise use standalone ``GlideClient``.
- ``username`` (str, default ``""``): Optional auth username.
- ``password`` (str, default ``""``): Optional auth password.
- ``key_prefix`` (str, default ``""``): Deployment-level key namespace,
  joined with the standard wire key by ``@``. **Must not contain**
  ``@``. Leave empty for a wire-format identical to the unprefixed
  layout.
- ``num_workers`` (int, default ``8``, > 0): Size of the internal worker
  thread pool. Each worker holds its own glide client, so this is the
  real I/O concurrency knob -- raise it to push throughput on large
  batches / cluster deployments.
- ``tls_enable`` (bool, default ``false``): Enable TLS. Required for
  managed Valkey services such as ElastiCache Serverless.
- ``database_id`` (int, optional): Logical database id for
  **standalone** mode. Ignored (with a warning) when
  ``cluster_mode: true``.
- ``request_timeout`` (float, default ``5.0``, > 0): Per-request timeout,
  in seconds.
- ``connection_timeout`` (float, default ``10.0``, > 0): Initial
  connection timeout, in seconds.
- ``max_capacity_gb`` (float, default ``0``, >= 0): Declared aggregate
  capacity in GB used to drive **LMCache-side** eviction. ``0`` (the
  default and the recommended setting) disables global LMCache
  eviction and defers to the Valkey server's own ``maxmemory`` /
  eviction policy; per-``cache_salt`` quotas configured on the
  coordinator still operate either way. Set ``> 0`` (together with an
  ``eviction`` block) only when a single LMCache writer privately owns
  the cluster -- do **not** combine ``> 0`` with server-side eviction
  or a TTL, both of which drift LMCache's byte accounting.
- ``ttl_seconds`` (int, optional): Opt-in per-key TTL in seconds; must
  be a positive integer when set. Omit (default) to write keys with no
  expiry. Enables server-side time-based expiry, required by any
  ``volatile-*`` eviction policy configured on the Valkey server. Do
  **not** combine with ``max_capacity_gb > 0`` (a warning is logged;
  server-side expiry is not reported back to LMCache's byte accounting).

**Configuration examples:**

.. code-block:: bash

    # Standalone Valkey / Redis server (server-side eviction)
    --l2-adapter '{"type": "valkey", "startup_nodes": "127.0.0.1:6379"}'

    # Valkey Cluster with TLS + auth (managed service)
    --l2-adapter '{
        "type": "valkey",
        "cluster_mode": true,
        "startup_nodes": "seed1.example.com:6379,seed2.example.com:6379",
        "username": "lmcache",
        "password": "secret",
        "tls_enable": true,
        "num_workers": 16
    }'

    # LMCache-driven eviction on a privately-owned Valkey cluster
    --l2-adapter '{
        "type": "valkey",
        "cluster_mode": true,
        "startup_nodes": "10.0.0.1:6379,10.0.0.2:6379",
        "max_capacity_gb": 512,
        "num_workers": 16
    }'

    # Per-key TTL (server-side expiry, no LMCache-side eviction)
    --l2-adapter '{
        "type": "valkey",
        "startup_nodes": "127.0.0.1:6379",
        "ttl_seconds": 3600
    }'

Wire key layout
---------------

Each stored value is keyed by::

    [<key_prefix>@]<model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>[@<cache_salt>]

The ``chunk_hash_hex`` component is a uniform content hash and the wire
key contains no Redis hashtag (``{...}``), so keys distribute evenly
across the 16384 cluster slots -- no manual key sharding is required.
