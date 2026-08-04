DAX
===

An L2 adapter that maps Device-DAX paths, such as ``/dev/daxX.X`` and
``/dev/daxY.Y``, and stores KV cache objects in fixed-size slots. This adapter
is intended for byte-addressable memory devices such as persistent memory or
CXL memory.

The MP ``dax`` adapter is volatile in this release.  It keeps the key index in
server memory and rebuilds an empty index on restart.  Old bytes may remain on
the DAX device, but they are unreachable after the LMCache server restarts.

**Required fields for the legacy single-device form:**

- ``device_path``: Path to the mmap-able DAX device or test file.
- ``max_dax_size_gb``: Number of GiB to map from ``device_path``.
- ``slot_bytes``: Fixed slot size in bytes. This must be large enough for one
  full LMCache chunk because MP memory descriptors do not expose the
  non-MP full-chunk size.

**Required fields for the multi-device form:**

- ``devices``: List of objects with ``device_path`` and ``max_dax_size_gb``.
  The list may be empty only when ``hotplug_enabled`` is ``true``.
- ``slot_bytes``: Fixed slot size in bytes shared by every DAX device in the
  adapter facade.

**Optional fields:**

- ``hotplug_enabled`` (bool, default ``false``): Enables runtime
  ``/reconfigure/dax/status``, ``/reconfigure/dax/add``,
  ``/reconfigure/dax/remove``, and ``/reconfigure/dax/resize``.
- ``num_store_workers`` (int, default ``1``): Store worker threads.
- ``num_lookup_workers`` (int, default ``1``): Lookup worker threads.
- ``num_load_workers`` (int, default ``min(4, os.cpu_count())``): Load worker
  threads.
- ``persist_enabled`` (bool): Accepted by common L2 config parsing but has no
  effect for ``dax`` because restart recovery is not implemented.

**Configuration examples:**

.. code-block:: bash

    # Backward-compatible single-device form.
    --l2-adapter '{
      "type": "dax",
      "device_path": "/dev/dax1.0",
      "max_dax_size_gb": 100,
      "slot_bytes": 268435456,
      "num_store_workers": 1,
      "num_lookup_workers": 1,
      "num_load_workers": 4,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.1
      }
    }'

.. code-block:: bash

    # Multi-device hotplug-ready form.
    --l2-adapter '{
      "type": "dax",
      "devices": [
        {"device_path": "/dev/daxX.X", "max_dax_size_gb": 100},
        {"device_path": "/dev/daxY.Y", "max_dax_size_gb": 100}
      ],
      "slot_bytes": 268435456,
      "hotplug_enabled": true,
      "num_store_workers": 1,
      "num_lookup_workers": 1,
      "num_load_workers": 4
    }'

Runtime management uses JSON bodies because DAX paths contain slashes. See the
:doc:`Device-DAX backend guide </kv_cache/storage_backends/dax>` for complete
examples. These routes use StorageManager's generic L2 adapter reconfiguration
API; the HTTP path selects the backend and operation, the DAX adapter
interprets the operation payload, and the same interface can be reused by
future adapters such as P2P.

.. code-block:: bash

    curl http://127.0.0.1:9000/reconfigure/dax/status
    curl -X POST http://127.0.0.1:9000/reconfigure/dax/add \
      -H 'Content-Type: application/json' \
      -d '{"device_path": "/dev/daxX.X", "size": "100GiB"}'

**Current limits:**

- Runtime hotplug changes only LMCache mappings and metadata. It does not
  create, destroy, or reconfigure kernel CXL or DAX devices.
- Per-TP partitions and on-device restart metadata are not implemented.
- Only single-buffer objects are supported. Multi-tensor objects are rejected.
- Capacity is slot-based, not payload-byte-based. L2 eviction and usage
  metrics count occupied slots.
- Lookups acquire DAX-side external locks. ``submit_unlock`` releases those
  locks after load/retrieve completes, making entries evictable again.
- Remove ``mode="evict"`` is destructive for the DAX tier. Remove
  ``mode="migrate"`` requires enough capacity on another active DAX device.
