# SPDX-License-Identifier: Apache-2.0

Device-DAX (/dev/dax)
=====================

Overview
--------

The DAX storage plugin maps a ``/dev/dax`` device using ``mmap(MAP_SHARED)``
and uses the mapped region as a fixed-size arena for KV cache chunks.
Typical ``/dev/dax`` devices include persistent memory,
CXL-attached memory, and other byte-addressable memory devices.

Data stored on the DAX device may survive process restarts,
but is not guaranteed to be durable.

KV cache data is stored in the DAX region as part of the backend's storage flow.
Reads copy data back into CPU-backed memory objects.


Configuration
-------------

.. code-block:: yaml

   local_cpu: true
   max_local_cpu_size: 80

   storage_plugins: ["dax"]
   extra_config:
     storage_plugin.dax.module_path: lmcache.v1.storage_backend.plugins.dax_backend
     storage_plugin.dax.class_name: DaxBackend

     dax.device_path: "/dev/dax1.0"
     dax.max_dax_size: 100
     dax.restore_workers: 8
     dax.restore_max_regions: 8
     dax.retrieve_staging_slab_bytes: 268435456


Multiprocess Mode
-----------------

In LMCache multiprocess mode, Device-DAX is configured as a built-in L2
adapter named ``dax``.  The MP adapter uses the normal L2 adapter
``submit -> event fd -> query`` contract; no vLLM connector protocol changes
are required.

.. code-block:: bash

   lmcache server \
     --l1-size-gb 80 \
     --eviction-policy LRU \
     --l2-adapter '{
       "type": "dax",
       "device_path": "/dev/dax1.0",
       "max_dax_size_gb": 100,
       "slot_bytes": 268435456,
       "num_store_workers": 1,
       "num_lookup_workers": 1,
       "num_load_workers": 4
     }'

The legacy single-device ``--l2-adapter`` JSON accepts these fields:

- ``device_path``: required path to a readable and writable DAX device.
- ``max_dax_size_gb``: required mapped size in GiB. The value must fit within
  the device capacity when capacity can be determined with ``fstat``.
- ``slot_bytes``: required fixed slot size in bytes. It must be large enough
  for one full LMCache chunk.
- ``num_store_workers``: optional store worker count, default ``1``.
- ``num_lookup_workers``: optional lookup worker count, default ``1``.
- ``num_load_workers``: optional load worker count, default
  ``min(4, os.cpu_count())``.
- ``persist_enabled``: accepted by common MP L2 parsing but ignored by
  ``dax`` in this release.

Runtime hotplug uses the multi-device form. The ``devices`` list may also be
empty when ``hotplug_enabled`` is ``true``.

.. code-block:: bash

   lmcache server \
     --l1-size-gb 80 \
     --eviction-policy LRU \
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

MP DAX stores opaque ``ObjectKey`` values in memory and is volatile-only in
this release.  Closing and reopening the server on the same DAX path starts
with an empty index, so previously written bytes are not discoverable after
restart.

MP DAX uses one stable adapter facade per LMCache server.  The facade owns
stable event fds and worker pools, and runtime add/remove/resize only changes
the mapped DAX cores behind that facade.  It does not add kernel-level CXL or
DAX reconfiguration, per-TP DAX partitions, on-device metadata, or restart
recovery. Capacity accounting and eviction are slot-based: a stored object
occupies one slot even if its payload is smaller than ``slot_bytes``.


Runtime Hotplug API
-------------------

Runtime hotplug is disabled unless ``hotplug_enabled`` is ``true``. The API
changes only LMCache runtime mappings and metadata; the ``/dev/dax*`` device
must already exist and be readable and writable by the LMCache server process.
The runtime endpoints are implemented through StorageManager's generic L2
adapter reconfiguration interface, which routes backend, operation name, and
adapter-specific payload to the selected adapter. DAX owns the path, mode,
migration, and resize semantics; the generic interface is reusable by other
adapters such as P2P.
Use JSON bodies because DAX paths contain slashes:

.. code-block:: bash

   curl http://127.0.0.1:9000/reconfigure/dax/status
   curl -X POST http://127.0.0.1:9000/reconfigure/dax/add \
     -H 'Content-Type: application/json' \
     -d '{"device_path": "/dev/daxX.X", "size": "100GiB"}'
   curl -X POST http://127.0.0.1:9000/reconfigure/dax/remove \
     -H 'Content-Type: application/json' \
     -d '{"device_path": "/dev/daxX.X", "mode": "migrate"}'
   curl -X POST http://127.0.0.1:9000/reconfigure/dax/resize \
     -H 'Content-Type: application/json' \
     -d '{"device_path": "/dev/daxX.X", "size": "200GiB"}'

``size`` is required for add and resize. Use an integer byte count or a string
such as ``"100GiB"``. ``remove`` supports these modes:

- ``migrate``: move DAX-resident KV to other active DAX devices before closing
  the source device.
- ``evict``: delete DAX-resident KV on the source device. This is destructive
  for the DAX tier.
- ``drain``: stop new writes to the source device and leave existing KV
  readable until it is evicted or the server closes.

``resize`` supports ``migrate`` and ``evict`` modes. It does not support
``drain`` because resize completes synchronously.

Hotplug operations are lock-safe by default. A remove or shrink that would
delete externally locked or borrowed slots returns ``409 Conflict`` unless
``force`` is set. A migration that has no active destination capacity returns
``507 Insufficient Storage``. Resize grow preserves the in-memory key index and
does not move KV payloads. Resize shrink never silently drops keys; entries
outside the new slot range must migrate first, or the request fails.


Hardware Validation Flow
------------------------

Use the same Qwen 8B or 14B long-context workload before and after a runtime
capacity change. Without hotplug support, ``/reconfigure/dax/status`` and
``/reconfigure/dax/add`` are not available; changing the DAX device set
requires restarting LMCache with a new ``--l2-adapter`` value, which drops the
volatile DAX key index.

.. code-block:: bash

   export MODEL=Qwen/Qwen3-8B  # or a local Qwen 8B/14B checkpoint
   curl http://127.0.0.1:9000/reconfigure/dax/status
   python benchmarks/long_doc_qa/long_doc_qa.py \
     --model "$MODEL" --num-documents 1 --document-length 1024 \
     --output-len 16 --repeat-count 2 --repeat-mode tile \
     --completions --host 127.0.0.1 --port 8000 --json-output
   curl -X POST http://127.0.0.1:9000/reconfigure/dax/add \
     -H 'Content-Type: application/json' \
     -d '{"device_path": "/dev/daxX.X", "size": "100GiB"}'
   curl http://127.0.0.1:9000/reconfigure/dax/status

Record these fields for the comparison:

- ``total_capacity_bytes`` before and after ``/reconfigure/dax/add``.
- ``total_used_bytes`` while the Qwen workload is running.
- Whether an LMCache restart was required.
- Whether the same cached prompt remains retrievable after the capacity change.


Using The Batched Restore Path
------------------------------

The current DAX optimization is a staged batched restore path for retrieval.
It is enabled automatically whenever the DAX backend is configured. No extra
feature flag is required.

The retrieve flow is:

1. Reserve a batched set of readable DAX chunks.
2. Allocate CPU restore buffers from ``LocalCPUBackend``.
3. Copy DAX data into a backend-owned pinned staging slab in coalesced regions.
4. Copy from the staging slab into the final CPU ``MemoryObj`` outputs.
5. Upload those CPU outputs through the normal GPU connector path.

The store flow is unchanged: KV data is still staged through CPU memory before
being written into the DAX arena.

The new DAX tuning knobs control the batched restore path:

- ``dax.restore_workers``: number of persistent worker threads used to execute
  restore regions in parallel.
- ``dax.restore_max_regions``: maximum number of restore regions in one wave.
  Larger values increase parallelism but also increase slab space requirements.
- ``dax.retrieve_staging_slab_bytes``: total size in bytes of the reusable
  pinned retrieve slab. This must be large enough to hold one full chunk per
  configured restore region.

For a first pass, start with:

- ``dax.restore_workers`` equal to the number of CPU workers you want devoted
  to DAX restores
- ``dax.restore_max_regions`` equal to ``dax.restore_workers``
- ``dax.retrieve_staging_slab_bytes`` at least
  ``dax.restore_max_regions * full_chunk_size``, then scale upward if larger
  batched restores are common

If retrieve throughput is low, increase the slab size first, then increase
worker and region counts together. If CPU pressure is high, reduce
``dax.restore_workers`` and ``dax.restore_max_regions``.


Runtime Requirements
--------------------

- ``extra_config['dax.device_path']`` is required and must point to a readable
  and writable DAX device.
- The process must have read-write access to the DAX device
  (e.g., via appropriate permissions or group membership).
- ``LocalCPUBackend`` must be enabled because DAX reads return CPU-backed
  memory objects.


Validation and Current Limits
-----------------------------

- Tensor parallelism is currently limited to TP=1
  (``metadata.world_size == 1``).
- Only single-tensor chunk layouts are supported. Multi-tensor put
  requests are rejected.
- Batched restore uses a backend-owned retrieve staging slab and persistent
  restore executors. The slab and region count can be tuned with
  ``dax.restore_workers``, ``dax.restore_max_regions``, and
  ``dax.retrieve_staging_slab_bytes``.
- Blocking batched restore preserves positional output semantics, while
  asynchronous batched restore returns only the consecutive hit prefix.
