Secondary KV Storage
====================

LMCache multiprocess mode supports a two-tier storage architecture:

- **L1 (fast tier)** -- CPU memory by default, or an NVMe slab via GPUDirect
  Storage (cuFile) when ``--gds-l1-path`` is set, managed by the L1 Manager.
  All KV cache chunks live here during active use. (Byte-array L2 adapters are
  unsupported under the GDS L1 tier, which exposes no L1 memory buffer.)
- **L2 (persistent)** -- Durable storage backends (NIXL-based or plain
  file-system/raw-block).  The StoreController asynchronously pushes data from L1
  to L2, and the PrefetchController loads data from L2 back into L1 on
  cache misses.

.. contents::
   :local:
   :depth: 2

Data Flow
---------

**Write path (L1 -> L2):**

1. vLLM stores KV cache chunks into L1 via the ``STORE`` RPC.
2. The ``StoreController`` detects new objects (via eventfd) and
   asynchronously submits store tasks to each configured L2 adapter.
3. The L2 adapter writes the data to its backend (e.g., local SSD via GDS).

**Read path (L2 -> L1):**

1. A ``LOOKUP`` RPC checks L1 for prefix hits.
2. For keys not found in L1, the ``PrefetchController`` submits lookup
   requests to L2 adapters.
3. If found in L2, the data is loaded back into L1 and read-locked for the
   pending ``RETRIEVE`` RPC.

Adapter Types
-------------

LMCache ships several L2 storage backends, grouped by medium under
:doc:`Supported Backends <supported_storages>`. Select one or more with the
``--l2-adapter`` flag.

.. toctree::
   :maxdepth: 2

   supported_storages

Multiple Adapters (Cascade)
---------------------------

You can configure multiple L2 adapters by repeating the ``--l2-adapter``
argument.  Adapters are used in the order they are specified.  The
``StoreController`` pushes data to all configured adapters, and the
``PrefetchController`` queries adapters in order during lookups.

.. code-block:: bash

    # SSD (fast, smaller) + NVMe GDS (larger capacity)
    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/ssd/l2", "use_direct_io": "false"}, "pool_size": 64}' \
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"}, "pool_size": 128}'

Store and Prefetch Policies
----------------------------

The **store policy** controls how keys flow from L1 to L2: which adapters
receive each key and whether keys are deleted from L1 after a successful
L2 store.  The **prefetch policy** controls how keys flow from L2 back to
L1: when multiple adapters have the same key, the policy decides which
adapter loads it.

Select policies via CLI:

.. code-block:: bash

    --l2-store-policy default \
    --l2-prefetch-policy default

**Built-in policies:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Flag
     - Name
     - Behaviour
   * - ``--l2-store-policy``
     - ``default``
     - Store all keys to all adapters.  Never delete from L1.
   * - ``--l2-store-policy``
     - ``skip_l1``
     - Buffer-only mode.  Store all keys to all adapters, then
       **delete them from L1** immediately.  Pair with
       ``--eviction-policy noop`` to avoid useless LRU overhead.
   * - ``--l2-prefetch-policy``
     - ``default``
     - For each key, pick the first (lowest-indexed) adapter that has it.
       Prefetched keys are **temporary** (deleted after the reader finishes).
   * - ``--l2-prefetch-policy``
     - ``retain``
     - Same load plan as ``default``, but prefetched keys are **retained**
       permanently in L1.  Useful when prefetched data is likely reused
       by subsequent requests (e.g. shared system-prompt chunks).

Prefetch Concurrency
~~~~~~~~~~~~~~~~~~~~~

The ``--l2-prefetch-max-in-flight`` flag limits the number of concurrent
prefetch requests that the ``PrefetchController`` can have in flight at
any time.  A higher value increases L2-to-L1 throughput but also
increases L1 memory pressure from in-flight data.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--l2-prefetch-max-in-flight``
     - ``8``
     - Maximum number of concurrent prefetch requests.

Buffer-Only Mode
~~~~~~~~~~~~~~~~~

When L1 is used purely as a write buffer (all data lives in L2), use
``--l2-store-policy skip_l1`` together with ``--eviction-policy noop``.
This combination deletes keys from L1 as soon as they are stored to L2
and disables the LRU eviction tracker entirely, reducing memory and CPU
overhead.

.. code-block:: bash

    --eviction-policy noop \
    --l2-store-policy skip_l1 \
    --l2-prefetch-policy default

Policies are extensible -- new policies can be added by creating a file
in ``storage_controllers/`` and calling ``register_store_policy()`` or
``register_prefetch_policy()`` at import time.  See the design doc
``l2_adapters/design_docs/overall.md`` for details.

Serde (compression / quantization)
----------------------------------

Each adapter can optionally run a **serde** (serializer / deserializer)
that transforms data on the way in and out of L2 — e.g. fp8 quantization
for disk backends, or encryption for remote adapters. See
:doc:`KV Cache Compression </mp/serde>` for details and configuration.

.. toctree::
   :maxdepth: 1

   /mp/serde

Eviction
--------

LMCache supports eviction at both storage tiers so that each tier
can operate within a fixed capacity budget.

L1 Eviction
~~~~~~~~~~~

L1 eviction runs a single background thread that monitors overall L1
memory usage. When usage exceeds ``trigger_watermark``, the eviction
policy evicts a fraction of the least-recently-used keys.

**CLI flags:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--eviction-policy``
     - *(required)*
     - Policy name: ``LRU``, ``IsolatedLRU``, or ``noop``. ``IsolatedLRU``
       maintains one LRU list per ``cache_salt`` and relies on per-salt
       quotas registered via the ``/quota`` HTTP endpoints; see
       :doc:`/mp/configuration` for the full description.
   * - ``--eviction-trigger-watermark``
     - ``0.8``
     - L1 usage fraction [0, 1] above which eviction is triggered.
   * - ``--eviction-ratio``
     - ``0.2``
     - Fraction of currently allocated L1 memory to evict per cycle.

**Example:**

.. code-block:: bash

    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2

L2 Eviction
~~~~~~~~~~~

L2 eviction is **per-adapter** and **opt-in**. Each adapter can
independently declare an eviction policy by adding an ``"eviction"``
sub-object to its ``--l2-adapter`` JSON spec. Adapters without an
``"eviction"`` key have no eviction controller.

When L2 eviction is enabled for an adapter, a dedicated background
thread monitors that adapter's ``get_usage()`` value. Once usage
exceeds ``trigger_watermark``, the policy evicts keys until usage
drops by ``eviction_ratio``.

**``"eviction"`` sub-object fields:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``eviction_policy``
     - *(required)*
     - Policy name: ``"LRU"`` or ``"noop"``.
   * - ``trigger_watermark``
     - ``0.8``
     - Adapter usage fraction [0, 1] above which eviction is triggered.
   * - ``eviction_ratio``
     - ``0.2``
     - Fraction of used capacity to evict per cycle.

**Example — nixl_store with LRU eviction:**

.. code-block:: bash

    --l2-adapter '{
      "type": "nixl_store",
      "backend": "POSIX",
      "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"},
      "pool_size": 128,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.8,
        "eviction_ratio": 0.2
      }
    }'

**Adapter support:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Adapter
     - L2 Eviction Support
   * - ``nixl_store``
     - Full support. ``delete`` frees pool slots; pinned keys (in-flight
       loads) are skipped and retried on the next cycle.
   * - ``nixl_store_dynamic``
     - Full support. ``delete`` removes data files from disk; pinned
       keys are skipped. ``get_usage`` is byte-based
       (``_total_bytes / max_capacity_bytes``).
   * - ``mock``
     - Full support. Useful for testing eviction behaviour without
       real storage hardware.
   * - ``raw_block``
     - Full shared/global eviction support. ``delete`` recycles raw-block
       slots; locked entries are skipped and retried on the next cycle.
   * - ``s3``
     - ``delete`` removes objects from the bucket and frees aggregate
       byte accounting. ``get_usage`` reports ``usage_fraction == -1.0``
       when ``max_capacity_gb`` is ``0`` (disabled); set a non-zero
       ``max_capacity_gb`` to enable the watermark-triggered eviction
       controller.
   * - ``hfbucket``
     - ``delete`` removes objects from the bucket and frees aggregate
       byte accounting. ``get_usage`` reports ``usage_fraction == -1.0``
       when ``max_capacity_gb`` is ``0`` (disabled); set a non-zero
       ``max_capacity_gb`` to enable the watermark-triggered eviction
       controller. Locked keys (in-flight loads) are skipped.
   * - ``dax``
     - Full support. ``delete`` removes unlocked keys from the in-memory
       index immediately and recycles fixed slots once active read borrows
       drain. Usage is slot-based.
   * - ``mooncake_store``
     - No eviction support (native connector adapter).
   * - ``fs``
     - No eviction support (``delete`` and ``get_usage`` are no-ops).
   * - native connectors
     - No eviction support.

.. note::

   Each L2 adapter instance gets its own independent eviction
   controller and policy.  Two adapters of the same type can have
   different watermarks or policies.

Combined L1 + L2 Eviction Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    --l1-size-gb 100 \
    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2 \
    --l2-adapter '{
      "type": "nixl_store",
      "backend": "GDS",
      "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"},
      "pool_size": 256,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.1
      }
    }'

In this setup:

- L1 evicts from memory when it is 80 % full, reclaiming 20 % of
  allocated memory per cycle.
- L2 (NIXL/GDS) evicts from the storage pool when 90 % of pool slots
  are occupied, reclaiming 10 % per cycle.
- Both tiers use independent LRU policies, so each evicts its own
  least-recently-used keys.

Verifying L2 Storage
--------------------

Set ``LMCACHE_LOG_LEVEL=DEBUG`` to see L2 activity in the server logs:

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}'

Expected log messages when L2 is active:

.. code-block:: text

    LMCache DEBUG: Submitted store task ...
    LMCache DEBUG: L2 store task N completed ...
    LMCache DEBUG: Prefetch request submitted: X total keys, Y L1 prefix hits, Z remaining for L2
