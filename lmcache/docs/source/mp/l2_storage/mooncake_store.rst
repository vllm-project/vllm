Mooncake Store
==============

An L2 adapter backed by the native C++ Mooncake Store connector.  Uses
`Mooncake <https://github.com/kvcache-ai/Mooncake>`_ for high-performance
distributed KV cache storage with RDMA support.

When Mooncake is configured with ``"protocol": "rdma"``, LMCache must also
have a valid contiguous L1 memory region available.  The distributed storage
manager passes this L1 memory descriptor to the adapter factory automatically
in MP mode.  If the descriptor is missing or invalid, adapter creation fails
with ``ValueError`` instead of silently falling back to a non-RDMA path.

**Prerequisites -- Building with Mooncake support:**

The Mooncake extension is **not** built by default.  You must explicitly
enable it:

.. code-block:: bash

    BUILD_MOONCAKE=1 pip install -e . --verbose

The ``BUILD_MOONCAKE`` environment variable controls compilation:

- ``BUILD_MOONCAKE=1``: Enable the Mooncake C++ extension.
- ``BUILD_MOONCAKE=0``: Force disable (highest priority), even if
  ``MOONCAKE_INCLUDE_DIR`` is set.
- **Not set**: Falls back to checking ``MOONCAKE_INCLUDE_DIR`` for
  backward compatibility.  If ``MOONCAKE_INCLUDE_DIR`` is also unset,
  the extension is skipped.

If the Mooncake headers are not installed in the system include path
(e.g., ``/usr/local/include``), you must point to them explicitly:

.. code-block:: bash

    BUILD_MOONCAKE=1 \
    MOONCAKE_INCLUDE_DIR=/path/to/mooncake/include \
    MOONCAKE_LIB_DIR=/path/to/mooncake/lib \
    pip install -e . --verbose

**LMCache-specific fields:**

- ``num_workers``: Number of C++ worker threads for the shared pool
  (default ``4``, must be > 0).

- ``per_op_workers`` (``dict[str, int]``, optional): A dict mapping lane keys
  to dedicated worker thread counts.  Supported keys:

  - ``"lookup"`` — threads for ``EXISTS`` operations.
  - ``"retrieve"`` — threads for ``GET`` / load operations.
  - ``"store"`` — threads for ``SET`` / put operations.
  - ``"delete"`` — threads for ``DELETE`` operations.

  Operations whose lane key is **not** present in the dict use the
  shared ``num_workers`` pool.  There is no requirement to set all
  keys — you can configure only the lanes that need dedicated pools.

**Mooncake fields:**

All other keys in the JSON config (except ``type``, ``num_workers``,
``per_op_workers``, and ``eviction``) are forwarded **as-is** to Mooncake's
``setup_internal(ConfigDict)``.  Refer to the
`Mooncake documentation <https://github.com/kvcache-ai/Mooncake>`_
for available setup keys (e.g., ``local_hostname``,
``metadata_server``, ``master_server_addr``, ``protocol``,
``rdma_devices``, ``global_segment_size``).

**Configuration example:**

.. code-block:: bash

    # Shared pool (default)
    --l2-adapter '{
      "type": "mooncake_store",
      "num_workers": 4,
      "local_hostname": "node01",
      "metadata_server": "http://localhost:8080/metadata",
      "master_server_addr": "localhost:50051",
      "protocol": "tcp",
      "local_buffer_size": "3221225472",
      "global_segment_size": "3221225472"
    }'

    # Per-operation pools (GET-heavy workload)
    --l2-adapter '{
      "type": "mooncake_store",
      "per_op_workers": {
        "lookup": 2,
        "retrieve": 16,
        "store": 4
      },
      "local_hostname": "node01",
      "metadata_server": "http://localhost:8080/metadata",
      "master_server_addr": "localhost:50051",
      "protocol": "tcp"
    }'

For full Mooncake setup instructions (master service, metadata server,
etc.), see `Mooncake <https://github.com/kvcache-ai/Mooncake>`_ .

**RDMA notes:**

- ``protocol: "rdma"`` requires a valid LMCache L1 memory descriptor.
- When using ``protocol: "rdma"``, it is recommended to disable lazy L1
  allocation with ``--no-l1-use-lazy`` so the L1 buffer is fully allocated
  before Mooncake registers it.
- ``protocol: "tcp"`` does not require L1 preregistration.
- If Mooncake RDMA initialization fails at adapter creation time, verify that
  LMCache L1 memory is enabled and that the descriptor has a non-zero pointer
  and size.
