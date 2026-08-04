FS (native)
===========

A file-system L2 adapter backed by the native C++ ``LMCacheFSClient``
wrapped with ``NativeConnectorL2Adapter``.  I/O is dispatched through a
C++ worker-thread pool with eventfd-driven completions, giving a true
I/O queue depth on a single Python thread.

**Required fields:**

- ``base_path``: Directory for storing KV cache files.

**Optional fields:**

- ``num_workers`` (int, default ``4``, > 0): Number of C++ worker threads
  inside the connector.  This is the real I/O queue depth -- raise to
  push throughput on filesystems whose aggregate BW exceeds per-stream
  BW.
- ``relative_tmp_dir`` (str, default ``""``): Relative sub-directory for
  temporary files during writes (atomic rename on completion).
- ``use_odirect`` (bool, default ``false``): Bypass the page cache via
  ``O_DIRECT``.  Required to measure real disk bandwidth.  See alignment
  caveat below.
- ``read_ahead_size`` (int, optional): Trigger filesystem readahead by
  issuing a warm-up read of this many bytes at open time.
- ``max_capacity_gb`` (float, default ``0``): Maximum L2 capacity in GB
  for client-side usage tracking.  Default ``0`` disables tracking.

.. important::

   ``O_DIRECT`` has two independent alignment requirements:

   1. **Length alignment.**  The transfer length must be a multiple of
      the filesystem's block size.  The connector queries the disk block
      size at construction time and, on each operation, checks
      ``len % disk_block_size``.  If the length is **not** a multiple,
      the connector silently falls back to a buffered open (no
      ``O_DIRECT``) for that operation -- correctness is preserved but
      you do not get true direct I/O.  To ensure ``O_DIRECT`` is
      actually used, choose ``--chunk-size`` so that the resulting
      per-chunk byte size is a multiple of the FS block size.  GPFS and
      similar parallel filesystems often use large blocks (e.g. several
      MiB).

   2. **Memory-buffer alignment.**  The I/O buffer pointer itself must
      also be aligned (typically to 4096 bytes on local disks, or to the
      FS block size on parallel filesystems).  This is controlled by
      ``--l1-align-bytes`` (default ``4096``) -- raise it to match the
      FS block size when running on a filesystem with larger blocks.  If
      the buffer is misaligned, the underlying ``read``/``write`` syscall
      returns ``EINVAL`` (this is **not** caught by the length-fallback
      path above and will surface as a runtime error).

   If unsure, start with ``use_odirect: false`` and confirm correctness
   before enabling ``O_DIRECT``.

**Configuration examples:**

.. code-block:: bash

    # Basic native FS adapter
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2"}'

    # Many worker threads for a parallel filesystem (e.g. GPFS, Lustre)
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32}'

    # O_DIRECT for real-disk benchmarking
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32, "use_odirect": true}'

**Buffer-only mode example.**  L1 acts as a pure write buffer that
absorbs the peak burst of in-flight chunks while the C++ worker pool
drains them to disk; nothing is retained in L1 once a store completes:

.. code-block:: bash

    lmcache server \
        --host 0.0.0.0 --port 5555 \
        --max-workers 32 \
        --l1-size-gb 32 --l1-use-lazy \
        --eviction-policy noop \
        --l2-store-policy skip_l1 \
        --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32, "use_odirect": true}'
