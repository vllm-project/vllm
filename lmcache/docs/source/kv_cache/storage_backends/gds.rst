GDS Backend
==================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


.. _gds-overview:

Overview
--------

This backend will work with any file system, whether local, remote, and remote
with GDS-based optimizations. Remote file systems allow for multiple LMCache
instances to share data seamlessly. The GDS (GPU-Direct Storage) optimizations
are used for zero-copy I/O from GPU memory to storage systems. Supports both
NVIDIA cuFile and AMD hipFile for GPU-direct storage.


Ways to configure LMCache GDS Backend
-----------------------------------------

**1. Environment Variables:**

.. code-block:: bash

    # 256 Tokens per KV Chunk
    export LMCACHE_CHUNK_SIZE=256
    # Path to store files
    export LMCACHE_GDS_PATH="/mnt/gds/cache"
    # GDS Buffer Size in MiB
    export LMCACHE_GDS_BUFFER_SIZE="8192"
    # Disabling CPU RAM offload is sometimes recommended as the
    # CPU can get in the way of GPUDirect operations
    export LMCACHE_LOCAL_CPU=False
    # Disk I/O Threads (default: 4, set to 0 to disable)
    export LMCACHE_EXTRA_CONFIG='{"disk_io_threads": 8}'

**2. Configuration File**:

Passed in through ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml``

Example ``config.yaml``:

.. code-block:: yaml

    # 256 Tokens per KV Chunk
    chunk_size: 256
    # Disable local CPU
    local_cpu: false
    # Path to file system, local, remote or GDS-enabled mount
    gds_path: "/mnt/gds/cache"
    # GDS Buffer Size in MiB
    gds_buffer_size: 8192
    # Disk I/O Threads (default: 4, set to 0 to disable)
    extra_config:
      disk_io_threads: 8


Disk I/O Thread Pool
--------------------

The backend uses a thread pool for parallel disk I/O. The thread pool is
always enabled regardless of whether GDS APIs or POSIX fallback is used,
so both code paths benefit from concurrent reads and writes.

The number of worker threads is controlled by ``extra_config.disk_io_threads``
(default: 4). Increase this on fast NVMe drives for higher parallelism. Set
to ``0`` to disable the thread pool entirely.

**Environment variable:**

.. code-block:: bash

    export LMCACHE_EXTRA_CONFIG='{"disk_io_threads": 8}'

**YAML config:**

.. code-block:: yaml

    extra_config:
      disk_io_threads: 8

.. note::

   The deprecated ``gds_io_threads`` key is no longer supported. Use
   ``disk_io_threads`` instead.


Multi-Path (Multi-Device) Support
---------------------------------

When a system has multiple NVMe drives, you can distribute GDS I/O across them
by specifying a comma-separated list of paths in ``gds_path``. The
``gds_path_sharding`` option controls how each GPU worker selects its path.
Currently only ``"by_gpu"`` is supported (the default), which selects a path
based on the device index (``device_id % num_paths``), so traffic is spread
evenly across the drives without any manual pinning.

**Why this helps:** a single PCIe Gen 4 x4 NVMe tops out at ~7 GB/s. With four
drives the aggregate bandwidth can reach ~28 GB/s, matching what multi-GPU
systems need for KV cache eviction and prefetch.

**Environment variables:**

.. code-block:: bash

    export LMCACHE_GDS_PATH="/mnt/nvme0/cache,/mnt/nvme1/cache,/mnt/nvme2/cache,/mnt/nvme3/cache"
    export LMCACHE_GDS_PATH_SHARDING="by_gpu"

**YAML config:**

.. code-block:: yaml

    gds_path: "/mnt/nvme0/cache,/mnt/nvme1/cache,/mnt/nvme2/cache,/mnt/nvme3/cache"
    gds_path_sharding: "by_gpu"

With the above configuration on a 4-GPU node:

- ``cuda:0`` writes to ``/mnt/nvme0/cache``
- ``cuda:1`` writes to ``/mnt/nvme1/cache``
- ``cuda:2`` writes to ``/mnt/nvme2/cache``
- ``cuda:3`` writes to ``/mnt/nvme3/cache``

If there are more GPUs than paths, the assignment wraps around (e.g. ``cuda:4``
maps back to ``/mnt/nvme0/cache``). A single path (no commas) works exactly as
before.

All directories are created automatically at startup. Every path in the list
must reside on a filesystem that the rest of the GDS configuration expects
(e.g., all paths on GDS-capable mounts when using cuFile).

**Read behavior:** on startup the backend scans **all** configured paths for
previously-stored KV cache entries, regardless of GPU affinity.  This means a
``cuda:0`` worker whose write affinity is ``/mnt/nvme0/cache`` will still
discover entries that were written to ``/mnt/nvme1/cache`` by ``cuda:1`` in a
prior run.  Writes, however, always go to the single affinity-selected path.

.. code-block:: text

   Startup scan (read):   iterate ALL gds_paths → populate hot_cache
   Runtime writes:        only the affinity path  (device_id % num_paths)
   Runtime reads:         look up hot_cache first; on miss, check ALL
                          gds_paths on disk → load from whichever path
                          the entry lives on


GDS Buffer Size Explanation
---------------------------

The backend currently pre-registers buffer space to speed up GDS operations. This buffer space
is registered in VRAM so options like ``--gpu-memory-utilization`` from ``vllm`` should be considered
when setting it. For example, a good rule of thumb for H100 which generally has 80GiBs of VRAM would
be to start with 8GiB and set ``--gpu-memory-utilization 0.85`` and depending on your workflow fine-tune
it from there.


Using AMD hipFile
-----------------

.. note::

   hipFile is alpha software and has been tested on limited hardware.
   For full installation details, see the
   `hipFile install guide <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/INSTALL.md>`__.

**Prerequisites:**

- **ROCm >= 7.2** with ``amdgpu-dkms >= 30.20.1``
  (see the `ROCm quick start installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`__)
- **Supported storage:** local NVMe drives only
- **Supported filesystems:** ext4 (mounted with ``data=ordered``) and xfs
- **Kernel:** ``CONFIG_PCI_P2PDMA`` must be enabled

**Quick install (Ubuntu 24.04):**

.. code-block:: bash

    sudo apt install libmount-dev wget

    # Install nightly hipFile packages
    wget https://github.com/ROCm/hipFile/releases/download/nightly/hipfile_0.2.0.70200-nightly.9999.24.04_amd64.deb
    wget https://github.com/ROCm/hipFile/releases/download/nightly/hipfile-dev_0.2.0.70200-nightly.9999.24.04_amd64.deb
    sudo dpkg -i hipfile-dev_0.2.0.70200-nightly.9999.24.04_amd64.deb hipfile_0.2.0.70200-nightly.9999.24.04_amd64.deb

You can verify that the HIP libraries and kernel support AIS (AMD Infinity Storage) by running:

.. code-block:: bash

    /opt/rocm/bin/ais-check

Successful output will show ``True`` for ``Kernel P2PDMA support``, ``HIP runtime``, and ``amdgpu``.

**LMCache configuration:**

To use AMD hipFile instead of NVIDIA cuFile, set the GDS backend:

**Environment Variables:**

.. code-block:: bash

    export LMCACHE_GDS_BACKEND=hipfile

**Configuration File:**

.. code-block:: yaml

    gds_backend: "hipfile"

Note: The ``gds_buffer_size`` configuration is used for both cuFile and hipFile buffers.


Setup Example
-------------

.. _gds-prerequisites:

**Prerequisites:**

- A Machine with at least one GPU. You can adjust the max model length of your vllm instance depending on your GPU memory.

- A mounted file system. A file system supportings GDS will work best.

- vllm and lmcache installed (:doc:`Installation Guide <../../getting_started/installation>`)

- Hugging Face access to ``meta-llama/Llama-3.1-8B-Instruct``

.. code-block:: bash

    export HF_TOKEN=your_hugging_face_token

**Step 1. Create cache directory under your file system mount:**

To find all the types of file systems supporting GDS in your system, use `gdscheck` from NVIDIA:

.. code-block:: bash

    sudo /usr/local/cuda-*/gds/tools/gdscheck -p

Check with your storage vendor on how to mount the remote file system.

(For example, if you want to use a GDS-enabled NFS driver, try the modified [NFS
stack](https://vastnfs.vastdata.com/), which is an open source driver that
works with any standard [NFS
RDMA](https://datatracker.ietf.org/doc/html/rfc5532) server. More
vendor-specific instructions will be added here in the future).

Create a directory under the file systew mount (the name here is arbitrary):

.. code-block:: bash

    mkdir /mnt/gds/cache

**Step 2. Start a vLLM server with file backend enabled:**

Create a an lmcache configuration file called: ``gds-backend.yaml``

.. code-block:: yaml

    local_cpu: false
    chunk_size: 256
    gds_path: "/mnt/gds/cache"
    gds_buffer_size: 8192
    extra_config:
      disk_io_threads: 8

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # LMCACHE_LOCAL_CPU=False \
    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_GDS_PATH="/mnt/gds/cache" \
    # LMCACHE_GDS_BUFFER_SIZE=8192 \
    # LMCACHE_EXTRA_CONFIG='{"disk_io_threads": 8}' \
    LMCACHE_CONFIG_FILE="gds-backend.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 65536 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'


POSIX fallback
--------------

In some cases, libcufile implements its own internal POSIX fallback without `GdsBackend` being aware.
In others, an error such as `RuntimeError: cuFileHandleRegister failed (cuFile err=5030, cuda_err=0)` may be throwned.
Thus, backend can be configured to fallback to its own POSIX implementation when the usage of the GDS APIs is not successful.

To force `GdsBackend` not use GDS APIs for any reason, you can override its behavior via configuration:

.. code-block:: yaml

    use_gds: false

Or via environment variable:

.. code-block:: bash

    LMCACHE_USE_GDS=False

The ``gds_backend`` field (default: ``cufile``) selects which GDS library to use. Supported
backends are ``cufile`` (NVIDIA cuFile) and ``hipfile`` (AMD hipFile):

.. code-block:: yaml

    use_gds: true
    gds_backend: "cufile"   # or "hipfile"

Note that under this mode it would still use CUDA APIs to map and do operations the pre-registered GPU memory.
