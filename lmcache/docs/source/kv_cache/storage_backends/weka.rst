Weka
====

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


.. _weka-overview:

Overview
--------

WekaFS is a high-performance, distributed filesystem and is a supported option for KV Cache offloading in
LMCache. Even though the local filesystem backend can work with a WekaFS mount, this particular backend is
optimized for Weka's characteristics. It leverages GPUDirect Storage for I/O and it allows data-sharing
between multiple LMCache instances.

Ways to configure LMCache WEKA Offloading
-----------------------------------------

**1. Environment Variables:**

.. code-block:: bash

    # 256 Tokens per KV Chunk
    export LMCACHE_CHUNK_SIZE=256
    # Path to Weka Mount
    export LMCACHE_GDS_PATH="/mnt/weka/cache"
    # GDS Buffer Size in MiB
    export LMCACHE_GDS_BUFFER_SIZE="8192"
    # Disabling CPU RAM offload is sometimes recommended as the
    # CPU can get in the way of GPUDirect operations
    export LMCACHE_LOCAL_CPU=False
    # GDS I/O Threads
    export LMCACHE_EXTRA_CONFIG='{"disk_io_threads": 32}'

**2. Configuration File**:

Passed in through ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml``

Example ``config.yaml``:

.. code-block:: yaml

    # 256 Tokens per KV Chunk
    chunk_size: 256
    # Disable local CPU
    local_cpu: false
    # Path to Weka Mount
    gds_path: "/mnt/weka/cache"
    # GDS Buffer Size in MiB
    gds_buffer_size: 8192
    # GDS I/O Threads
    extra_config:
      disk_io_threads: 32

GDS Buffer Size Explanation
---------------------------

The backend currently pre-registers buffer space to speed up GDS operations. This buffer space
is registered in VRAM so options like ``--gpu-memory-utilization`` from ``vllm`` should be considered
when setting it. For example, a good rule of thumb for H100 which generally has 80GiBs of VRAM would
be to start with 8GiB and set ``--gpu-memory-utilization 0.85`` and depending on your workflow fine-tune
it from there.


Setup Example
-------------

.. _weka-prerequisites:

**Prerequisites:**

- A Machine with at least one GPU. You can adjust the max model length of your vllm instance depending on your GPU memory.

- Weka already installed and mounted.

- vllm and lmcache installed (:doc:`Installation Guide <../../getting_started/installation>`)

- Hugging Face access to ``meta-llama/Llama-3.1-8B-Instruct``

.. code-block:: bash

    export HF_TOKEN=your_hugging_face_token

**Step 1. Create cache directory under your Weka mount:**

To find all your WekaFS mounts run:

.. code-block:: bash

    mount -t wekafs

For the sake of this example let's say that the above returns:

.. code-block:: text

    10.27.1.1/default on /mnt/weka type wekafs (rw,relatime,writecache,inode_bits=auto,readahead_kb=32768,dentry_max_age_positive=1000,dentry_max_age_negative=0,container_name=client)

Then create a directory under it (the name here is arbitrary):

.. code-block:: bash

    mkdir /mnt/weka/cache

**Step 2. Start a vLLM server with Weka offloading enabled:**

Create a an lmcache configuration file called: ``weka-offload.yaml``

.. code-block:: yaml

    local_cpu: false
    chunk_size: 256
    gds_path: "/mnt/weka/cache"
    gds_buffer_size: 8192
    extra_config:
      disk_io_threads: 32

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # LMCACHE_LOCAL_CPU=False \
    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_GDS_PATH="/mnt/weka/cache" \
    # LMCACHE_GDS_BUFFER_SIZE=8192 \
    # LMCACHE_EXTRA_CONFIG='{"disk_io_threads": 32}' \
    LMCACHE_CONFIG_FILE="weka-offload.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 65536 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'


