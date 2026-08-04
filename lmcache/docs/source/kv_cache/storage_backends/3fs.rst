3FS
====

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


.. _3FS-overview:

Overview
--------

3FS (Fire-Flyer File System) is an AI native distributed file-system which provides high performance and 
low latency. It is a supported option for KV Cache offloading in LMCache. Even though the FSConnector 
backend can work with 3FS storage cluster, but it access data in 3FS storage cluster by FUSE interfaces.
It can't leverage 3FS high performance. This particular backend uses 3FS native USRBIO(User Space Ring 
Based IO) interfaces to access 3FS storage cluster which get high performance.

Configure LMCache 3FS Offloading
-----------------------------------------

Passed in through ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml``

Example ``config.yaml``:

.. code-block:: yaml

    # 256 Tokens per KV Chunk
    chunk_size: 256
    local_cpu: False
    
    # Plugin name mode, obtain base_path in extra_config session
    remote_storage_plugins: ["hf3fs.primary"]
    
    # URL mode, obtain base_path from the URL
    #remote_url: "hf3fs:///3fs/stage/hello, /3fs/stage/world"

    extra_config:

        # base_path, for a plugin instance
        remote_storage_plugin.hf3fs.primary.base_path: "/3fs/stage/dir1,/3fs/stage/dir2"
        
        # base_path, for all plugin instances 
        #hf3fs_base_path: "/3fs/stage/dir1,/3fs/stage/dir2"

        # Mount point of 3FS
        hf3fs_mount_point: "/3fs/stage"

        # Shared memory size for Iov in hf3fs client, 
        # range in [104857600(100MB), 2147483648(2GB)], default:209715200
        hf3fs_iov_size: 209715200 #200MB

        # Max num of concurrent requests that can be submitted in Ior
        # range in [128,1024], default: 256
        hf3fs_ior_entries: 256

        # Control with I/O depth. 0, no control
        # >0, only when io_depth requests are in queue, and issue them in one batch
        # <0, wait for at most -io_depth requests are in queue and issue them in one batch
        # range in [-128, 128], default: 0
        hf3fs_io_depth: 0

        # NUMA ID for Ior shared memory, -1 for current process NUMA ID.
        hf3fs_numa_id: -1

        # Number of io thread
        # range in [2,16], default: 4
        hf3fs_io_thread_num: 4

There are 2 methods to config hf3fs remote backend:
 1. Plugin name mode, uses the parameter remote_storage_plugins(Recommend)
 2. URL mode, uses the parameter remote_url(Deprecated, will be removed in a future)

For URL mode, the base_path is contained in the url. For plugin name mode, there are 2 methods to set base_path:
 1. remote_storage_plugin.{plugin name}.base_path, it sets the base_path for a plugin instance.
     e.g.:remote_storage_plugin.hf3fs.primary.base_path
 2. hf3fs_base_path, it set the base_path for all plugin instances

Installation
-------------

.. _3FS-prerequisites:

**Prerequisites:**

- A Machine with at least one GPU. You can adjust the max model length of your vllm instance depending on your GPU memory.

- vllm and lmcache installed

**Step 1. Install 3FS hf3fs_py_usrbio package**

    The inference server need install 3FS hf3fs_py_usrbio package, recommend to build the package from source:

    .. code-block:: bash

        git clone https://github.com/deepseek-ai/3fs
        cd 3fs
        git submodule update --init --recursive
        ./patches/apply.sh
        Install dependencies
        pip install -e .

    `3FS Build <https://github.com/deepseek-ai/3FS/blob/main/README.md#build-3fs>`_


**Step 2. Setup 3FS storage cluster**

    `3FS Setup <https://github.com/deepseek-ai/3FS/blob/main/deploy/README.md>`_


**Step 3. Deploy 3FS FUSE client in inference server**

    The inference server must deploy 3FS FUSE client(a fuse daemon process provided by 3FS), 
    otherwise, it can't access 3FS storage cluster

    `Setup 3FS FUSE Client <https://github.com/deepseek-ai/3FS/blob/main/deploy/README.md#step-8-fuse-client>`_


**Step 4. Start a vLLM server with 3FS offloading enabled**

    Create a lmcache configuration file called: ``3fs-offload.yaml``

    .. code-block:: yaml

        # 256 Tokens per KV Chunk
        chunk_size: 256
        local_cpu: False
        # support multiple paths
        remote_storage_plugins: ["hf3fs.primary"]

        extra_config:
            # base_path
            remote_storage_plugin.hf3fs.primary.base_path: "/3fs/stage/dir1,/3fs/stage/dir2"

            # Mount point of 3FS
            hf3fs_mount_point: "/3fs/stage"

            # Shared memory size for Iov in hf3fs client, 
            # range in [104857600(100MB), 2147483648(2GB)], default:209715200 (200MB)
            hf3fs_iov_size: 209715200 

            # Max num of concurrent requests that can be submitted in Ior
            # range in [128,1024], default: 256
            hf3fs_ior_entries: 256

            # Control with I/O depth. 0, no control
            # >0, only when io_depth requests are in queue, and issue them in one batch
            # <0, wait for at most -io_depth requests are in queue and issue them in one batch
            # range in [-128, 128], default: 0
            hf3fs_io_depth: 0

            # NUMA ID for Ior shared memory, -1 for current process NUMA ID.
            hf3fs_numa_id: -1

            # Number of io thread
            # range in [2,16], default: 4
            hf3fs_io_thread_num: 4

    Start vllm:

    .. code-block:: bash

        export VLLM_USE_V1=0
        export LMCACHE_USE_EXPERIMENTAL=True
        export LMCACHE_LOG_LEVEL=INFO
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export VLLM_ENABLE_V1_MULTIPROCESSING=1
        LMCACHE_CONFIG_FILE="3fs-offload.yaml" \
        vllm serve \
            meta-llama/Llama-3.1-8B-Instruct \
            --max-model-len 65536 \
            --kv-transfer-config \
            '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
