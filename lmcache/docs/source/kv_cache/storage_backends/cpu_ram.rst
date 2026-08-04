CPU RAM
=======

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


.. _cpu_ram-overview:

Overview
--------

CPU RAM and Local Storage are the two ways of offloading KV cache onto non-GPU
memory of the same machine that is running inference.

Two ways to configure LMCache CPU Offloading:
---------------------------------------------

**1. Environment Variables:**

.. code-block:: bash

    # 256 Tokens per KV Chunk
    export LMCACHE_CHUNK_SIZE=256
    # Enable CPU memory backend
    export LMCACHE_LOCAL_CPU=True # default
    # 5GB of Pinned CPU memory
    export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0 # default

**2. Configuration File**:

Passed in through ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml``

Example ``config.yaml``:

.. code-block:: yaml

    # 256 Tokens per KV Chunk
    chunk_size: 256
    # Enable CPU memory backend
    local_cpu: true # default
    # 5GB of Pinned CPU memory
    max_local_cpu_size: 5.0 # default

CPU RAM Explanation:
---------------------

The ``LMCACHE_MAX_LOCAL_CPU_SIZE`` is the amount of page-locked (for fast GPU transfer)
CPU memory that LMCache will reserve and must be set to a number greater than 0 since
local and remote backends also use CPU RAM as an intermediate buffer when transferring KV caches
with the GPU. This means it is possible to set ``LMCACHE_LOCAL_CPU=False`` even
though ``LMCACHE_MAX_LOCAL_CPU_SIZE`` is set to a non-zero number.


However, it is recommended to *always* set ``LMCACHE_LOCAL_CPU=True`` (the default is ``True`` so if you
don't specify, CPU offloading will automatically be enabled) since this allows all currently unused pinned CPU RAM that
LMCache has reserved to hold KV caches. When the pinned CPU RAM is required for any disk or remote transfers, the CPU KV caches will be LRU evicted to make
space so there is no danger of running out of pinned CPU RAM.

When ``LMCACHE_LOCAL_CPU=True`` is used in conjunction with the disk backend or
a remote backend (:doc:`Redis <./redis>`, :doc:`Mooncake <./mooncake>`, :doc:`Valkey <./valkey>`,
or :doc:`Infinistore <./infinistore>`), we can think of the CPU RAM as a "hot cache" that
will contain the "hottest" (most recently accessed)subset of KV caches from Disk and Remote storage.

Thus, the cache engine also has a **prefetch** mechanism to preload the KV caches for specified
tokens into the pinned CPU RAM from the disk or remote storage (*if* the KV caches for these
tokens are already stored there). This can preemptively avoid the latency of the disk and
remote KV transfer if we predict these tokens will be requested soon (e.g. structured or agentic workflows).

.. _cpu_ram-hugepage-support:

Hugepage Support
-----------------

By default LMCache allocates CPU-pinned memory using regular 4 KiB pages.
For large KV cache buffers (multiple gigabytes), enabling **Linux hugepages**
(2 MiB pages) can reduce TLB (Translation Lookaside Buffer) pressure and
improve memory access performance.

**System prerequisite**

Hugepages must be pre-allocated at the OS level before LMCache starts.
TO find the number of pages needed, divide the desired buffer size by 2 MiB and round up.
For example, 5 GB requires at least 2560 pages:

.. code-block:: bash

    # Allocate 2560 hugepages (5 GB)
    sudo sysctl -w vm.nr_hugepages=2560

    # Make persistent across reboots
    echo 'vm.nr_hugepages=2560' | sudo tee -a /etc/sysctl.conf

Verify that pages are available:

.. code-block:: bash

    grep HugePages /proc/meminfo
    # HugePages_Total:    2560
    # HugePages_Free:     2560

**Configuration**

.. code-block:: yaml

    local_cpu_use_hugepages: true

Or via environment variable:

.. code-block:: bash

    export LMCACHE_LOCAL_CPU_USE_HUGEPAGES=true

**Restrictions**

- Hugepages are **not compatible with P2P mode** (``enable_p2p: true``).
- Hugepages are **not compatible with shared memory** (``shm_name`` is set).
- On non-CUDA platforms, hugepages are not supported. Regular allocation will be used as fallback.

.. _cpu_ram-online-inference-example:

Online Inference Example
------------------------

Let's feel the TTFT (time to first token) differential!

.. _cpu_ram-prerequisites:

**Prerequisites:**

- A Machine with at least one GPU. Adjust the max model length of your vllm instance depending on your GPU memory and the long context you want to use.

- vllm and lmcache installed (:doc:`Installation Guide <../../getting_started/installation>`)

- Hugging Face access to ``meta-llama/Meta-Llama-3.1-8B-Instruct``

.. code-block:: bash

    export HF_TOKEN=your_hugging_face_token

- A few packages:

.. code-block:: bash

    pip install openai transformers

**Step 0. Set up a directory for this example:**

.. code-block:: bash

    mkdir lmcache-cpu-ram-example
    cd lmcache-cpu-ram-example

**Step 1. Prepare a long context!**

We want a context long enough that vllm's prefix caching will not be able to hold the KV caches in
GPU memory and LMCache is necessary to keep KV caches in non-GPU memory:

.. code-block:: bash

    # 382757 bytes
    man bash > man-bash.txt

**Step 2. Start a vLLM server with CPU offloading enabled:**

Create a an lmcache configuration file called: ``cpu-offload.yaml``

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5.0

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_LOCAL_CPU=True \
    # LMCACHE_MAX_LOCAL_CPU_SIZE=5.0 \
    LMCACHE_CONFIG_FILE="cpu-offload.yaml" \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 16384 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

- ``--kv-transfer-config``: This is the parameter that actually tells vLLM to use LMCache for KV cache offloading.
    - ``kv_connector``: Specifies the LMCache connector for vLLM V1
    - ``kv_role``: Set to "kv_both" for both storing and loading KV cache (important because we will run two queries and the first will produce/store a KV cache while the second will consume/load that KV cache)

**Step 3. Query TTFT improvements with LMCache:**

Once the Open AI compatible server is running on default vllm port 8000, let's query it twice with the same long context!

Create a script called ``query-twice.py`` and paste the following code:

.. code-block:: python

    import time
    from openai import OpenAI
    from transformers import AutoTokenizer

    client = OpenAI(
        api_key="dummy-key",  # required by OpenAI client even for local servers
        base_url="http://localhost:8000/v1"
    )

    models = client.models.list()
    model = models.data[0].id

    # 119512 characters total
    # 26054 tokens total
    long_context = ""
    with open("man-bash.txt", "r") as f:
        long_context = f.read()

    # a truncation of the long context for the --max-model-len 16384
    # if you increase the --max-model-len, you can decrease the truncation i.e.
    # use more of the long context
    long_context = long_context[:70000]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    question = "Summarize bash in 2 sentences."

    prompt = f"{long_context}\n\n{question}"

    print(f"Number of tokens in prompt: {len(tokenizer.encode(prompt))}")

    def query_and_measure_ttft():
        start = time.perf_counter()
        ttft = None

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
            stream=True,
        )

        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if ttft is None:
                    ttft = time.perf_counter()
                print(chunk_message, end="", flush=True)

        print("\n")  # New line after streaming
        return ttft - start

    print("Querying vLLM server with cold LMCache CPU Offload")
    cold_ttft = query_and_measure_ttft()
    print(f"Cold TTFT: {cold_ttft:.3f} seconds")

    print("\nQuerying vLLM server with warm LMCache CPU Offload")
    warm_ttft = query_and_measure_ttft()
    print(f"Warm TTFT: {warm_ttft:.3f} seconds")

    print(f"\nTTFT Improvement: {(cold_ttft - warm_ttft):.3f} seconds \
        ({(cold_ttft/warm_ttft):.1f}x faster)")

Then run:

.. code-block:: bash

    python query-twice.py

Since we're in streaming mode, you'll be able to feel the TTFT differential in
real time!

**Example Output:**

.. code-block:: text

    Number of tokens in prompt: 15376
    Querying vLLM server with cold LMCache
    Bash is a Unix shell and command-line interpreter that executes commands read
    from the standard input or from a file, incorporating features from the Korn
    and C shells. It is an sh-compatible command language interpreter that can be
    configured to be POSIX-conformant by default and is intended to be a conformant
    implementation of the Shell and Utilities portion of the IEEE POSIX specification.

    Cold TTFT: 6.537 seconds

    Querying vLLM server with warm LMCache
    Bash is a Unix shell and command-line interpreter that eead from the standard
    input or from a file, incorporatinhe Korn and C shells. It is intended to be a
    conformant tation of the IEEE POSIX specification and can be configured to be
    POSIX-conformant by default, with options for setting the shell's behavior and
    interacting with the user.

    Warm TTFT: 0.147 seconds

    TTFT Improvement: 6.390 seconds (44.5x faster)

If you look at the logs of your vLLM server, you should see (the logs are truncated for cleanliness):

.. code-block:: text

    # Cold LMCache Miss and then Store

    LMCache INFO: Reqid: chatcmpl-8676f9b9ebf04c79a5d47b9ada7b65fd, Total tokens 15410,
    LMCache hit tokens: 0, need to load: 0

    # you should see 8 of these storing logs total
    # 2048 tokens is a multiple of the chunk size
    LMCache INFO: Storing KV cache for 2048 out of 12288 tokens for request
    chatcmpl-8676f9b9ebf04c79a5d47b9ada7b65fd

    LMCache INFO: Storing KV cache for 2048 out of 14336 tokens for request
    chatcmpl-8676f9b9ebf04c79a5d47b9ada7b65fd

    LMCache INFO: Storing KV cache for 1074 out of 15410 tokens for request
    chatcmpl-8676f9b9ebf04c79a5d47b9ada7b65fd

    # Warm LMCache Hit!!

    LMCache INFO: Reqid: chatcmpl-136d9dac1ba94bd4b4ae85007e8ad437, Total tokens 15410,
    LMCache hit tokens: 15409, need to load: 1

.. _cpu_ram-tips:

Tips:
-----

- If you want to run the ``query-twice.py`` script multiple times, you'll need to either restart the vLLM LMCache server or change the prefix of the context you pass in since you've already warmed LMCache.

- The max model length here was decided by running an L4 with only 23GB of GPU memory. If you have more memory, you can increase the max model length and modify ``query-twice.py`` to use more of the long context. LMCache TTFT improvement becomes more pronounced as the context length increases!
