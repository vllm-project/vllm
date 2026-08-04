.. _offload_kv_cache:

Example: Offload KV cache to CPU
================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


In this example, we will show you how to offload KV cache to CPU memory.

.. note::
    Besides CPU memory, LMCache also supports offloading KV cache to many different destinations.
    See :ref:`getting_started/quickstart/offload_kv_cache:Supported offloading destinations` for more details.

Prerequisites
-------------

Before you begin, make sure you have:

- vLLM v1 with LMCache installed (see :doc:`Installation <../installation>`)
- A GPU that can run a LLM


Use CPU offloading in offline inference
---------------------------------------

This section demonstrates how to use CPU memory offloading in offline inference scenarios using LMCache with vLLM.
The example script we use here is available in `vLLM examples <https://github.com/vllm-project/vllm/blob/main/examples/others/lmcache/cpu_offload_lmcache.py>`_.
See the `examples README <https://github.com/vllm-project/vllm/tree/main/examples/others/lmcache#2-cpu-offload-examples>`_ to understand how to run the script for vLLM v1.

First, set up the necessary environment variables for LMCache:

.. code-block:: python

    import os

    # Set token chunk size to 256
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    # Enable CPU memory backend
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    # Set CPU memory limit to 5GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

Next, configure vLLM with LMCache integration:

.. code-block:: python

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    # Configure KV cache transfer to use LMCache
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    # Initialize LLM with LMCache configuration
    # Adjust gpu_memory_utilization based on your GPU memory
    llm = LLM(model="Qwen/Qwen3-8B",
              kv_transfer_config=ktc,
              max_model_len=8000,
              gpu_memory_utilization=0.8)

Now you can run inference with automatic KV cache offloading:

.. code-block:: python

    # Create example prompts with shared prefix
    shared_prompt = "Hello, how are you?" * 1000
    prompts = [
        shared_prompt + "Hello, my name is",
    ]

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    # Run inference
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

When the inference is complete, clean up the LMCache backend:

.. code-block:: python

    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME

    LMCacheEngineBuilder.destroy(ENGINE_NAME)

During inference, LMCache will automatically handle storing and managing KV cache in CPU memory. You can monitor this through the logs, which will show messages like::

    LMCache INFO: Storing KV cache for 6006 out of 6006 tokens for request 0

This indicates that the KV cache has been successfully offloaded to CPU memory.

.. note::
    - Adjust ``gpu_memory_utilization`` based on your GPU's available memory
    - The CPU offloading buffer size can be adjusted through ``LMCACHE_MAX_LOCAL_CPU_SIZE``

Use CPU offloading in online inference
--------------------------------------

This section demonstrates how to use CPU memory offloading in online serving scenarios. 

First, create a configuration file named ``lmcache_config.yaml`` with the following content:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5

.. note::
    LMCache supports extensive configuration through a ``lmcache_config.yaml`` file where you can customize chunk sizes, memory limits, storage backends, and more. We'll cover advanced configuration options in later examples. For now, let's run a minimal example with default configuration.

Launch the vLLM server with LMCache integration using environment variables. Here's an example command:

.. code-block:: bash

    LMCACHE_CONFIG_FILE=lmcache_config.yaml \
    vllm serve \
        Qwen/Qwen3-8B \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1",
          "kv_role":"kv_both"
        }'

Key parameters explained:

- ``LMCACHE_CONFIG_FILE``: Path to the LMCache configuration file.
- ``--kv-transfer-config``: Configures LMCache integration
    - ``kv_connector``: Specifies the LMCache connector 
    - ``kv_role``: Set to "kv_both" for both storing and loading KV cache

Once the server is running, you can send requests to it using curl. Here's an example of how to send a request to the vLLM server with LMCache integration:

.. code-block:: bash

    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen3-8B",
        "prompt": "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "max_tokens": 100,
        "temperature": 0.7
      }'

You should see the following logs:

.. code-block:: text
    :emphasize-lines: 1

    LMCache INFO: Storing KV cache for 31 out of 31 tokens for request cmpl-274bcaa80837444dbf9fbba4155d2620-0 (vllm_v1_adapter.py:497:lmcache.integration.vllm.vllm_v1_adapter)

Once you send the same curl request again, you should see the following logs:

.. code-block:: text
    :emphasize-lines: 1

    LMCache INFO: Reqid: cmpl-4ddf8863a6ac4dc3b6a952f2a107e9b2-0, Total tokens 31, LMCache hit tokens: 30, need to load: 14 (vllm_v1_adapter.py:543:lmcache.integration.vllm.vllm_v1_adapter)


Example: CPU offloading benefits
--------------------------------

This section demonstrates the performance benefits of using CPU offloading with LMCache. We'll use a script that generates multiple prompts and compare the performance with and without LMCache.

Prerequisites (Setup)
~~~~~~~~~~~~~~~~~~~~~~

- A CUDA GPU. The example picks a model that fits the GPU automatically:

  - ``Qwen/Qwen3-8B`` (bf16) when the GPU has ~36 GiB or more (e.g. A100-80G, H100).
  - ``Qwen/Qwen3-8B-FP8`` with ``kv_cache_dtype="fp8"`` when the GPU has ~24 GiB
    and supports native FP8 (Ada Lovelace / Hopper, ``sm_89+``; e.g. L4, L40, RTX 4090).
  - ``Qwen/Qwen3-1.7B`` as the fallback for smaller GPUs (~10 GiB and up),
    including Ampere 24 GiB cards (RTX A5000, RTX 3090) where FP8 is unsupported.

- Sufficient CPU memory. The example clamps the LMCache pinned host buffer to
  fit your system RAM and ``RLIMIT_MEMLOCK`` (``ulimit -l``), so it also works
  on smaller hosts without manual tuning.

Example script
~~~~~~~~~~~~~~

Save the following script as ``cpu-offloading.py``:

.. code-block:: python

    # SPDX-License-Identifier: Apache-2.0
    """
    This file demonstrates the example usage of cpu offloading
    with LMCache in vLLM v1.

    Note that lmcache needs to be installed to run this example.
    Learn more about LMCache in https://github.com/LMCache/LMCache.
    """
    import os
    import torch
    import argparse
    import time
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    def parse_arguments() -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="CPU offloading example with LMCache")
        parser.add_argument("--num-prompts", type=int, default=10,
                          help="Number of prompts to generate (default: 10)")
        parser.add_argument("--num-tokens", type=int, default=10000,
                          help="Number of tokens per prompt (default: 10000)")
        parser.add_argument("--enable-lmcache", action="store_true",
                          help="Enable LMCache for CPU offloading (default: True)")
        return parser.parse_args()

    def pick_cpu_size_gb(workload_gb: float) -> float:
        """
        Clamp the LMCache pinned host buffer to fit system RAM and RLIMIT_MEMLOCK.

        cudaHostAlloc pins pages, so the buffer cannot exceed total RAM nor the
        per-process memlock limit (`ulimit -l`). On hosts where either is small,
        the original "1.5 GB per 10k tokens" formula fails with cudaErrorMemoryAllocation.

        Args:
            workload_gb: Desired buffer size for the workload, in GiB.
        Returns:
            float: A buffer size in GiB that fits both caps, never below 1.0.
        """
        import psutil

        ram_gib = psutil.virtual_memory().total / (1024 ** 3)
        try:
            import resource
            memlock_soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            memlock_gib = (
                float("inf")
                if memlock_soft == resource.RLIM_INFINITY
                else memlock_soft / (1024 ** 3)
            )
        except ImportError:
            # `resource` is POSIX-only; on Windows treat memlock as unbounded.
            memlock_gib = float("inf")
        return max(min(workload_gb, ram_gib * 0.5, memlock_gib * 0.9), 1.0)

    def setup_lmcache_environment(num_prompts: int, num_tokens: int) -> None:
        """
        Configure LMCache environment variables.
        Args:
            num_prompts: Number of prompts to process
            num_tokens: Number of tokens per prompt
        """
        workload_gb = num_prompts * num_tokens * 1.5 / 10000  # 1.5 GB per 10k tokens
        cpu_size = pick_cpu_size_gb(workload_gb)

        env_vars = {
            "LMCACHE_CHUNK_SIZE": "256",         # Set tokens per chunk
            "LMCACHE_LOCAL_CPU": "True",         # Enable local CPU backend
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(cpu_size)  # CPU memory limit (GB)
        }
        for key, value in env_vars.items():
            os.environ[key] = value

    def pick_model_and_kwargs() -> tuple[str, dict]:
        """
        Pick a Qwen model that fits the current GPU's memory and compute capability.

        Tiers:
            - >= 36 GiB                    -> Qwen/Qwen3-8B (bf16)
            - >= 20 GiB and sm >= 89       -> Qwen/Qwen3-8B-FP8 (native FP8)
            - >= 10 GiB                    -> Qwen/Qwen3-1.7B
            - otherwise                    -> RuntimeError

        Returns:
            tuple[str, dict]: (model id, extra kwargs to pass to ``LLM``).
        Raises:
            RuntimeError: If no CUDA GPU is visible or it is too small.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available")

        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        major, minor = torch.cuda.get_device_capability(0)
        sm = major * 10 + minor
        has_fp8 = sm >= 89  # Ada Lovelace / Hopper

        if total_gib >= 36:
            return "Qwen/Qwen3-8B", {}
        if total_gib >= 20 and has_fp8:
            print(f"[fallback] GPU {total_gib:.1f} GiB sm_{sm}: using Qwen3-8B-FP8")
            return "Qwen/Qwen3-8B-FP8", {"kv_cache_dtype": "fp8"}
        if total_gib >= 10:
            print(f"[fallback] GPU {total_gib:.1f} GiB sm_{sm}: using Qwen3-1.7B")
            return "Qwen/Qwen3-1.7B", {}
        raise RuntimeError(
            f"GPU has {total_gib:.1f} GiB; need at least 10 GiB for Qwen3-1.7B"
        )

    def create_test_prompts(num_prompts: int = 10, num_tokens: int = 1000) -> list[str]:
        """
        Create test prompts with index prefix and dummy body.
        Args:
            num_prompts: Number of prompts to generate
            num_tokens: Approximate number of tokens per prompt (using 'Hi ' as token unit)
        Returns:
            list: List of prompts with format '[index] Hi Hi Hi...'
        """
        prompts = []
        dummy_text = "Hi " * num_tokens

        for i in range(num_prompts):
            prompt = f"[Prompt {i}] {dummy_text} how are you?"
            prompts.append(prompt)

        return prompts

    def initialize_llm(max_len: int = 16384, enable_lmcache: bool = True) -> LLM:
        """
        Initialize the LLM with a model auto-selected for the current GPU.
        Args:
            max_len: Maximum sequence length
            enable_lmcache: Whether to wire up the LMCache KV connector
        Returns:
            LLM: Configured LLM instance
        """
        model_name, extra_kwargs = pick_model_and_kwargs()

        ktc = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
        ) if enable_lmcache else None

        return LLM(
            model=model_name,
            kv_transfer_config=ktc,
            max_model_len=max_len,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.9,
            **extra_kwargs,
        )

    def generate_and_print_output(
        llm: LLM,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> float:
        """
        Generate text and print the results.
        Args:
            llm: LLM instance
            prompts: List of input prompts
            sampling_params: Configured sampling parameters
        Returns:
            float: Time taken for generation in seconds
        """
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}")

        return end_time - start_time

    def main() -> None:
        """Main execution function."""
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup environment if LMCache is enabled
        if args.enable_lmcache:
            setup_lmcache_environment(args.num_prompts, args.num_tokens)
        
        # Create prompts and sampling parameters
        prompts = create_test_prompts(num_prompts=args.num_prompts, num_tokens=args.num_tokens)
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
        
        # Initialize model
        llm = initialize_llm(enable_lmcache=args.enable_lmcache)
        
        # First run
        print("\nFirst run:")
        first_run_time = generate_and_print_output(llm, prompts, sampling_params)
        print(f"First run time: {first_run_time:.2f} seconds")
        
        # Second run
        print("\nSecond run:")
        second_run_time = generate_and_print_output(llm, prompts, sampling_params)
        print(f"Second run time: {second_run_time:.2f} seconds")
        
        # Print speedup
        if first_run_time > 0:
            speedup = first_run_time / second_run_time
            print(f"\nSpeedup (first run / second run): {speedup:.2f}x")
        
        # Cleanup if LMCache was enabled
        if args.enable_lmcache:
            LMCacheEngineBuilder.destroy(ENGINE_NAME)

    if __name__ == "__main__":
        main()

Running the Example
~~~~~~~~~~~~~~~~~~~

1. First, run the script without LMCache:

   .. code-block:: bash

       python cpu-offloading.py 

   You'll see output like:

   .. code-block:: text

       Speedup (first run / second run): 1.00x

   Without LMCache, there's no speedup between runs even if vLLM has prefix caching enabled.
   This is because the KV cache exceeds GPU memory and can't be reused.

2. Now, run with LMCache enabled:

   .. code-block:: bash

       python cpu-offloading.py --enable-lmcache

   You'll see output like:

   .. code-block:: text

       Speedup (first run / second run): 7.43x

The significant speedup in the second case demonstrates how LMCache effectively manages KV cache offloading to CPU memory. 
When the total size of KV cache exceeds GPU memory, LMCache allows you to store and reuse the cache from CPU memory, 
resulting in much faster subsequent generations for prompts with shared prefixes.


Supported offloading destinations
---------------------------------

LMCache now supports offloading KV cache to the following destinations:

- :doc:`CPU memory <../../kv_cache/storage_backends/cpu_ram>`
- :doc:`Local file system <../../kv_cache/storage_backends/local_storage>`
- :doc:`Mooncake Storage <../../kv_cache/storage_backends/mooncake>`
- :doc:`InfiniStore <../../kv_cache/storage_backends/infinistore>`
- :doc:`Redis <../../kv_cache/storage_backends/redis>`
- :doc:`ValKey <../../kv_cache/storage_backends/valkey>`

Troubleshooting
---------------

If you encounter the following error:

.. code-block:: text

    (EngineCore_DP0 pid=55437) ERROR 10-04 14:44:47 [core.py:708] RuntimeError: 
    Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method

You can resolve this issue using one of the following methods:

- Set ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` in the environment variables.
- Or update the Python code to guard usage of vllm behind a if ``__name__ == '__main__':`` block.

.. code-block:: python

    if __name__ == '__main__':
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.integration.vllm.utils import ENGINE_NAME
        main()

For details, please refer to the `vLLM Troubleshooting Guide: Python multiprocessing <https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing>`_.