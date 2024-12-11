.. _torch_compile:

Tuning ``torch.compile``
========================

vLLM provides several options for users to optimize the performance of their models using ``torch.compile``. As the integration progresses, more flags will become available. Currently, there are two main flags that users can utilize:

- ``level``: Set this to 3 for optimal performance. Other levels are intended for internal testing and debugging.
- ``candidate_compile_sizes``: A list of batch sizes for which the user wants to compile the model. Only the batch sizes that are part of cudagraph capture sizes will be compiled.

To effectively use ``torch.compile``, the TL;DR; is:

- Ensure GPUs are busy executing the model before enabling ``torch.compile``.
- Profile your workload to identify the most common batch sizes.
- Compile the model for these batch sizes and, if using Docker or Kubernetes for deployment, carry the compilation cache to the deployment environment.
- Benefit from the performance improvements.

Usage
-----

When using vLLM from command line, the ``torch.compile`` flags can be specified via the command line argument ``-O`` or ``--compilation-config``. We use ``-O`` to mimic the behavior of traditional C/C++ compilers' optimization level control. The flag value should be a simple integer (which means the compilation level), or a string form of a Python dict containing the values. For example:

.. code-block:: bash

    $ # Compile the model with level 3, for a general shape
    $ vllm serve model -O 3
    $ # Compile the model with level 3, for a general shape plus batchsize 1
    $ vllm serve model -O "{'level': 3, 'candidate_compile_sizes': [1,]}"
    $ # Compile the model with level 3, for a general shape plus batchsize 1, 2, 4, 8
    $ # Note that candidate_compile_sizes from cli are 1, 2, 3, 4, 5, 6, 7, 8,
    $ # but the final compiled sizes are 1, 2, 4, 8 because only the ones in cudagraph capture sizes will be compiled.
    $ vllm serve model -O "{'level': 3, 'candidate_compile_sizes': [$(seq -s, 1 1 8)]}"

When passing the full config from command line, it is recommended to use double quotes for the outermost quotes, and single quotes for the keys of the dict, so that you can use shell commands or environment variables in the value part. This is because shell will not expand variables inside single quotes.

When using vLLM from Python, the ``torch.compile`` flags can be specified via the ``compilation_config`` argument of the ``LLM`` class. For example:

.. code-block:: python

    from vllm import LLM

    # Compile the model with level 3, for a general shape
    llm = LLM(
        model="model",
        compilation_config=3,
    )

    # Compile the model with level 3, for a general shape plus batchsize 1
    llm = LLM(
        model="model",
        compilation_config={"level": 3, "candidate_compile_sizes": [1]},
    )

    # Compile the model with level 3, for a general shape plus batchsize 1, 2, 4, 8
    llm = LLM(
        model="model",
        compilation_config={"level": 3, "candidate_compile_sizes": list(range(1, 9))},
    )

Understanding the compilation time
----------------------------------

It takes time to compile the model to get the speedups. However, vLLM guarantees that all compilation happens before we serve any requests. No compilation will happen during the serving time. To understand the time taken for compilation, you can check the logs of the vLLM server, e.g. ``init engine (profile, create kv cache, warmup model) took 14.18 seconds``. This is the time taken to start the engine. Compare it with and without the compilation, and you will get an idea of how much time the compilation took.

The breakdown of the compilation time is as follows:

- **Dynamo bytecode transform**: This is the time taken for TorchDynamo to analyze the Python code. Check the logs for ``Dynamo bytecode transform time: 4.60 s``.
- **Inductor graph compilation**: This is the time taken for the inductor to compile the computation graph into triton kernels. It includes two parts: compilation for a general shape and compilation for specific shapes. Check the logs for ``Compiling a graph for general shape takes 14.77 s`` and ``Compiling a graph for shape 1 takes 13.52 s``.
- **Triton kernel compilation**: This is the time taken for Triton to compile the triton kernels into GPU kernels. No logs are available for this part.

There will also be a log ``torch.compile takes 19.37 s in total``, which is the sum of Dynamo bytecode transform time, inductor graph compilation time. If you want to know the triton kernel compilation time, you need to use the increase of ``init engine`` time as a reference, and then subtract the Dynamo bytecode transform time and inductor graph compilation time.

For example:

.. code-block:: bash

    $ vllm serve meta-llama/Meta-Llama-3-8B
    init engine (profile, create kv cache, warmup model) took 14.18 seconds
    $ vllm serve meta-llama/Meta-Llama-3-8B -O3
    Dynamo bytecode transform time: 4.60 s
    Compiling a graph for general shape takes 14.77 s
    torch.compile takes 19.37 s in total
    init engine (profile, create kv cache, warmup model) took 39.34 seconds

In this example, the increase of ``init engine`` time is 25.16 seconds. The triton kernel compilation time is 25.16 - 4.60 - 14.77 = 5.79 seconds.

Exploiting the compilation cache
---------------------------------

When you first compile for a specific shape, e.g. via ``-O "{'level': 3, 'candidate_compile_sizes': [1]}"``, the compilation for batchsize 1 will take some time because Inductor will run autotuning to find the best kernel for this shape. The result of the autotuning will be saved in the inductor compilation cache. By default the location is the system temp directory under ``torchinductor_<username>``, and you can also set ``TORCHINDUCTOR_CACHE_DIR`` environment variable to change the location. Check `PyTorch documentation <https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html#torchinductor-cache-dir>`_ for more information.

The second time you compile for the same shape, the autotuning will be skipped and the result will be loaded from the cache. This will save a lot of compilation time.

Profiling the workload
----------------------

We find that ``torch.compile`` mainly helps with the performance of the model for fixed shapes. Since it takes time to compile every shapes, it is recommended to profile the workload and find the most common shapes. Then you can compile the model for these shapes to get the best performance.

For example, when we run ``python benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1``, we know the main workload is batchsize 1. We can compile the model for batchsize 1 to get the best performance, without wasting time on compiling for other shapes:

.. code-block:: bash

    $ python benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1
    Avg latency: 0.9704469823899369 seconds
    $ python benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 -O "{'level': 3, 'candidate_compile_sizes': [1]}"
    Avg latency: 0.8950413154981409 seconds

The end-to-end latency (the smaller the better) is reduced from 0.9704 seconds to 0.8950 seconds (7.7% improvement), with the help of ``torch.compile``.

For a dynamic workload, we can use the ``VLLM_LOG_BATCHSIZE_INTERVAL`` environment variable to monitor the batchsize distribution:

.. code-block:: bash

    $ # run the baseline setting
    $ python benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --model meta-llama/Meta-Llama-3-8B --num-scheduler-steps 64
    Throughput: 44.39 requests/s, 22728.17 total tokens/s, 11364.08 output tokens/s
    $ # run the same setting with profiling
    $ VLLM_LOG_BATCHSIZE_INTERVAL=1.0 python benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --model meta-llama/Meta-Llama-3-8B --num-scheduler-steps 64
    INFO 12-10 15:42:47 forward_context.py:58] Batchsize distribution (batchsize, count): [(256, 769), (232, 215), ...]
    $ # the most common batchsizes are 256 and 232, so we can compile the model for these two batchsizes
    $ python benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --model meta-llama/Meta-Llama-3-8B --num-scheduler-steps 64 -O "{'level': 3, 'candidate_compile_sizes': [232, 256]}"
    Throughput: 46.11 requests/s, 23606.51 total tokens/s, 11803.26 output tokens/s

The end-to-end throughput (the larger the better) is increased from 44.39 requests/s to 46.11 requests/s (3.9% improvement), with the help of ``torch.compile``.

Note that ``torch.compile`` only helps to accelerate the model forwarding. To see the benefit, please make sure GPUs are already busy executing the model, otherwise the benefit will be hidden because GPUs are idle. That's why we have added ``--num-scheduler-steps 64`` to the command line arguments.

Supported Models
----------------

Most models in vLLM are supported by ``torch.compile``. If a model is not supported, but you turn on ``torch.compile``, you will see a warning like ``torch.compile is turned on, but the model does not support it`` , and the ``torch.compile`` configs will be ignored. If you want to get this model supported, please file an issue.

Feature Compatibility
---------------------

Most features in vLLM are compatible with ``torch.compile``, including tensor parallel, pipeline parallel, quantization, etc. There are two features that are not compatible with ``torch.compile``:

- **CPU offloading**: It is not compatible with ``torch.compile`` right now, but should be compatible in the future. Check `this issue <https://github.com/vllm-project/vllm/issues/10612>`__ for more information.
- **Lora serving**: It can be made compatible with ``torch.compile``, but the benefit would be minimal. Check `this issue <https://github.com/vllm-project/vllm/issues/10617>`__ for more information.
