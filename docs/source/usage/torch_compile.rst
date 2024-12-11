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

.. warning::

    ``torch.compile`` is an experimental feature and the flags and behavior are subject to change. Please refer to the latest documentation for the most up-to-date information.

Usage
-----

When using vLLM from the command line, the ``torch.compile`` flags can be specified via the command line argument ``-O`` or ``--compilation-config``. The ``-O`` flag mimics the behavior of traditional C/C++ compilers' optimization level control. The flag value can be a simple integer (indicating the compilation level) or a string representation of a Python dictionary containing the values. For example:

.. code-block:: bash

    $ # Compile the model with level 3, for a general shape
    $ vllm serve model -O 3

    $ # Compile the model with level 3, for a general shape plus batch size 1
    $ vllm serve model -O "{'level': 3, 'candidate_compile_sizes': [1]}"

    $ # Compile the model with level 3, for a general shape plus batch sizes 1, 2, 4, 8
    $ # Note that candidate_compile_sizes from CLI are 1, 2, 3, 4, 5, 6, 7, 8,
    $ # but the final compiled sizes are 1, 2, 4, 8 because only the ones in cudagraph capture sizes will be compiled.
    $ vllm serve model -O "{'level': 3, 'candidate_compile_sizes': [$(seq -s, 1 1 8)]}"

When passing the full config from the command line, it is recommended to use double quotes for the outermost quotes and single quotes for the keys of the dictionary. This allows you to use shell commands or environment variables in the value part, as the shell will not expand variables inside single quotes.

When using vLLM from Python, the ``torch.compile`` flags can be specified via the ``compilation_config`` argument of the ``LLM`` class. For example:

.. code-block:: python

    from vllm import LLM

    # Compile the model with level 3, for a general shape
    llm = LLM(
        model="model",
        compilation_config=3,
    )

    # Compile the model with level 3, for a general shape plus batch size 1
    llm = LLM(
        model="model",
        compilation_config={"level": 3, "candidate_compile_sizes": [1]},
    )

    # Compile the model with level 3, for a general shape plus batch sizes 1, 2, 4, 8
    llm = LLM(
        model="model",
        compilation_config={"level": 3, "candidate_compile_sizes": list(range(1, 9))},
    )

Understanding the compilation time
----------------------------------

Compiling the model to achieve speedups takes time. No matter how long the compilation takes, vLLM ensures that all compilation occurs before serving any requests. No compilation will happen during serving time. To understand the compilation time, you can check the vLLM server logs, e.g., ``init engine (profile, create kv cache, warmup model) took 14.18 seconds``. Compare this with and without compilation to gauge the compilation time.

The breakdown of the compilation time is as follows:

- **Dynamo bytecode transform**: Time taken for TorchDynamo to analyze the Python code. Check the logs for ``Dynamo bytecode transform time: 4.60 s``.
- **Inductor graph compilation**: Time taken for the inductor to compile the computation graph into Triton kernels. It includes compilation for a general shape and specific shapes. Check the logs for ``Compiling a graph for general shape takes 14.77 s`` and ``Compiling a graph for shape 1 takes 13.52 s``.
- **Triton kernel compilation**: Time taken for Triton to compile the Triton kernels into GPU kernels. No specific logs are available for this part.

There will also be a log ``torch.compile takes 19.37 s in total``, which sums up the Dynamo bytecode transform time and inductor graph compilation time. To determine the Triton kernel compilation time, use the increase in ``init engine`` time as a reference, then subtract the Dynamo bytecode transform time and inductor graph compilation time.

For example:

.. code-block:: bash

    $ vllm serve meta-llama/Meta-Llama-3-8B
    init engine (profile, create kv cache, warmup model) took 14.18 seconds

    $ vllm serve meta-llama/Meta-Llama-3-8B -O3
    Dynamo bytecode transform time: 4.60 s
    Compiling a graph for general shape takes 14.77 s
    torch.compile takes 19.37 s in total
    init engine (profile, create kv cache, warmup model) took 39.34 seconds

In this example, the increase in ``init engine`` time is 25.16 seconds. The Triton kernel compilation time is calculated as 25.16 - 4.60 - 14.77 = 5.79 seconds.

Exploiting the compilation cache
--------------------------------

When you first compile for a specific shape, such as via ``-O "{'level': 3, 'candidate_compile_sizes': [1]}"``, the compilation for batch size 1 will take some time because Inductor will run autotuning to find the best kernel for this shape. The result of the autotuning will be saved in the Inductor compilation cache. By default, the location is the system temp directory under ``torchinductor_<username>``. You can also set the ``TORCHINDUCTOR_CACHE_DIR`` environment variable to change the location. Check the `PyTorch documentation <https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html#torchinductor-cache-dir>`_ for more information.

The second time you compile for the same shape, the autotuning will be skipped, and the result will be loaded from the cache. This will save a significant amount of compilation time.

Profiling the workload
----------------------

``torch.compile`` primarily enhances the performance of models with static shapes. Since compiling each shape takes time, it is advisable to profile your workload to identify the most common shapes. Then, compile the model for these shapes to achieve optimal performance.

.. note::

    For LLM inference, the batch size (number of tokens we process at every step) is usually the only shape that changes. Therefore, we use the term "shape" and "batch size" interchangeably.

For example, when running ``python benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1``, it is obvious that the main workload involves batch size 1. By compiling the model specifically for batch size 1, you can improve the performance without wasting time on compiling other batch sizes.

.. code-block:: bash

    $ # running a 8B model on H100 with batch size 1, 36.39 seconds of compilation time, 7.7% improvement in latency

    $ python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 --load-format dummy
    init engine (profile, create kv cache, warmup model) took 11.79 seconds
    Avg latency: 0.9704469823899369 seconds

    $ python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 --load-format dummy -O "{'level': 3, 'candidate_compile_sizes': [1]}"
    init engine (profile, create kv cache, warmup model) took 48.18 seconds
    Avg latency: 0.8950413154981409 seconds

    $ # running a 8B model on L4 with batch size 1, 66.54 seconds of compilation time, 4.1 % improvement in latency

    $ python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 --load-format dummy
    init engine (profile, create kv cache, warmup model) took 20.63 seconds
    Avg latency: 7.81603614680001 seconds

    $ python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 --load-format dummy -O "{'level': 3, 'candidate_compile_sizes': [1]}"
    init engine (profile, create kv cache, warmup model) took 87.17 seconds
    Avg latency: 7.495755991366673 seconds

For a dynamic workload, we can use the ``VLLM_LOG_BATCHSIZE_INTERVAL`` environment variable to monitor the batch size distribution:

.. code-block:: bash

    $ # running an 8B model on H100 with various batch sizes, 72.76 seconds of compilation time, 3.9% improvement in throughput
    $
    $ # 1. Run the baseline setting
    $ python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --model meta-llama/Meta-Llama-3-8B --load-format dummy --num-scheduler-steps 64
    init engine (profile, create kv cache, warmup model) took 14.42 seconds
    Throughput: 44.39 requests/s, 22728.17 total tokens/s, 11364.08 output tokens/s

    $ # 2. Run the same setting with profiling
    $ VLLM_LOG_BATCHSIZE_INTERVAL=1.0 python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --model meta-llama/Meta-Llama-3-8B --load-format dummy --num-scheduler-steps 64
    INFO 12-10 15:42:47 forward_context.py:58] Batchsize distribution (batchsize, count): [(256, 769), (232, 215), ...]

    $ # 3. The most common batch sizes are 256 and 232, so we can compile the model for these two batch sizes
    $ python3 benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --model meta-llama/Meta-Llama-3-8B --load-format dummy --num-scheduler-steps 64 -O "{'level': 3, 'candidate_compile_sizes': [232, 256]}"
    init engine (profile, create kv cache, warmup model) took 87.18 seconds
    Throughput: 46.11 requests/s, 23606.51 total tokens/s, 11803.26 output tokens/s

Note that ``torch.compile`` only helps to accelerate the model forwarding. To see the benefit, please make sure GPUs are already busy executing the model; otherwise, the benefit will be hidden because GPUs are idle. That's why we have added ``--num-scheduler-steps 64`` to the command line arguments when benchmarking the throughput.

How does it work?
-----------------

For text-only models, we compile the part of the model from input token IDs to final hidden states, excluding the LM head and logits processing.

For multi-modality models, we compile the text-only part of the model from input embeddings to final hidden states, excluding the vision encoder part and the part of merging multi-modality embeddings with text embeddings.

By carefully compiling the main computation of the model, we can avoid unnecessary compilation time and achieve better performance.

Supported Models
----------------

Most models in vLLM are supported by ``torch.compile``. You should see logs like ``torch.compile takes 19.37 s in total`` in the server logs when you enable ``torch.compile``. If a model does not support ``torch.compile`` but you enable it, there will be a warning ``torch.compile is turned on, but the model does not support it``, and the ``torch.compile`` configurations will be ignored. If you want to get this model supported, please file an issue.

The following models are currently not supported by ``torch.compile``, because their computation graphs are too dynamic to compile:

- ``InternLM2VEForCausalLM``, ``InternVLChatModel``
- cross-attention models like ``MllamaForConditionalGeneration`` and ``BartForConditionalGeneration``

The following models should be supported by ``torch.compile`` in the future, but not supported yet due to bandwidth limitations:

- ``Mamba`` related models
- ``ChameleonModel``, ``ChatGLMModel``, ``DbrxModel``, ``DeepseekModel``, ``MixtralModel``, ``Olmo2Model``, ``Phi3SmallModel``, ``StableLMEpochModel``

Supported Hardware
------------------

Right now ``torch.compile`` is supported on NVIDIA GPUs and AMD GPUs. Google TPUs also use ``torch.compile``, but in a different way.

Feature Compatibility
---------------------

Engine features are naturally compatible with ``torch.compile`` because they do not affect model execution. For example, chunked prefill, prefix caching, and multi-step scheduling are all compatible with ``torch.compile``.

For features that do affect model execution, such as tensor parallel, pipeline parallel, and quantization, we have ensured compatibility with ``torch.compile``. However, there are two features that are currently not compatible with ``torch.compile``:

- **CPU offloading**: This feature is not compatible with ``torch.compile`` at the moment but is expected to be compatible in the future. Track the progress on `this issue <https://github.com/vllm-project/vllm/issues/10612>`__.
- **LoRA serving**: While it can be made compatible with ``torch.compile``, the benefits would be minimal. Models with LoRA adapters primarily use custom (punica) kernels that ``torch.compile`` cannot optimize. Therefore, when LoRA is enabled, ``torch.compile`` will be disabled. For more information, check `this issue <https://github.com/vllm-project/vllm/issues/10617>`__.

Future Directions
-----------------

We plan to further reduce the compilation time of ``torch.compile``. Ideally, when running the same model with the same ``candidate_compile_sizes``, the first run (cold start) will take some time to compile, but subsequent runs (warm start) should take close to zero time to compile. This is a challenging problem, but we are actively working on it with the PyTorch team.

Another direction is to introduce more compilation optimization passes. Currently, we use Inductor to compile the computation graph into Triton kernels directly. There are many graph-level optimizations that can be done before the compilation. Please stay tuned for more updates.
