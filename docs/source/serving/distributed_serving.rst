.. _distributed_serving:

How to decide the distributed inference strategy?
=================================================

Before going into the details of distributed inference and serving, let's first make it clear when to use distributed inference and what are the strategies available. The common practice is:

- **Single GPU (no distributed inference)**: If your model fits in a single GPU, you probably don't need to use distributed inference. Just use the single GPU to run the inference.
- **Single-Node Multi-GPU (tensor parallel inference)**: If your model is too large to fit in a single GPU, but it can fit in a single node with multiple GPUs, you can use tensor parallelism. The tensor parallel size is the number of GPUs you want to use. For example, if you have 4 GPUs in a single node, you can set the tensor parallel size to 4.
- **Multi-Node Multi-GPU (tensor parallel plus pipeline parallel inference)**: If your model is too large to fit in a single node, you can use tensor parallel together with pipeline parallelism. The tensor parallel size is the number of GPUs you want to use in each node, and the pipeline parallel size is the number of nodes you want to use. For example, if you have 16 GPUs in 2 nodes (8GPUs per node), you can set the tensor parallel size to 8 and the pipeline parallel size to 2.

In short, you should increase the number of GPUs and the number of nodes until you have enough GPU memory to hold the model. The tensor parallel size should be the number of GPUs in each node, and the pipeline parallel size should be the number of nodes.

After adding enough GPUs and nodes to hold the model, you can run vLLM first, which will print some logs like ``# GPU blocks: 790``. Multiply the number by ``16`` (the block size), and you can get roughly the maximum number of tokens that can be served on the current configuration. If this number is not satisfying, e.g. you want higher throughput, you can further increase the number of GPUs or nodes, until the number of blocks is enough.

.. note::
    There is one edge case: if the model fits in a single node with multiple GPUs, but the number of GPUs cannot divide the model size evenly, you can use pipeline parallelism, which splits the model along layers and supports uneven splits. In this case, the tensor parallel size should be 1 and the pipeline parallel size should be the number of GPUs.

Distributed Inference and Serving
=================================

vLLM supports distributed tensor-parallel inference and serving. Currently, we support `Megatron-LM's tensor parallel algorithm <https://arxiv.org/pdf/1909.08053.pdf>`_.  We also support pipeline parallel as a beta feature for online serving. We manage the distributed runtime with either `Ray <https://github.com/ray-project/ray>`_ or python native multiprocessing. Multiprocessing can be used when deploying on a single node, multi-node inferencing currently requires Ray.

Multiprocessing will be used by default when not running in a Ray placement group and if there are sufficient GPUs available on the same node for the configured :code:`tensor_parallel_size`, otherwise Ray will be used. This default can be overridden via the :code:`LLM` class :code:`distributed-executor-backend` argument or :code:`--distributed-executor-backend` API server argument. Set it to :code:`mp` for multiprocessing or :code:`ray` for Ray. It's not required for Ray to be installed for the multiprocessing case.

To run multi-GPU inference with the :code:`LLM` class, set the :code:`tensor_parallel_size` argument to the number of GPUs you want to use. For example, to run inference on 4 GPUs:

.. code-block:: python

    from vllm import LLM
    llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
    output = llm.generate("San Franciso is a")

To run multi-GPU serving, pass in the :code:`--tensor-parallel-size` argument when starting the server. For example, to run API server on 4 GPUs:

.. code-block:: console

    $ vllm serve facebook/opt-13b \
    $     --tensor-parallel-size 4

You can also additionally specify :code:`--pipeline-parallel-size` to enable pipeline parallelism. For example, to run API server on 8 GPUs with pipeline parallelism and tensor parallelism:

.. code-block:: console

    $ vllm serve gpt2 \
    $     --tensor-parallel-size 4 \
    $     --pipeline-parallel-size 2 \
    $     --distributed-executor-backend ray

.. note::
    Pipeline parallel is a beta feature. It is only supported for online serving and the ray backend for now, as well as LLaMa and GPT2 style models.

To scale vLLM beyond a single machine, install and start a `Ray runtime <https://docs.ray.io/en/latest/ray-core/starting-ray.html>`_ via CLI before running vLLM:

.. code-block:: console

    $ pip install ray

    $ # On head node
    $ ray start --head

    $ # On worker nodes
    $ ray start --address=<ray-head-address>

After that, you can run inference and serving on multiple machines by launching the vLLM process on the head node by setting :code:`tensor_parallel_size` multiplied by :code:`pipeline_parallel_size` to the number of GPUs to be the total number of GPUs across all machines.

.. warning::
    Please make sure you downloaded the model to all the nodes, or the model is downloaded to some distributed file system that is accessible by all nodes.
