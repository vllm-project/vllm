.. _distributed_serving:

Distributed Inference and Serving
=================================

vLLM supports distributed tensor-parallel inference and serving. Currently, we support `Megatron-LM's tensor parallel algorithm <https://arxiv.org/pdf/1909.08053.pdf>`_. We manage the distributed runtime with either `Ray <https://github.com/ray-project/ray>`_ or python native multiprocessing. Multiprocessing can be used when deploying on a single node, multi-node inferencing currently requires Ray.

Multiprocessing will be used by default when not running in a Ray placement group and if there are sufficient GPUs available on the same node for the configured :code:`tensor_parallel_size`, otherwise Ray will be used. This default can be overridden via the :code:`LLM` class :code:`distributed-executor-backend` argument or :code:`--distributed-executor-backend` API server argument. Set it to :code:`mp` for multiprocessing or :code:`ray` for Ray. It's not required for Ray to be installed for the multiprocessing case.

To run multi-GPU inference with the :code:`LLM` class, set the :code:`tensor_parallel_size` argument to the number of GPUs you want to use. For example, to run inference on 4 GPUs:

.. code-block:: python

    from vllm import LLM
    llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
    output = llm.generate("San Franciso is a")

To run multi-GPU serving, pass in the :code:`--tensor-parallel-size` argument when starting the server. For example, to run API server on 4 GPUs:

.. code-block:: console

    $ python -m vllm.entrypoints.api_server \
    $     --model facebook/opt-13b \
    $     --tensor-parallel-size 4

To scale vLLM beyond a single machine, install and start a `Ray runtime <https://docs.ray.io/en/latest/ray-core/starting-ray.html>`_ via CLI before running vLLM:

.. code-block:: console

    $ pip install ray

    $ # On head node
    $ ray start --head

    $ # On worker nodes
    $ ray start --address=<ray-head-address>

After that, you can run inference and serving on multiple machines by launching the vLLM process on the head node by setting :code:`tensor_parallel_size` to the number of GPUs to be the total number of GPUs across all machines.